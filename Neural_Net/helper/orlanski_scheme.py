"""Differentiable east-Orlanski OBC scheme for NN training.

Copied / adapted from `_orlanski_east` in
`OSSWEM_helpers/OSSWEM_obc_orlanski_2D.py`, re-expressed in **torch** and
vectorized over the PACKED stencil layout produced by the NN data packer
(see NN_offline_training.ipynb). The neural net will replace the analytic
phase-speed estimate; the net itself is NOT implemented here.

Two pieces of the original scheme are provided:
  * `orlanski_east_update`   -- the differentiable UPDATE (the loss core):
        OUTFLOW -> 2D Orlanski radiation with the network's (cx, cy);
        INFLOW  -> nudge toward the recorded external column with a FIXED
                   coefficient alpha_in (chosen before the run, NOT learned).
  * `orlanski_east_analytic` -- the ANALYTIC estimator (the original phase-speed
        ratio + inflow flag), kept as the baseline to beat, as pretraining
        targets, and for the Phase-1.6 round-trip check.

Packed stencil X has shape (..., 8); columns are RAW physical values:        
    0: (b-1, j-1) @ n     3: (b, j-1) @ n      6: (b-1, j) @ n+1   = pim1     
    1: (b-1, j  ) @ n     4: (b, j  ) @ n      7: (b-2, j) @ n+1   = pim2     
    2: (b-1, j+1) @ n     5: (b, j+1) @ n          (4 = phi_prev_b_j)         

Everything is elementwise + torch.where/clamp, so the functions run on torch
tensors (with autograd) or on plain numpy via `torch.as_tensor`.
"""
import torch

# packed-column indices
BM1_JM, BM1_J, BM1_JP = 0, 1, 2   # (b-1, j-1/j/j+1) @ n
B_JM,   B_J,   B_JP   = 3, 4, 5   # (b,   j-1/j/j+1) @ n   (B_J = phi_prev_b_j)
PIM1,   PIM2          = 6, 7      # (b-1,j) , (b-2,j) @ n+1

# [BASEFIX] the small-denominator floor hard-coded in the DEPLOYED estimator
# (`eps_denom` in `_orlanski_east` and `_phase_inflow_np`,
# OSSWEM_helpers/Remove_periodicity_2D_NS_orlanski.py).  Modules that use a different
# value (e.g. 2layer_jet/..._2layer.py uses 1e-5) must pass their own.
EPS_DENOM_KERNEL = 1e-3


def orlanski_east_update(X, cx, cy, inflow=None, phi_ext=None, alpha_in=1.0,
                         inflow_law='const_alpha'):   # [RXNUDGE] TRAINING-side inflow law selector
    """East-Orlanski boundary update (the loss core), faithful to `_orlanski_east`.

    OUTFLOW (inflow False): 2D Orlanski radiation with the network's (cx, cy):
        dy_b  = (phi_prev_b_j - b_jm)  if cy >= 0  else  (b_jp - phi_prev_b_j)
        phi_b = (phi_prev_b_j + cx*pim1 - cy*dy_b) / (1 + cx)

    INFLOW (inflow True): nudge column b toward the recorded external value with a
    FIXED coefficient alpha_in -- a hyperparameter chosen BEFORE the run, NOT learned;
    cx, cy are unused on inflow:
        phi_b = phi_prev_b_j + alpha_in * (phi_ext - phi_prev_b_j)
              = (1 - alpha_in) * phi_prev_b_j + alpha_in * phi_ext
        alpha_in = 1 -> hard prescribe   (phi_b = phi_ext)
        alpha_in = 0 -> keep dynamic     (phi_b = phi_prev_b_j)
        0 < alpha_in < 1 -> soft nudge toward phi_ext

    X        : (..., 8) raw physical stencil (torch tensor)
    cx, cy   : (...) network phase speed; used on OUTFLOW points only
    inflow   : (...) bool mask. None -> treat every point as outflow (pure radiation)
    phi_ext  : (...) external (recorded control) value at column b; needed if any inflow
    alpha_in : FIXED scalar nudging coefficient in [0, 1]  (ignored when inflow_law='rx_nudge')

    inflow_law : [RXNUDGE] which formula the INFLOW branch uses.  TRAINING-ONLY knob -- the
        deployed scheme (_orlanski_east / _obc_update_np) always uses the constant-alpha nudge.
        'const_alpha' (default, unchanged, bitwise-identical to before):
              phi_b = phi_prev_b_j + alpha_in * (phi_ext - phi_prev_b_j)
            cx does NOT enter, so inflow points carry ZERO gradient to the network.
        'rx_nudge':
              phi_b = (phi_prev_b_j - cx * phi_ext) / (1 - cx),        cx < 0 on inflow
                    = (1 - a) * phi_prev_b_j + a * phi_ext,   a = -cx / (1 - cx)
            The nudging coefficient is no longer constant: a runs 0 -> 1 as cx runs 0- -> -inf,
            so the update is always a proper convex combination of the interior value and the
            exterior (it can never overshoot, for any cx < 0).  alpha_in is UNUSED here.
            Inflow points now DO carry gradient:  d phi_b / d cx = (phi_prev_b_j - phi_ext)/(1-cx)^2.
    """
    if inflow_law not in ('const_alpha', 'rx_nudge'):                    # [RXNUDGE]
        raise ValueError(f"inflow_law must be 'const_alpha' or 'rx_nudge', got {inflow_law!r}")
    phi_prev_b_j = X[..., B_J]    # (b, j)   @ n   = phi_prev_b_j               
    pim1         = X[..., PIM1]   # (b-1, j) @ n+1                              
    b_jm         = X[..., B_JM]   # (b, j-1) @ n                                
    b_jp         = X[..., B_JP]   # (b, j+1) @ n                                

    def _radiate(cx_, cy_, phi_, pim1_, b_jm_, b_jp_):  
        dy_b = torch.where(cy_ >= 0.0, phi_ - b_jm_, b_jp_ - phi_)
        return (phi_ + cx_ * pim1_ - cy_ * dy_b) / (1.0 + cx_)                

    # if inflow is None or phi_ext is None:   
    # Outflow
    if inflow is None:                                   
        return _radiate(cx, cy, phi_prev_b_j, pim1, b_jm, b_jp)
                                                      

    # inflow: nudge toward the recorded external column
    out = ~inflow
    if inflow_law == 'const_alpha':
        nudge = phi_prev_b_j + alpha_in * (phi_ext - phi_prev_b_j)   # no cx,cy dependence -> safe everywhere
    else:                                                            # [RXNUDGE] 'rx_nudge'
        # phi_b = (phi_prev_b_j - cx*phi_ext) / (1 - cx),  i.e. alpha_eff = -cx/(1-cx) in [0,1).
        #
        # cx MUST be zeroed on the outflow rows first.  masked_scatter below builds the result
        # FROM this tensor, so the expression is evaluated at EVERY point, outflow included --
        # and an outflow row with cx == 1 (the CFL=1 case, and the head is an unbounded
        # nn.Linear) would divide by zero -> inf, then masked_scatter's backward multiplies
        # that by an exact 0 -> NaN in cx.grad, while the forward loss still prints a normal
        # number.  With cx_in the denominator is exactly 1 on outflow rows and > 1 on inflow
        # rows, and those outflow values are discarded by the scatter anyway.
        cx_in = torch.where(inflow, cx, torch.zeros_like(cx))         # [RXNUDGE]
        nudge = (phi_prev_b_j - cx_in * phi_ext) / (1.0 - cx_in)      # [RXNUDGE]
    rad_out = _radiate(cx[out], cy[out], phi_prev_b_j[out], pim1[out], b_jm[out], b_jp[out])
    return nudge.masked_scatter(out, rad_out)         

def orlanski_east_update_radin(X, cx, cy, inflow, phi_ext, phi_ext_out, alpha_in=0.0):   # [IM]
    """[IM] Differentiable update with RADIATION-INFLOW on inflow points.

    OUTFLOW (inflow False): unchanged 2D Orlanski radiation -- IDENTICAL to
    `orlanski_east_update`, so any outflow-only metric is scheme-independent.
    INFLOW (inflow True): upwinding flips east, so the upstream point is the EXTERIOR
    column b+1 and (cx, cy) now enter the inflow branch too (inflow samples therefore
    carry gradient to the network, unlike the fixed-alpha_in nudge):
        phi_b = (phi_prev_b_j - cx*phi_ext_out - cy*dy_b) / (1 - cx),      cx < 0
    then the same fixed alpha_in nudge toward phi_ext (nudging_mode=1 analogue).
    cx is zeroed on outflow rows before the division, so 1 - cx >= 1 everywhere and a
    cx = -1 row cannot produce a NaN/inf that would poison the backward pass; those rows
    are overwritten by the radiation branch via masked_scatter.

    phi_ext_out : (...) exterior value at column b+1 at n+1 (the same time level as pim1).
    alpha_in    : FIXED nudging coefficient; 0.0 (default) = pure radiation-inflow.

    Equivalence: matches `_obc_update_np(inflow_mode=1)` EXACTLY.  It matches the deployed
    kernel `_orlanski_east(inflow_mode=1, nudging_mode=1)` only up to the r_max CFL clamp,
    which the kernel applies to rx and ry and this function deliberately does not -- the
    same unclamped-training-baseline convention as `orlanski_east_analytic` above."""
    phi_prev_b_j = X[..., 4]; pim1 = X[..., 6]; b_jm = X[..., 3]; b_jp = X[..., 5]       # [IM]
    dy_b  = torch.where(cy >= 0.0, phi_prev_b_j - b_jm, b_jp - phi_prev_b_j)            # [IM]
    cx_in = torch.where(inflow, cx, torch.zeros_like(cx))                               # [IM]
    val   = (phi_prev_b_j - cx_in * phi_ext_out - cy * dy_b) / (1.0 - cx_in)            # [IM]
    val   = val + alpha_in * (phi_ext - val)                                            # [IM] extra col-b nudge
    out   = ~inflow                                                                     # [IM]
    rad   = (phi_prev_b_j[out] + cx[out] * pim1[out] - cy[out] * dy_b[out]) / (1.0 + cx[out])   # [IM]
    return val.masked_scatter(out, rad)                                                                         


def orlanski_east_update_1d(X, cx, inflow=None, phi_ext=None, alpha_in=1.0):  # [OBC-1D]
    """1D east-Orlanski boundary update: the 2D update with the tangential cy term dropped.  # [OBC-1D]

    OUTFLOW (inflow False): 1D Orlanski radiation with the network's cx (no cy):  # [OBC-1D]
        phi_b = (phi_prev_b_j + cx*pim1) / (1 + cx)  # [OBC-1D]
    INFLOW (inflow True): same fixed-alpha_in nudge as the 2D scheme; cx unused.  # [OBC-1D]
        phi_b = phi_prev_b_j + alpha_in * (phi_ext - phi_prev_b_j)  # [OBC-1D]

    X        : (..., 8) raw physical stencil (torch tensor)  # [OBC-1D]
    cx       : (...) network phase speed; used on OUTFLOW points only  # [OBC-1D]
    inflow   : (...) bool mask. None -> treat every point as outflow (pure radiation)  # [OBC-1D]
    phi_ext  : (...) external (recorded control) value at column b; needed if any inflow  # [OBC-1D]
    alpha_in : FIXED scalar nudging coefficient in [0, 1]  # [OBC-1D]
    """
    phi_prev_b_j = X[..., B_J]    # (b, j)   @ n   = phi_prev_b_j  # [OBC-1D]
    pim1         = X[..., PIM1]   # (b-1, j) @ n+1                 # [OBC-1D]

    rad = (phi_prev_b_j + cx * pim1) / (1.0 + cx)   # [OBC-1D] 1D radiation (no cy*dy term)

    if inflow is None or phi_ext is None:           # [OBC-1D]
        return rad                                  # [OBC-1D]

    nudge = phi_prev_b_j + alpha_in * (phi_ext - phi_prev_b_j)  # [OBC-1D]
    return torch.where(inflow, nudge, rad)          # [OBC-1D]


def orlanski_east_analytic_1d(X):  # [OBC-1D]
    """Analytic per-point 1D phase speed cx + inflow flag from the stencil.  # [OBC-1D]

    1D reduction of `orlanski_east_analytic`: |grad phi|^2 -> dphi_x^2, so  # [OBC-1D]
    cx = -dphi_t/dphi_x. Returns (cx, inflow): cx in [0,1] (same clip on outflow);  # [OBC-1D]
    inflow == (raw cx < 0).  # [OBC-1D]

    [BASEFIX] NOTE: this 1D variant still has NO eps_denom floor (unlike the 2D function
    above and the deployed kernel).  Its hard clamp cx -> [0, 1] bounds the blow-up, so the
    baseline it produces is bounded but still not bit-for-bit the deployed 1D estimator.
    Left unchanged deliberately: no caller in this project uses it right now, and changing
    it would shift previously recorded 1D baselines.
    """
    pim1, pim2 = X[..., PIM1], X[..., PIM2]   # [OBC-1D]
    bm1_j      = X[..., BM1_J]                # [OBC-1D]

    dphi_t = pim1 - bm1_j        # phi^{n+1}_{b-1} - phi^n_{b-1}      # [OBC-1D]
    dphi_x = pim1 - pim2         # phi^{n+1}_{b-1} - phi^{n+1}_{b-2}  # [OBC-1D]

    denom   = dphi_x * dphi_x                                          # [OBC-1D] |grad phi|^2 in 1D
    safe    = denom > 0.0                                              # [OBC-1D]
    denom_s = torch.where(safe, denom, torch.ones_like(denom))        # [OBC-1D] avoid 0-div
    cx = torch.where(safe, -dphi_t * dphi_x / denom_s, torch.zeros_like(denom))  # [OBC-1D]

    inflow = cx < 0.0                                                  # [OBC-1D]
    cx = torch.clamp(cx, 0.0, 1.0)                # inflow->0; outflow->[0,1]  # [OBC-1D]
    return cx, inflow                                                 # [OBC-1D]


def orlanski_east_analytic(X, eps_denom=EPS_DENOM_KERNEL, r_max=None,   # [BASEFIX] guard + clamp knobs
                           obc_order=1):   # [O2] MUST be 1: the packed 8-pt stencil has no b-3 column / n-1 level, so the kernel's 2nd-order estimator cannot be reproduced here. Hard raise, no silent fallback.
    """Analytic per-point phase speed (cx, cy) + inflow flag from the stencil.

    [O2] obc_order: only 1 is supported.  The deployed kernel (`_orlanski_east`) gained a
    2nd-order estimator (obc_order=2, reads column b-3 and time level n-1), but this packed
    stencil (indices 0-7) carries neither, so a training baseline CANNOT match an
    obc_order=2 run -- passing 2 raises instead of silently degrading ([BASEFIX] lesson).

    Faithful copy of the estimation branch of the 2D `_orlanski_east`. Returns
    (cx, cy, inflow); inflow == (raw cx < 0), i.e. taken BEFORE any clamp, exactly as
    the kernel does.

    [BASEFIX] The two numerical safeguards of the deployed kernel are now reproduced
    here, because a baseline that lacks them is NOT the scheme the run compares against:

      eps_denom : small-|grad phi| floor, r_d = 1/max(denom, eps_denom).  Without it the
          ratio divides by a near-zero on quiet rows; measured on a 2-layer run this
          fires on 75% of the u samples and 73% of v (h: 0.0%), inflating the reported
          'analytic baseline' loss by 1.45x / 1.28x and making it trivially beatable.
          Default = the kernel's value.  Pass eps_denom=None for the historical
          unguarded form (reproduces every baseline loss recorded before this change).
      r_max   : the kernel's CFL clamp on |cx|, |cy| (run_obc's r_max / R_MAX_ANALYTIC).
          None (default) leaves the phase speed unclamped; pass the run's value to make
          this function bit-for-bit the deployed estimator.

    Use as the analytic baseline, as pretraining targets, or to supply the inflow mask
    to `orlanski_east_update`.
    """
    if int(obc_order) != 1:                                                        # [O2]
        raise NotImplementedError('[O2] orlanski_east_analytic supports obc_order=1 only '
                                  '(8-pt stencil lacks b-3 / n-1); an obc_order=2 baseline '
                                  'must come from run_obc itself.')                # [O2]
    pim1, pim2          = X[..., PIM1], X[..., PIM2]
    bm1_jm, bm1_j, bm1_jp = X[..., BM1_JM], X[..., BM1_J], X[..., BM1_JP]

    dphi_t = pim1 - bm1_j        # phi^{n+1}_{b-1} - phi^n_{b-1}
    dphi_x = pim1 - pim2         # phi^{n+1}_{b-1} - phi^{n+1}_{b-2}
    cen    = bm1_jp - bm1_jm     # central diff in j @ n
    dphi_y = torch.where(dphi_t * cen > 0.0,                                    # upwind in j
                         bm1_j - bm1_jm, bm1_jp - bm1_j)

    denom = dphi_x * dphi_x + dphi_y * dphi_y          # |grad phi|^2
    if eps_denom is None:                                                          # [BASEFIX] historical form
        safe  = denom > 0.0
        denom_s = torch.where(safe, denom, torch.ones_like(denom))                  # avoid 0-div
        rx = torch.where(safe, -dphi_t * dphi_x / denom_s, torch.zeros_like(denom))
        ry = torch.where(safe, -dphi_t * dphi_y / denom_s, torch.zeros_like(denom))
    else:                                                                          # [BASEFIX] kernel's floor
        # [EPSREL] eps_denom may now be a TENSOR broadcastable over denom (per (field, layer)
        # [EPSREL] floors), not just a scalar -- torch.clamp(min=...) takes only a number, so
        # [EPSREL] use torch.maximum for the array case.  The training baseline MUST use the
        # [EPSREL] same table the deployed kernel uses, or it is [BASEFIX] all over again.
        if torch.is_tensor(eps_denom):                                             # [EPSREL]
            r_d = 1.0 / torch.maximum(denom, eps_denom.to(denom.dtype).to(denom.device))  # [EPSREL]
        else:                                                                      # [EPSREL]
            r_d = 1.0 / torch.clamp(denom, min=float(eps_denom))                   # [BASEFIX]
        rx = -dphi_t * dphi_x * r_d                                                # [BASEFIX]
        ry = -dphi_t * dphi_y * r_d                                                # [BASEFIX]

    inflow = rx < 0.0                             # from the RAW cx, before the clamp
    if r_max is not None:                                                          # [BASEFIX] kernel's CFL clamp
        _rm = float(r_max)                                                         # [BASEFIX] outflow: cx<=r_max
        rx = torch.clamp(rx, -_rm, _rm)                                            # [BASEFIX] inflow(mode 1): cx>=-r_max
        ry = torch.clamp(ry, -_rm, _rm)                                            # [BASEFIX]
    return rx, ry, inflow
