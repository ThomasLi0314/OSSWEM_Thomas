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

Packed stencil X has shape (..., 8); columns are RAW physical values:        # [NN-OBC]
    0: (b-1, j-1) @ n     3: (b, j-1) @ n      6: (b-1, j) @ n+1   = pim1     # [NN-OBC]
    1: (b-1, j  ) @ n     4: (b, j  ) @ n      7: (b-2, j) @ n+1   = pim2     # [NN-OBC]
    2: (b-1, j+1) @ n     5: (b, j+1) @ n          (4 = phi_prev_b_j)         # [NN-OBC]

Everything is elementwise + torch.where/clamp, so the functions run on torch
tensors (with autograd) or on plain numpy via `torch.as_tensor`.
"""
import torch

# packed-column indices                                                       # [NN-OBC]
BM1_JM, BM1_J, BM1_JP = 0, 1, 2   # (b-1, j-1/j/j+1) @ n                       # [NN-OBC]
B_JM,   B_J,   B_JP   = 3, 4, 5   # (b,   j-1/j/j+1) @ n   (B_J = phi_prev_b_j) # [NN-OBC]
PIM1,   PIM2          = 6, 7      # (b-1,j) , (b-2,j) @ n+1                    # [NN-OBC]


def orlanski_east_update(X, cx, cy, inflow=None, phi_ext=None, alpha_in=1.0):
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
    alpha_in : FIXED scalar nudging coefficient in [0, 1]
    """
    phi_prev_b_j = X[..., B_J]    # (b, j)   @ n   = phi_prev_b_j               # [NN-OBC]
    pim1         = X[..., PIM1]   # (b-1, j) @ n+1                              # [NN-OBC]
    b_jm         = X[..., B_JM]   # (b, j-1) @ n                                # [NN-OBC]
    b_jp         = X[..., B_JP]   # (b, j+1) @ n                                # [NN-OBC]

    # outflow: 2D Orlanski radiation (upwind tangential diff selected by sign(cy))  # [NN-OBC]
    dy_b = torch.where(cy >= 0.0, phi_prev_b_j - b_jm, b_jp - phi_prev_b_j)    # [NN-OBC]
    rad  = (phi_prev_b_j + cx * pim1 - cy * dy_b) / (1.0 + cx)                 # [NN-OBC]

    if inflow is None or phi_ext is None:                                     # [NN-OBC]
        return rad                                                            # [NN-OBC] pure radiation (no inflow handling)

    # inflow: nudge toward the recorded external column with the FIXED alpha_in  # [NN-OBC]
    nudge = phi_prev_b_j + alpha_in * (phi_ext - phi_prev_b_j)                # [NN-OBC]
    return torch.where(inflow, nudge, rad)                                    # [NN-OBC]


def orlanski_east_analytic(X):
    """Analytic per-point phase speed (cx, cy) + inflow flag from the stencil.

    Faithful copy of the estimation branch of the 2D `_orlanski_east`,
    vectorized. Returns (cx, cy, inflow): cx in [0,1], cy in [-1,1] (same clip as
    the scheme on outflow); inflow == (raw cx < 0). Use as the analytic baseline,
    as pretraining targets, or to supply the inflow mask to `orlanski_east_update`.
    """
    pim1, pim2          = X[..., PIM1], X[..., PIM2]                           # [NN-OBC]
    bm1_jm, bm1_j, bm1_jp = X[..., BM1_JM], X[..., BM1_J], X[..., BM1_JP]      # [NN-OBC]

    dphi_t = pim1 - bm1_j        # phi^{n+1}_{b-1} - phi^n_{b-1}               # [NN-OBC]
    dphi_x = pim1 - pim2         # phi^{n+1}_{b-1} - phi^{n+1}_{b-2}           # [NN-OBC]
    cen    = bm1_jp - bm1_jm     # central diff in j @ n                       # [NN-OBC]
    dphi_y = torch.where(dphi_t * cen > 0.0,                                   # [NN-OBC] upwind in j
                         bm1_j - bm1_jm, bm1_jp - bm1_j)                       # [NN-OBC]

    denom = dphi_x * dphi_x + dphi_y * dphi_y          # |grad phi|^2          # [NN-OBC]
    safe  = denom > 0.0                                                        # [NN-OBC]
    denom_s = torch.where(safe, denom, torch.ones_like(denom))                 # [NN-OBC] avoid 0-div
    cx = torch.where(safe, -dphi_t * dphi_x / denom_s, torch.zeros_like(denom))# [NN-OBC]
    cy = torch.where(safe, -dphi_t * dphi_y / denom_s, torch.zeros_like(denom))# [NN-OBC]

    inflow = cx < 0.0                                  # rx<0 -> inflow        # [NN-OBC]
    cx = torch.clamp(cx, 0.0, 1.0)                     # inflow->0; outflow->[0,1] # [NN-OBC]
    cy = torch.clamp(cy, -1.0, 1.0)                                            # [NN-OBC]
    return cx, cy, inflow
