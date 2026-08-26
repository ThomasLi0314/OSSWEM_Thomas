import numpy as np
from numba import njit, prange, get_num_threads

# This is a 2D scheme. 

# --- Numba-JIT shift helpers ---

@njit(cache=True)
def _nb_im1(a):
    """Shift a by -1 in the i-direction (last axis, periodic)."""
    result = np.empty_like(a)
    result[..., 0] = a[..., -1]
    result[..., 1:] = a[..., :-1]
    return result

@njit(cache=True)
def _nb_ip1(a):
    """Shift a by +1 in the i-direction (last axis, periodic)."""
    result = np.empty_like(a)
    result[..., -1] = a[..., 0]
    result[..., :-1] = a[..., 1:]
    return result

@njit(cache=True)
def _nb_jm1(a):
    """Shift a by -1 in the j-direction (second-to-last axis, periodic)."""
    result = np.empty_like(a)
    result[..., 0, :] = a[..., -1, :]
    result[..., 1:, :] = a[..., :-1, :]
    return result

@njit(cache=True)
def _nb_jp1(a):
    """Shift a by +1 in the j-direction (second-to-last axis, periodic)."""
    result = np.empty_like(a)
    result[..., -1, :] = a[..., 0, :]
    result[..., :-1, :] = a[..., 1:, :]
    return result

# --- Numba-JIT averaging helpers ---

@njit(cache=True)
def _nb_h2u(a):
    """Averages from h- to u- points. Also does v- to q-."""
    return 0.5 * ( a + _nb_im1(a) )

@njit(cache=True)
def _nb_q2v(a):
    """Averages from q- to v- points."""
    return 0.5 * ( a + _nb_ip1(a) )

@njit(cache=True)
def _nb_u2q(a):
    """Averages from u- to q- points."""
    return 0.5 * ( a + _nb_jm1(a) )

@njit(cache=True)
def _nb_q2u(a):
    """Averages from q- to u- points."""
    return 0.5 * ( a + _nb_jp1(a) )

# --- Numba-JIT difference helpers ---

@njit(cache=True)
def _nb_dih(a):
    """Difference h- points to u- points. Also does v- to q-."""
    return a - _nb_im1(a)

@njit(cache=True)
def _nb_djh(a):
    """Difference h- points to v- points. Also does u- to q-."""
    return a - _nb_jm1(a)

# --- Numba-JIT physical functions ---

@njit(cache=True)
def _nb_vxuy(u, v, rdx, rdy):
    """Calculate components of relative vorticity"""
    return _nb_dih( v ) * rdx, _nb_djh( u ) * rdy

######################################## Orlanski OBC ########################################
# [RMAX] Default outflow CFL clamp for the east Orlanski scheme.  This module has always
# [RMAX] used 1/50; the NN driver module (NP_OSSWEM_orlanski_2D_NN_1step) used 8/50.  Keeping
# [RMAX] 1/50 as the default means every existing caller of THIS module is unchanged, while
# [RMAX] the NN notebooks can pass r_max=8/50 to reproduce their previous results exactly.
R_MAX_DEFAULT = 1.0 / 50.0                                                          # [RMAX]

# OBC Orlanski Implementation [OBC - Orlanski]
# This is a 1D SCHEME
@njit(cache=True)
def _orlanski_east(phi, phi_prev, b_obc, rx_out,
                   nudging_mode, phi_ext, alpha_in,
                   rxy_out,
                   inflow_persistent,
                   ry_on,
                   y_periodic,                           # [YP] tangential stencil: wrap vs mirror ghost
                   r_max,                                # [RMAX] outflow CFL clamp on |rx|,|ry| (was hard-coded 1/50)
                   inflow_mode, phi_ext_out,             # [IM] inflow-scheme knob + col-(b+1) exterior data @ n+1; REQUIRED (numba cannot type a None default against phi_ext_out[k,j]) -- pass a dummy (nk,nj) array when inflow_mode == 0
                   use_nn_rx,                            # [NNRX] 1 -> use pre-computed rx_nn,ry_nn instead of computing analytically
                   rx_nn, ry_nn,                         # [NNRX] (nk,nj) pre-computed per-row (rx,ry); read only when use_nn_rx==1; pass dummies when 0
                   alpha_from_rx,                        # [ALPHARX] 1 -> inflow-nudge coefficient = min(|rx|,1) per row instead of the fixed alpha_in (inflow_mode==0 nudging branch only)
                   obc_order,                            # [O2] 1 -> classic 1st-order differences (unchanged); 2 -> 2nd-order one-sided D_t/D_x (needs phi_prev slot 2 = phi^{n-1}_{b-1} and column b-3); dphi_y and the update law stay 1st order
                   eps_arr):                             # [EPSREL] (nk,) small-|grad phi|^2 floor for THIS field, one value per layer. Historically a hard-coded scalar 1e-3 for every field and layer -- but rx = -dphi_t*dphi_x/denom is scale-invariant in phi, so an ABSOLUTE floor is not: it binds on ~55%/95% of u,v rows (layer 0/1) and 0% of h rows, because velocity increments are ~250x (L0) to ~1400x (L1) smaller than thickness increments. A floored row has |rx| shrunk toward 0, i.e. the boundary PERSISTS instead of radiating. Pass np.full(nk, 1e-3) to reproduce the historical behaviour exactly.
    """
        b here is the column for implementing the orlanski.

          0 -> PRESCRIBE scheme: on inflow HARD-prescribe column b to the     
               recorded external solution, phi_b^{n+1} = phi_ext
          1 -> NUDGING scheme: on inflow SOFT-prescribe column b toward the recorded external solution phi_ext,                         
                 phi_b^{n+1} = phi_b^n + alpha_in*(phi_ext - phi_b^n),       
            This is the paper's adaptive scheme (alpha_in=1 == hard prescribe).   
        phi_ext : (nk, nj) external (recorded) value of THIS field at column
               b for the current step; read on inflow in BOTH modes.  
        phi_ext_out : (nk, nj) external value of THIS field at column b_obc + 1 at time  
               n+1 (the step being computed); only read when inflow_mode == 1.           
        inflow_mode : [IM] inflow-branch selector.                                       
               0 -> unchanged: PRESCRIBE / NUDGING toward phi_ext (per nudging_mode).    
               1 -> radiation-INFLOW scheme: implicit upwind of the radiation            
                    equation with the upstream point at column b+1,                      
                      (phi_b^{n+1}-phi_b^n) + rx*(phi_ext_out - phi_b^{n+1}) + ry*dy_b = 0  
                      ->  phi_b^{n+1} = (phi_b^n - rx*phi_ext_out - ry*dy_b) / (1 - rx),  rx < 0.  
                    A convex blend of persistence and phi_ext_out (weights               
                    1/(1+|rx|), |rx|/(1+|rx|)): bounded, no CFL restriction, and         
                    continuous with the outflow branch at rx -> 0.  |rx| and |ry|        
                    are clamped at r_max (symmetric with the outflow clamp).  When       
                    nudging_mode == 1 the result is additionally nudged toward the       
                    col-b external data: phi_b += alpha_in*(phi_ext - phi_b).            
        alpha_in : inflow nudging coefficient in (0,1]; only used if nudging.
        Outflow is ALWAYS pure radiation in both modes (no nudge applied).  
        ry_on : 1 -> full 2D scheme.  0 -> 1D Orlanski limit: the tangential
               difference is dropped from BOTH the phase-speed denominator and
               the update, so r_x collapses to the classic -phi_t/phi_x and
               r_y == 0.
        r_max : [RMAX] outflow CFL clamp: rx is capped at r_max and ry at +/-r_max.
               This module historically hard-coded 1/50; the NN driver module used
               8/50.  It is a parameter so both can be reproduced exactly -- the
               default (1/50) leaves every existing caller of THIS module unchanged.
               NOTE: the numpy/torch analytic paths used to build NN training
               baselines (_phase_inflow_np here, orlanski_east_analytic in
               helper/orlanski_scheme.py) are deliberately left UNCLAMPED, so a
               training baseline is not directly comparable to a deployed run.

        Input
        phi_prev = phi[k, all j, b-1:b] here k for layer
    """
    nk, nj, ni = phi.shape
    # Initial two variables for diagonise, sum is used for average, max is used for maximum.
    maxr = 0.0;    sumr = 0.0

    for k in range(nk):
        for j in range(nj):
            # [NP] free-slip N/S wall: one-sided tangential difference at the wall rows (was y-periodic)
            jm = j - 1 if j > 0 else ( nj - 1 if y_periodic else 0 )              # [YP] wrap / mirror ghost (south)
            jp = j + 1 if j < nj - 1 else ( 0 if y_periodic else nj - 1 )         # [YP] wrap / mirror ghost (north)
            pim1 = phi[k, j, b_obc-1] #phi^{n+1}_{b-1, j}
            pim2 = phi[k, j, b_obc-2] #phi^{n+1}_{b-2, j}

            if obc_order == 2:                                                          # [O2]
                # [O2] 2nd-order one-sided differences, SCALED BY 1/2 so they stay
                # [O2] ~ dt*phi_t and ~ dx*phi_x: 3f0 - 4f-1 + f-2 = 2*Delta*f' + O(D^3).
                # [O2] Without the 1/2 the 2D denominator dphi_x^2 + dphi_y^2 would mix a
                # [O2] doubled x-difference with an unscaled y-difference, silently changing
                # [O2] eps_denom, r_max, and the inflow test.  dphi_y stays 1st order.
                pim3 = phi[k, j, b_obc-3]                                               # [O2] phi^{n+1}_{b-3,j}
                dphi_t = (3.0 * pim1 - 4.0 * phi_prev[k, j, 0] + phi_prev[k, j, 2]) * 0.5   # [O2] slot 2 = phi^{n-1}_{b-1,j}
                dphi_x = (3.0 * pim1 - 4.0 * pim2 + pim3) * 0.5                         # [O2]
            else:                                                                       # [O2]
                dphi_t = pim1 - phi_prev[k, j, 0]  # - phi^n_{b-1,j}
                dphi_x = pim1 - pim2

            # centeral difference
            cen = phi_prev[k, jp, 0] - phi_prev[k, jm, 0] #phi^n_{b-1, j+1} - phi^n_{b-1, j-1}
            if dphi_t * cen > 0.0:
                dphi_y = phi_prev[k, j, 0] - phi_prev[k, jm, 0] # phi_{b-1, j} - phi_{b-1, j-1}
            else:
                dphi_y = phi_prev[k, jp, 0] - phi_prev[k, j, 0] # phi_{b-1, j+1} - phi_{b-1, j}

            if ry_on == 1:                                      
                denom = dphi_x*dphi_x + dphi_y*dphi_y # THis is the norm of gradient
            else:                                               
                denom = dphi_x*dphi_x                           

            # [NNRX] use pre-computed (rx,ry) from an external source (e.g. neural network)
            # when available, skipping the pointwise analytic estimation entirely.
            # inflow_persistent still applies when use_nn_rx==1: it gates rx on the
            # pre-computed value, same as the analytic path.
            flg = 0.0                                    # [O2] diag bitmask: 1=denom floor, 2=rx clamp, 4=ry clamp
            if use_nn_rx == 0:
                eps_denom = eps_arr[k]                   # [EPSREL] per-layer floor for this field (was hard-coded 1e-3)
                if denom < eps_denom:                    # [O2] record the floor binding
                    r_d = 1.0 / eps_denom
                    flg = 1.0                            # [O2]
                else:
                    r_d = 1.0 / denom
                rx = -dphi_t * dphi_x * r_d
                ry = -dphi_t * dphi_y * r_d if ry_on == 1 else 0.0
            else:
                rx = rx_nn[k, j]
                ry = ry_nn[k, j]

            if inflow_persistent : 
                if rx < 0.0:
                    rx = 0.0
                    ry = 0.0                           

            # inflow 
  
            if rx < 0.0:
                if inflow_mode == 0:
                    if nudging_mode == 1:
                        # phi_b^{n+1} = alpha * phi^{ext} + phi_{b,j}^n - \alpha \phi_{b,j}^n
                        # Here alpha = \Delta t / \tau_in. So when \alpha = 1 it is a hard prescrib.
                        # [ALPHARX] alpha_from_rx==1: the nudge coefficient is min(|rx|,1) per row --
                        # [ALPHARX] the (NN) phase speed sets how hard the exterior is pulled in, so
                        # [ALPHARX] the inflow branch carries gradient w.r.t. rx (fixed alpha has none).
                        if alpha_from_rx == 1:                                       # [ALPHARX]
                            a_eff = -rx if -rx < 1.0 else 1.0                        # [ALPHARX] min(|rx|,1), rx<0 here
                        else:                                                        # [ALPHARX]
                            a_eff = alpha_in                                         # [ALPHARX] unchanged default
                        phi[k, j, b_obc] = phi_prev[k, j, 1] + a_eff * (phi_ext[k, j] - phi_prev[k, j, 1])
                        phi[k, j, b_obc + 1] = phi[k,j, b_obc]
                        if b_obc + 2 < ni: phi[k, j, b_obc + 2] = phi[k,j, b_obc]  
                    else:
                        phi[k, j, b_obc] = phi_ext[k, j]
                        phi[k, j, b_obc + 1] = phi[k,j, b_obc]
                        if b_obc + 2 < ni: phi[k, j, b_obc + 2] = phi[k,j, b_obc]  
                        
                if inflow_mode == 1:
                    if rx < -r_max : rx = -r_max         
                    if ry > r_max : ry = r_max            
                    elif ry < -r_max: ry = -r_max          

                    if ry >= 0.0:                                                                     
                        dy_b = phi_prev[k, j, 1] - phi_prev[k, jm, 1] # phi^n_{b,j} - \phi_{b, j-1}   
                    else:                                                                             
                        dy_b = phi_prev[k, jp, 1] - phi_prev[k, j, 1]                                 

                    phi_new = (phi_prev[k, j, 1] - rx * phi_ext_out[k, j] - ry * dy_b) / (1.0 - rx)   
                    if nudging_mode == 1:                                                             
                        phi_new = phi_new + alpha_in * (phi_ext[k, j] - phi_new)                      
                    phi[k, j, b_obc] = phi_new
                    phi[k, j, b_obc + 1] = phi[k,j, b_obc]
                    if b_obc + 2 < ni:                       # [TRUNC] guard, as the other branches:
                        phi[k, j, b_obc + 2] = phi[k,j, b_obc]   # b = ni-2 is legal (truncated grid)
                elif inflow_mode == 2:
                    phi[k, j, b_obc] = phi_prev[k, j, 1]

            else:
                if rx > r_max:                           # [RMAX] was 1/50
                    rx = r_max
                    flg += 2.0                           # [O2] rx clamp bound
                if ry > r_max:                           # [RMAX]
                    ry = r_max
                    flg += 4.0                           # [O2] ry clamp bound
                elif ry < -r_max:                        # [RMAX]
                    ry = -r_max
                    flg += 4.0                           # [O2]


                if ry >= 0.0:
                    dy_b = phi_prev[k, j, 1] - phi_prev[k, jm, 1] # phi^n_{b,j} - \phi_{b, j-1}
                else:  
                    dy_b = phi_prev[k, jp, 1] - phi_prev[k, j, 1] # phi^n_{b,j+1} - \phi_{b, j}                  

                # The 2D Orlanski Scheme
                phi[k, j, b_obc] = (phi_prev[k, j, 1] + rx * pim1 - ry * dy_b) / (1.0 + rx)
                phi[k, j, b_obc + 1] = phi[k,j, b_obc]
                if b_obc + 2 < ni:                                                   
                    phi[k, j, b_obc + 2] = phi[k,j, b_obc]

            sumr += rx
            if rx > maxr:
                maxr = rx
            rxy_out[k, j, 0] = rx   # [OBC - 2D - PhaseDiag] per-row applied rx
            rxy_out[k, j, 1] = ry   # [OBC - 2D - PhaseDiag] per-row applied ry
            rxy_out[k, j, 2] = flg  # [O2] diag bitmask (1=denom floor, 2=rx clamp, 4=ry clamp); pure diagnostics, no numerics effect
    rx_out[0] = sumr / (nk * nj)
    rx_out[1] = maxr


# ============ [NN] Lifted (numpy) east-Orlanski OBC for the neural-network path ============
def _phase_inflow_np(X8, eps_denom=1e-3):   # [EPSREL] scalar, or an array broadcastable over X8[...,0]
    """Analytic per-point (rx, ry) + inflow flag from the packed stencil (numpy). Faithful copy
    of the estimation branch of _orlanski_east; inflow == (raw rx<0).
    [RMAX] Deliberately UNCLAMPED -- this builds NN training baselines, and clamping here would
    shift every previously recorded baseline loss. The deployed kernel clamps at r_max.
    [EPSREL] eps_denom must match the value the KERNEL uses for the same field and layer --
    run_obc_nn calls this for its analytic fallback, so a mismatch silently desyncs the NN run
    from the analytic run. Default 1e-3 = the historical uniform floor."""
    pim1, pim2            = X8[..., 6], X8[..., 7]
    bm1_jm, bm1_j, bm1_jp = X8[..., 0], X8[..., 1], X8[..., 2]
    dphi_t = pim1 - bm1_j
    dphi_x = pim1 - pim2
    cen    = bm1_jp - bm1_jm
    dphi_y = np.where(dphi_t * cen > 0.0, bm1_j - bm1_jm, bm1_jp - bm1_j)   # upwind in j
    denom  = dphi_x * dphi_x + dphi_y * dphi_y

    # [OBC-eps] vectorized small-denominator guard: r_d = 1/eps_denom where denom<eps_denom
    # else 1/denom -- same as the scalar in-step _orlanski_east, but array-safe (np.maximum
    # avoids the Python `if denom<eps` ambiguity AND the 0-div warning since denom>=0).
    r_d = 1.0 / np.maximum(denom, eps_denom)      # [OBC-eps]  # [EPSREL] floor now a parameter
    rx = -dphi_t * dphi_x * r_d                   # [OBC-eps]
    ry = -dphi_t * dphi_y * r_d                   # [OBC-eps]
    inflow = rx < 0.0                             # inflow flag from RAW rx (before any clip)
    return rx, ry, inflow


def _obc_update_np(X8, cx, cy, inflow, phi_ext, alpha_in, dphi_in=None, dphi_out=None,   # [INCORR] +dphi_in  # [ANCH] +dphi_out
                   inflow_mode=0, phi_ext_out=None, inflow_nudge=True,                   # [IM] radiation-inflow scheme knob + col-(b+1) exterior @ n+1
                   alpha_from_rx=False):                                                 # [ALPHARX] True -> inflow nudge coeff = min(|cx|,1) per row (mode-0 branch only)
    """East-Orlanski boundary update (numpy), faithful to _orlanski_east.
    OUTFLOW: phi_b = (phi_prev_b + cx*pim1 - cy*dy_b)/(1+cx) [+ dphi_out].
    INFLOW : phi_b = phi_prev_b + alpha_in*(phi_ext - phi_prev_b) [+ dphi_in]  (cx,cy unused; alpha_in=1 == hard prescribe).
    alpha_from_rx : [ALPHARX] True -> the mode-0 inflow nudge coefficient is min(|cx|,1)
        per row instead of the fixed alpha_in, so the (NN) phase speed sets how hard the
        exterior is pulled in and the inflow branch carries gradient w.r.t. cx.
    dphi_in  : optional learned additive correction applied to the INFLOW nudge only (None -> no correction).
    dphi_out : optional learned additive correction applied to the OUTFLOW radiate only (None -> no correction) [ANCH].
    inflow_mode : [IM] 1 -> replace the inflow nudge with the RADIATION-INFLOW scheme     # [IM]
        phi_b = (phi_prev_b - cx*phi_ext_out - cy*dy_b)/(1 - cx)  (cx < 0 on inflow),     # [IM]
        with an optional extra alpha_in nudge toward phi_ext when inflow_nudge is True    # [IM]
        (mirrors nudging_mode==1 in _orlanski_east).  cx/cy arrive PRE-CLAMPED from the   # [IM]
        caller (run_obc_nn's r_max) when clamping is requested; this function itself      # [IM]
        applies no clamp.  0 -> unchanged.                                                # [IM]
    phi_ext_out : [IM] (...) external value at column b+1 at n+1; required when inflow_mode=1."""   # [INCORR]
    phi_prev_b_j = X8[..., 4]    # (b, j)   @ n
    pim1         = X8[..., 6]    # (b-1, j) @ n+1
    b_jm         = X8[..., 3]    # (b, j-1) @ n
    b_jp         = X8[..., 5]    # (b, j+1) @ n
    dy_b = np.where(cy >= 0.0, phi_prev_b_j - b_jm, b_jp - phi_prev_b_j)
    rad  = (phi_prev_b_j + cx * pim1 - cy * dy_b) / (1.0 + cx)
    if dphi_out is not None:                                              # [ANCH] additive outflow-radiate correction (anchored_corr)
        rad = rad + dphi_out                                              # [ANCH]
    if inflow_mode == 1:                                                  # [IM] radiation-inflow scheme on the inflow rows
        cx_in = np.where(inflow, cx, 0.0)                                 # [IM] keeps 1-cx away from 0 on outflow rows (those values are discarded by the final where)
        nudge = (phi_prev_b_j - cx_in * phi_ext_out - cy * dy_b) / (1.0 - cx_in)   # [IM] (phi^n - rx*phi_ext_out - ry*dy_b)/(1-rx), rx<0
        if inflow_nudge:                                                  # [IM] mirrors nudging_mode==1 in _orlanski_east
            nudge = nudge + alpha_in * (phi_ext - nudge)                  # [IM]
    else:                                                                 # [IM]
        if alpha_from_rx:                                                 # [ALPHARX] per-row alpha = min(|cx|,1)
            a_in = np.minimum(np.abs(cx), 1.0)                            # [ALPHARX]
        else:                                                             # [ALPHARX]
            a_in = alpha_in                                               # [ALPHARX] unchanged default
        nudge = phi_prev_b_j + a_in * (phi_ext - phi_prev_b_j)
    if dphi_in is not None:                                               # [INCORR] additive inflow-nudge correction (config 3.2)
        nudge = nudge + dphi_in                                           # [INCORR]
    return np.where(inflow, nudge, rad)


def _continuity_np(h0, u, v, dt, dx, dy, iter_num, y_periodic,
                   h_interface_relax=0.0, h_interface_target=None, D=None):
    """[FRESH] Numpy replica of _step_numba's directional-split continuity (plus the
    [INTERFACE_RELAX] source), used by run_obc_nn to obtain the n+1 interior at
    columns b-1, b-2 BEFORE the njit kernel runs, so the NN can predict this
    step's (rx, ry) with the SAME stencil timing as the in-kernel analytic
    Orlanski (and as its training data).

    Elementwise formulas and sweep order match the kernel exactly: i-direction
    wraps periodically; the j-direction is free-slip-walled or periodic per
    y_periodic; the second sweep reads the h already updated by the first; the
    i-first/j-first parity follows iter_num, as in the kernel.  The interface-
    relax zonal mean uses numpy's pairwise summation, which can differ from the
    kernel's sequential sum in the last ulp -- physically nil.

    Columns b-1 and b-2 after continuity depend ONLY on time-n data (the current
    step's OBC writes columns >= b, and continuity reads no updated columns), so
    this replica is exact for the boundary stencil."""
    h = h0.copy()
    cx = dt * (1.0 / dx)
    cy = dt * (1.0 / dy)
    do_i_first = (iter_num % 2 == 0)
    for sweep in range(2):
        if (sweep == 0) == do_i_first:
            # i-direction: hu = u_pos*h[i-1] + u_neg*h[i]; h -= cx*(hu[i+1]-hu[i])
            up = np.maximum(u, 0.0); un = np.minimum(u, 0.0)
            hu = up * np.roll(h, 1, axis=2) + un * h
            h = h - cx * (np.roll(hu, -1, axis=2) - hu)
        else:
            # j-direction: hv = v_pos*h[j-1] + v_neg*h[j]; h -= cy*(hv[j+1]-hv[j])
            vp = np.maximum(v, 0.0); vn = np.minimum(v, 0.0)
            if y_periodic:
                h_jm = np.roll(h, 1, axis=1)
            else:                                     # south wall: mirror row 0
                h_jm = np.concatenate([h[:, :1, :], h[:, :-1, :]], axis=1)
            hv = vp * h_jm + vn * h
            if y_periodic:
                hv_jp = np.roll(hv, -1, axis=1)
            else:                                     # north wall: zero flux (wn=0)
                hv_jp = np.concatenate([hv[:, 1:, :], np.zeros_like(hv[:, :1, :])], axis=1)
            h = h - cy * (hv_jp - hv)
    # [INTERFACE_RELAX] same source as the kernel (post-sweep h, zonal mean per row)
    if h_interface_relax > 0.0 and h.shape[0] == 2:
        c_ir = dt * h_interface_relax
        eta1_mean = (h[1] - D).mean(axis=-1)                  # (nj,)
        c_inter = c_ir * (eta1_mean - h_interface_target)     # (nj,)
        h[0] += c_inter[:, None]
        h[1] -= c_inter[:, None]
    return h


@njit(parallel=True, cache=True)
def _step_numba(u, v, h, D, taux, tauy, f, f_at_u, f_at_v,
                dt, dx, dy, g, epsilon, nu_h, nu_v, alpha_f, alpha_nu,
                h_zonal_target, h_target, u_target, v_target,
                h_zonal_relax, h_relax, u_relax, v_relax,
                h_relax_on, u_relax_on, v_relax_on, hsub, iter_num,
                h_interface_relax, h_interface_target,        # [INTERFACE_RELAX] eq.(2) term 1: rate [s-1] (0 -> off) + eta_3/2 reference profile (nj,)
                bc_mode, bc_cols, h_bc, u_bc, v_bc, h_diff, u_diff, v_diff,
                obc_on, b_obc, h_prev, u_prev, v_prev, rx_h, rx_u, rx_v,
                rxy_h, rxy_u, rxy_v,                         
                nudging_mode, h_ext, u_ext, v_ext, alpha_in,  
                h_zerograd,                                  
                inflow_persistent,
                ry_on,
                y_periodic,                                   # [YP] True -> j wraps; False -> free-slip walls
                r_max,                                        # [RMAX] outflow CFL clamp passed to _orlanski_east
                inflow_mode, h_ext_out, u_ext_out, v_ext_out,  # [IM] east-OBC inflow-scheme knob + per-field col-(b+1) exterior data @ n+1 (dummies when inflow_mode == 0)
                # [NNRX] pre-computed per-row (rx,ry) for h,u,v from an external source (e.g. NN).
                # When use_nn_rx==1, _orlanski_east reads from these arrays instead of computing
                # rx,ry internally.  When 0, the arrays are ignored (pass dummies).
                use_nn_rx,                                     # [NNRX] 0 -> compute rx,ry analytically (unchanged); 1 -> use pre-computed arrays below
                # [O2] obc_order threaded to _orlanski_east appears at the END of the signature (after alpha_from_rx)
                h_rx_pre, h_ry_pre,                            # [NNRX] (nk,nj) pre-computed rx,ry for h
                u_rx_pre, u_ry_pre,                            # [NNRX] (nk,nj) pre-computed rx,ry for u
                v_rx_pre, v_ry_pre,                            # [NNRX] (nk,nj) pre-computed rx,ry for v
                alpha_from_rx,                                 # [ALPHARX] 1 -> inflow nudge coeff = min(|rx|,1) (mode-0 nudging branch); 0 -> fixed alpha_in
                obc_order,                                     # [O2] estimator order for _orlanski_east (1 = unchanged, 2 = 2nd-order D_t/D_x)
                eps_h_arr, eps_u_arr, eps_v_arr,               # [EPSREL] per-field, per-layer denominator floors, each (nk,); np.full(nk,1e-3) each = historical behaviour
                ):
    """JIT-compiled, multi-threaded time step. Modifies u, v, h in place; state
    arrays have shape (nk, nj, ni). The work is organized as fused loops that
    recompute stencil quantities inline rather than building full-grid
    temporaries: restoring, pre-continuity hq, continuity, kinetic energy +
    Bernoulli + h-at-u/v reciprocals, the explicit momentum accelerations (PV
    Coriolis fluxes, Bernoulli gradient, viscous stress divergence), the
    cross-component averaging, and the solver tail (interfacial-stress
    coefficients computed inline as scalars; the cancellation-free TDMAH2
    implicit solve run per column with scalar locals).

    The j-row loops use prange and run across numba threads; rows are
    independent, so the loops are race-free. This is a memory-bandwidth-bound
    stencil, so set the thread count to the physical core count (not the SMT
    thread count) with numba.set_num_threads().

    IMPORTANT for parallel=True correctness: every reduction/broadcast is
    written as an explicit loop, NOT a numpy array expression. In particular the
    zonal-mean h restoring is an explicit per-row loop rather than
    h[0].sum(axis=-1)/reshape/broadcast, because numba's ParallelAccelerator
    mis-compiles axis reductions + broadcasts (it produced garbage / blow-ups,
    e.g. in the 1-layer smoke test, while every per-term test stayed
    bit-identical). Keep new terms in explicit-loop form.

    Layer thickness h is the prognostic; eta = h - D is diagnosed for the
    pressure gradient. The TDMAH2 recurrence row-scales each column by h_k to
    symmetrize the tridiagonal + Coriolis system and bundle Coriolis as
    h_k(1+ic); every denominator is a sum of nonnegative-real plus
    positive-real-part-complex terms, so it stays robust as h_k -> 0.
    Re(delta_w) gives delta_u; Im(delta_w) gives delta_v.
    
    obc_on int (0/1) for whether turn on or off of orlanski boundary condition. 


    """

    nk, nj, ni = u.shape
    rdx = 1 / dx
    rdy = 1 / dy

    # Pre-continuity hq at q-points (used for PV in the explicit loop); kept as
    # an array because PV is needed at several q-points. Fused u2q(h2u(h)).
    hq_pre = np.empty((nk, nj, ni))
    for k in range(nk):
        for j in prange(nj):
            jm = j - 1 if j > 0 else ( nj - 1 if y_periodic else 0 )       # [YP]
            for i in range(ni):
                im = i - 1 if i > 0 else ni - 1
                hq_pre[k,j,i] = 0.5 * ( 0.5 * ( h[k,j,i]  + h[k,j,im]  )
                                      + 0.5 * ( h[k,jm,i] + h[k,jm,im] ) )

    # Continuity: order-sensitive directional split. Each sub-step computes its
    # upwinded flux (hu/hv, retained for the PV fluxes below) then updates h in
    # place; the second sub-step sees the h updated by the first.
    hu = np.empty((nk, nj, ni))
    hv = np.empty((nk, nj, ni))
    cx = dt * rdx
    cy = dt * rdy
    do_i_first = ( iter_num % 2 == 0 )
    for sweep in range(2):
        if ( sweep == 0 ) == do_i_first:
            # i-direction: hu = u_pos*h[i-1] + u_neg*h[i]; h -= cx*(hu[i+1]-hu[i]).
            for k in range(nk):
                for j in prange(nj):
                    for i in range(ni):
                        im = i - 1 if i > 0 else ni - 1
                        ui = u[k,j,i]
                        up = ui if ui > 0.0 else 0.0
                        un = ui if ui < 0.0 else 0.0
                        hu[k,j,i] = up * h[k,j,im] + un * h[k,j,i]
            for k in range(nk):
                for j in prange(nj):
                    for i in range(ni):
                        ip = i + 1 if i < ni - 1 else 0
                        h[k,j,i] -= cx * ( hu[k,j,ip] - hu[k,j,i] )
        else:
            # j-direction: hv = v_pos*h[j-1] + v_neg*h[j]; h -= cy*(hv[j+1]-hv[j]).
            for k in range(nk):
                for j in prange(nj):
                    jm = j - 1 if j > 0 else ( nj - 1 if y_periodic else 0 )  # [YP] wall: hv[0] uses v[0]=0 -> zero flux through south wall
                    for i in range(ni):
                        vi = v[k,j,i]
                        vp = vi if vi > 0.0 else 0.0
                        vn = vi if vi < 0.0 else 0.0
                        hv[k,j,i] = vp * h[k,jm,i] + vn * h[k,j,i]
            for k in range(nk):
                for j in prange(nj):
                    jp = j + 1 if j < nj - 1 else ( 0 if y_periodic else nj - 1 )   # [YP]
                    wn = 1.0 if ( y_periodic or j < nj - 1 ) else 0.0                # [YP] north wall: zero meridional flux
                    for i in range(ni):
                        hv_jp = wn * hv[k,jp,i]                                      # [YP]
                        h[k,j,i] -= cy * ( hv_jp - hv[k,j,i] )
    
    # [INTERFACE_RELAX]
    if h_interface_relax > 0 and nk == 2: #ensure h_interface_relax is on and the model has two layers
        c_ir = dt * h_interface_relax
        for j in prange(nj):
            eta1_mean = 0.0
            for i in range(ni):
                eta1_mean += h[1, j, i] - D[j, i]
            eta1_mean /= ni
            c_inter = c_ir * (eta1_mean - h_interface_target[j])
            for i in range(ni):
                h[0, j, i] += c_inter #(3 - 2n) = +1 for n = 1 upper layer. 
                h[1, j, i] -= c_inter
                
    # bc_mode == 1 is storing the boundary, this is for control run
    n_bc = bc_cols.shape[0]                                   # number of columns
    if bc_mode == 1:                                          # record
        for k in range(nk):
            for j in range(nj):
                for c in range(n_bc):
                    h_bc[k,j,c] = h[k,j,bc_cols[c]]
    # bc_mode == 2 is for comparision (only used for testing), in this mode, the western sponge layer columns
    # are replaced by the corresponding stored values. 
    elif bc_mode == 2:                                        
        maxd = 0.0
        ss = 0.0
        for k in range(nk):
            for j in range(nj):
                for c in range(n_bc):
                    d = h[k,j,bc_cols[c]] - h_bc[k,j,c]
                    if abs(d) > maxd: maxd = abs(d)
                    ss += d*d
                    h[k,j,bc_cols[c]] = h_bc[k,j,c] #update h with the stored values. 
        h_diff[0] = maxd
        h_diff[1] = ( ss / ( nk*nj*n_bc ) )**0.5
    
    # [OBC] Orlanski boundary condition (only when obc_on is on)
    if obc_on == 1:
        if h_zerograd == 1:
            for k in range(nk):
                for j in range(nj):
                    h[k, j, b_obc] = h[k, j, b_obc - 1]
        else:
            _orlanski_east(h, h_prev, b_obc, rx_h,
                           nudging_mode, h_ext, alpha_in,
                           rxy_h,
                           inflow_persistent,
                           ry_on,
                           y_periodic,                                          # [YP]
                           r_max,                                               # [RMAX]
                           inflow_mode, h_ext_out,                              # [IM]
                           use_nn_rx, h_rx_pre, h_ry_pre,                      # [NNRX]
                           alpha_from_rx,                                       # [ALPHARX]
                           obc_order,                                           # [O2]
                           eps_h_arr)                                           # [EPSREL]

    # Interface positions eta (cumulative from bottom) and Montgomery potential
    # M (cumulative from top); cheap k-recursive arrays, kept as in reference.
    eta = np.empty_like(h)
    eta[nk-1] = h[nk-1] - D
    for k in range(nk-2, -1, -1):
        eta[k] = eta[k+1] + h[k]
    M = np.empty_like(h)
    M[0] = g[0] * eta[0]
    for k in range(1, nk):
        M[k] = M[k-1] + g[k] * eta[k]

    # Fused: Bernoulli B = M + KE, post-continuity h at u/v points and their
    # reciprocals (reused by the explicit loop, the interfacial coefficients,
    # and the TDMAH2 solve). KE uses the upwind-signed velocities inline.
    B = np.empty((nk, nj, ni))
    h_at_u = np.empty((nk, nj, ni))
    h_at_v = np.empty((nk, nj, ni))
    rhu = np.empty((nk, nj, ni))
    rhv = np.empty((nk, nj, ni))
    for k in range(nk):
        for j in prange(nj):
            jm = j - 1 if j > 0 else ( nj - 1 if y_periodic else 0 )            # [YP] wrap / mirror ghost row
            jp = j + 1 if j < nj - 1 else ( 0 if y_periodic else nj - 1 )       # [YP]
            wn = 1.0 if ( y_periodic or j < nj - 1 ) else 0.0                   # [YP] v = 0 on north wall
            for i in range(ni):
                im = i - 1 if i > 0 else ni - 1
                ip = i + 1 if i < ni - 1 else 0
                ui = u[k,j,i];   up    = ui  if ui  > 0.0 else 0.0
                uip = u[k,j,ip]; un_ip = uip if uip < 0.0 else 0.0
                vi = v[k,j,i];   vp    = vi  if vi  > 0.0 else 0.0
                vjp = wn * v[k,jp,i]; vn_jp = vjp if vjp < 0.0 else 0.0         # [YP]
                kin = 0.5 * ( up*up + un_ip*un_ip ) + 0.5 * ( vp*vp + vn_jp*vn_jp )
                B[k,j,i] = M[k,j,i] + kin
                hau = 0.5 * ( h[k,j,i] + h[k,j,im] )
                hav = 0.5 * ( h[k,j,i] + h[k,jm,i] )
                h_at_u[k,j,i] = hau; rhu[k,j,i] = 1.0 / ( hau + hsub )
                h_at_v[k,j,i] = hav; rhv[k,j,i] = 1.0 / ( hav + hsub )

    # --- FUSED explicit accelerations: udot, vdot in one pass ---
    udot = np.empty((nk, nj, ni))
    vdot = np.empty((nk, nj, ni))
    for k in range(nk):
        for j in prange(nj):
            jm = j - 1 if j > 0 else ( nj - 1 if y_periodic else 0 )            # [YP] wrap / mirror ghost for h/u/f
            jp = j + 1 if j < nj - 1 else ( 0 if y_periodic else nj - 1 )       # [YP] the v family uses the wn-masked locals below
            wn = 1.0 if ( y_periodic or j < nj - 1 ) else 0.0                   # [YP] north wall -> v = 0, hv = 0 across the north face
            for i in range(ni):
                im = i - 1 if i > 0 else ni - 1
                ip = i + 1 if i < ni - 1 else 0
                v_jp_i   = wn * v[k,jp,i]                                       # [YP]
                v_jp_im  = wn * v[k,jp,im]                                      # [YP]
                hv_jp_i  = wn * hv[k,jp,i]                                      # [YP]
                hv_jp_im = wn * hv[k,jp,im]                                     # [YP]
                # Masked PV at q-points (j,i), (jp,i), (j,ip):
                #   q = (f + vx - uy) ; q *= r ; q *= (hqp*r),  r = 1/(hqp+hsub)
                vort_c  = f[j, i]  + ( v[k,j,i]  - v[k,j,im]  ) * rdx - ( u[k,j,i]  - u[k,jm,i]  ) * rdy
                r_c     = 1.0 / ( hq_pre[k,j,i] + hsub )
                qpv_c   = ( vort_c * r_c ) * ( hq_pre[k,j,i] * r_c )
                # qpv_c   = ( vort_c * r_c )
                vort_jp = f[jp, i] + ( v_jp_i - v_jp_im ) * rdx - ( u[k,jp,i] - u[k,j,i]   ) * rdy   # [NP] v=0 on north wall
                r_jp    = 1.0 / ( hq_pre[k,jp,i] + hsub )
                qpv_jp  = ( vort_jp * r_jp ) * ( hq_pre[k,jp,i] * r_jp )
                # qpv_jp  = ( vort_jp * r_jp ) 
                vort_ip = f[j, ip] + ( v[k,j,ip] - v[k,j,i]   ) * rdx - ( u[k,j,ip] - u[k,jm,ip] ) * rdy
                r_ip    = 1.0 / ( hq_pre[k,j,ip] + hsub )
                qpv_ip  = ( vort_ip * r_ip ) * ( hq_pre[k,j,ip] * r_ip )
                # qpv_ip  = ( vort_ip * r_ip )

                # Bernoulli gradient.
                Bx = ( B[k,j,i] - B[k,j,im] ) * rdx
                By = ( B[k,j,i] - B[k,jm,i] ) * rdy

                # PV Coriolis fluxes (q2u/v2q and q2v/u2q expanded).
                qhv = 0.5 * ( qpv_c  * 0.5 * ( hv[k,j,i]  + hv[k,j,im]  )
                            + qpv_jp * 0.5 * ( hv_jp_i + hv_jp_im ) )   # [NP] zero flux on north wall
                qhu = 0.5 * ( qpv_c  * 0.5 * ( hu[k,j,i]  + hu[k,jm,i]  )
                            + qpv_ip * 0.5 * ( hu[k,j,ip] + hu[k,jm,ip] ) )

                # Stress tensor: nu_h*h*D_tension at h-points {(j,i),(j,im),(jm,i)};
                # nu_h*hq*D_shear at q-points {(j,i),(jp,i),(j,ip)}.
                Dt_c  = ( u[k,j,ip] - u[k,j,i]  ) * rdx - ( v_jp_i - v[k,j,i]  ) * rdy   # [NP] v=0 on north wall
                nuhDt_c  = nu_h * h[k,j,i]  * Dt_c
                Dt_im = ( u[k,j,i]  - u[k,j,im] ) * rdx - ( v_jp_im - v[k,j,im] ) * rdy   # [NP]
                nuhDt_im = nu_h * h[k,j,im] * Dt_im
                Dt_jm = ( u[k,jm,ip]- u[k,jm,i] ) * rdx - ( v[k,j,i]  - v[k,jm,i] ) * rdy
                nuhDt_jm = nu_h * h[k,jm,i] * Dt_jm

                Ds_c  = ( u[k,j,i]  - u[k,jm,i] ) * rdy + ( v[k,j,i]  - v[k,j,im]  ) * rdx
                hqp_c  = min( min( h[k,j,i],  h[k,j,im]  ), min( h[k,jm,i], h[k,jm,im] ) )
                nuhqDs_c  = nu_h * hqp_c  * Ds_c
                Ds_jp = ( u[k,jp,i] - u[k,j,i]  ) * rdy + ( v_jp_i - v_jp_im ) * rdx   # [NP] v=0 on north wall
                hqp_jp = min( min( h[k,jp,i], h[k,jp,im] ), min( h[k,j,i],  h[k,j,im]  ) )
                nuhqDs_jp = nu_h * hqp_jp * Ds_jp
                Ds_ip = ( u[k,j,ip] - u[k,jm,ip]) * rdy + ( v[k,j,ip] - v[k,j,i]   ) * rdx
                hqp_ip = min( min( h[k,j,ip], h[k,j,i]   ), min( h[k,jm,ip],h[k,jm,i]  ) )
                nuhqDs_ip = nu_h * hqp_ip * Ds_ip

                uxxyy = ( ( nuhDt_c - nuhDt_im ) * rdx + ( nuhqDs_jp - nuhqDs_c ) * rdy ) * rhu[k,j,i]
                vxxyy = ( ( nuhqDs_ip - nuhqDs_c ) * rdx - ( nuhDt_c - nuhDt_jm ) * rdy ) * rhv[k,j,i]

                ud = ( qhv - Bx ) + uxxyy
                vd = - ( qhu + By ) + vxxyy
                if k == 0:
                    ud += taux[j,i] * rhu[0,j,i]
                    vd += tauy[j,i] * rhv[0,j,i]
                udot[k,j,i] = ud
                vdot[k,j,i] = vd                   


    # --- explicit -(L u^n), -(L v^n), per column ---
    # Interfacial-stress coefficients are recomputed inline as scalars (no
    # a_top/a_bot arrays): a_{k-1/2} = 2*nu_v/(h_{k-1}+h_k) interior, top = 0,
    # bottom (k=nk-1) = epsilon. (L u)_k = ((a_top+a_bot) u_k - a_top u_{k-1}
    # - a_bot u_{k+1}) / h_k.
    adt = alpha_nu * dt
    for j in prange(nj):
        for i in range(ni):
            for k in range(nk):
                atu = 0.0 if k == 0 else 2.0 * nu_v / ( h_at_u[k-1,j,i] + h_at_u[k,j,i] + hsub )
                abu = epsilon if k == nk-1 else 2.0 * nu_v / ( h_at_u[k,j,i] + h_at_u[k+1,j,i] + hsub )
                atv = 0.0 if k == 0 else 2.0 * nu_v / ( h_at_v[k-1,j,i] + h_at_v[k,j,i] + hsub )
                abv = epsilon if k == nk-1 else 2.0 * nu_v / ( h_at_v[k,j,i] + h_at_v[k+1,j,i] + hsub )
                Lu = ( atu + abu ) * u[k,j,i]
                Lv = ( atv + abv ) * v[k,j,i]
                if k > 0:
                    Lu -= atu * u[k-1,j,i]
                    Lv -= atv * v[k-1,j,i]
                if k < nk - 1:
                    Lu -= abu * u[k+1,j,i]
                    Lv -= abv * v[k+1,j,i]
                udot[k,j,i] -= Lu * rhu[k,j,i]
                vdot[k,j,i] -= Lv * rhv[k,j,i]
    if obc_on == 1: 
        udot[:, :, b_obc] = udot[:, :, b_obc - 1]

    # --- implicit TDMAH2 (cancellation-free; recurrence identical to reference,
    # done per column with scalar locals) ---
    # Fused interpolation of the cross-component accelerations: vdot to
    # u-points = q2u(v2q(vdot)), udot to v-points = q2v(u2q(udot)).
    vdot_at_u = np.empty((nk, nj, ni))
    udot_at_v = np.empty((nk, nj, ni))
    for k in range(nk):
        for j in prange(nj):
            jm = j - 1 if j > 0 else ( nj - 1 if y_periodic else 0 )            # [YP] wrap / mirror ghost row
            jp = j + 1 if j < nj - 1 else ( 0 if y_periodic else nj - 1 )       # [YP]
            wn = 1.0 if ( y_periodic or j < nj - 1 ) else 0.0                   # [YP] v-tendency = 0 on north wall
            for i in range(ni):
                im = i - 1 if i > 0 else ni - 1
                ip = i + 1 if i < ni - 1 else 0
                vdot_jp_i  = wn * vdot[k,jp,i]                                  # [YP]
                vdot_jp_im = wn * vdot[k,jp,im]                                 # [YP]
                vdot_at_u[k,j,i] = 0.5 * ( 0.5 * ( vdot[k,j,i]  + vdot[k,j,im]  )
                                         + 0.5 * ( vdot_jp_i + vdot_jp_im ) )
                udot_at_v[k,j,i] = 0.5 * ( 0.5 * ( udot[k,j,i]  + udot[k,jm,i]  )
                                         + 0.5 * ( udot[k,j,ip] + udot[k,jm,ip] ) )

    # q and y_prime hold the per-column forward-sweep ratios; kept as arrays so
    # the backward sweep can read them (and so the loops stay prange-safe).
    q       = np.empty((nk, nj, ni), dtype=np.complex128)
    y_prime = np.empty((nk, nj, ni), dtype=np.complex128)

    # u-point pass: take Re(delta_w) for delta_u.
    for j in prange(nj):
        for i in range(ni):
            ic = alpha_f * dt * f_at_u[j,i]
            hc   = h_at_u[0,j,i] * ( 1.0 + 1j * ic )
            a_b  = adt * ( epsilon if nk == 1 else 2.0 * nu_v / ( h_at_u[0,j,i] + h_at_u[1,j,i] + hsub ) )
            beta = 1.0 / ( hc + a_b )            # a_top[0] = 0
            q[0,j,i] = a_b * beta
            Q    = hc * beta
            y_prime[0,j,i] = h_at_u[0,j,i] * ( dt * udot[0,j,i] + 1j * ( dt * vdot_at_u[0,j,i] ) ) * beta
            for k in range(1, nk):
                a_t  = adt * ( 2.0 * nu_v / ( h_at_u[k-1,j,i] + h_at_u[k,j,i] + hsub ) )
                a_b  = adt * ( epsilon if k == nk-1 else 2.0 * nu_v / ( h_at_u[k,j,i] + h_at_u[k+1,j,i] + hsub ) )
                hc   = h_at_u[k,j,i] * ( 1.0 + 1j * ic )
                beta = 1.0 / ( hc + a_t * Q + a_b )
                q[k,j,i] = a_b * beta
                Q    = ( hc + a_t * Q ) * beta
                y_k  = h_at_u[k,j,i] * ( dt * udot[k,j,i] + 1j * ( dt * vdot_at_u[k,j,i] ) )
                y_prime[k,j,i] = ( y_k + a_t * y_prime[k-1,j,i] ) * beta
            delta_w = y_prime[nk-1,j,i]
            u[nk-1,j,i] += delta_w.real
            for k in range(nk-2, -1, -1):
                delta_w = y_prime[k,j,i] + q[k,j,i] * delta_w
                u[k,j,i] += delta_w.real

    # v-point pass: take Im(delta_w) for delta_v.
    for j in prange(nj):
        for i in range(ni):
            ic = alpha_f * dt * f_at_v[j,i]
            hc   = h_at_v[0,j,i] * ( 1.0 + 1j * ic )
            a_b  = adt * ( epsilon if nk == 1 else 2.0 * nu_v / ( h_at_v[0,j,i] + h_at_v[1,j,i] + hsub ) )
            beta = 1.0 / ( hc + a_b )
            q[0,j,i] = a_b * beta
            Q    = hc * beta
            y_prime[0,j,i] = h_at_v[0,j,i] * ( dt * udot_at_v[0,j,i] + 1j * ( dt * vdot[0,j,i] ) ) * beta
            for k in range(1, nk):
                a_t  = adt * ( 2.0 * nu_v / ( h_at_v[k-1,j,i] + h_at_v[k,j,i] + hsub ) )
                a_b  = adt * ( epsilon if k == nk-1 else 2.0 * nu_v / ( h_at_v[k,j,i] + h_at_v[k+1,j,i] + hsub ) )
                hc   = h_at_v[k,j,i] * ( 1.0 + 1j * ic )
                beta = 1.0 / ( hc + a_t * Q + a_b )
                q[k,j,i] = a_b * beta
                Q    = ( hc + a_t * Q ) * beta
                y_k  = h_at_v[k,j,i] * ( dt * udot_at_v[k,j,i] + 1j * ( dt * vdot[k,j,i] ) )
                y_prime[k,j,i] = ( y_k + a_t * y_prime[k-1,j,i] ) * beta
            delta_w = y_prime[nk-1,j,i]
            v[nk-1,j,i] += delta_w.imag
            for k in range(nk-2, -1, -1):
                delta_w = y_prime[k,j,i] + q[k,j,i] * delta_w
                v[k,j,i] += delta_w.imag

    # [OBC] block B: record/replace (u,v) at the prescribed boundary columns (bc_cols,
    # [OBC] any length), done AFTER the implicit (u,v) solve but BEFORE restoring, so the
    # [OBC] stored values are the same pre-restoring phase point as block A's h. Serial loops.
    n_bc = bc_cols.shape[0]                                   # [OBC] number of columns
    if bc_mode == 1:                                          # [OBC] record
        for k in range(nk):
            for j in range(nj):
                for c in range(n_bc):
                    u_bc[k,j,c] = u[k,j,bc_cols[c]]
                    v_bc[k,j,c] = v[k,j,bc_cols[c]]
    elif bc_mode == 2:                                        # [OBC] compare + replace
        umaxd = 0.0; uss = 0.0
        vmaxd = 0.0; vss = 0.0
        for k in range(nk):
            for j in range(nj):
                for c in range(n_bc):
                    du = u[k,j,bc_cols[c]] - u_bc[k,j,c]
                    dv = v[k,j,bc_cols[c]] - v_bc[k,j,c]
                    if abs(du) > umaxd: umaxd = abs(du)
                    if abs(dv) > vmaxd: vmaxd = abs(dv)
                    uss += du*du
                    vss += dv*dv
                    u[k,j,bc_cols[c]] = u_bc[k,j,c]
                    v[k,j,bc_cols[c]] = v_bc[k,j,c]
        u_diff[0] = umaxd; u_diff[1] = ( uss / ( nk*nj*n_bc ) )**0.5
        v_diff[0] = vmaxd; v_diff[1] = ( vss / ( nk*nj*n_bc ) )**0.5
    
    # [OBC - Orlanski]
    if obc_on == 1:
        # [OBC - Orlanski - 2D - IndepRx] u and v each estimate their own phase speed.
        _orlanski_east(u, u_prev, b_obc, rx_u,
                       nudging_mode, u_ext, alpha_in,
                       rxy_u,  # [OBC - Orlanski - 2D - Nuding] +nudging args; [OBC - 2D - PhaseDiag] +per-row rxy_u
                       inflow_persistent,  # [PERSIST]
                       ry_on,  # [RY1D]
                       y_periodic,  # [YP]
                       r_max,  # [RMAX]
                       inflow_mode, u_ext_out,  # [IM]
                       use_nn_rx, u_rx_pre, u_ry_pre,  # [NNRX]
                       alpha_from_rx,  # [ALPHARX]
                       obc_order,  # [O2]
                       eps_u_arr)  # [EPSREL]
        _orlanski_east(v, v_prev, b_obc, rx_v,
                       nudging_mode, v_ext, alpha_in,
                       rxy_v,  # [OBC - Orlanski - 2D - Nuding] +nudging args; [OBC - 2D - PhaseDiag] +per-row rxy_v
                       inflow_persistent,  # [PERSIST]
                       ry_on,  # [RY1D]
                       y_periodic,  # [YP]
                       r_max,  # [RMAX]
                       inflow_mode, v_ext_out,  # [IM]
                       use_nn_rx, v_rx_pre, v_ry_pre,  # [NNRX]
                       alpha_from_rx,  # [ALPHARX]
                       obc_order,  # [O2]
                       eps_v_arr)  # [EPSREL]

    # --- restoring (backward Euler) ---
    # Applied after the dynamics, so each field relaxes implicitly toward its
    # target: q^{n+1} = (q + c*q_targ)/(1+c) with c = dt*rate. This is unconditionally
    # stable for any rate (amplification 1/(1+c) in (0,1]) and clamps to the target
    # as c->inf. Algebraically q^{n+1} = q - c/(1+c)*(q - q_targ), so it keeps the
    # explicit "-=" form with the rate coefficient c replaced by c/(1+c).
    # Zonal-mean h restoring on layer 0 toward h_zonal_target[0,:] (scalar rate).
    # Explicit per-row loop (NOT h[0].sum(axis=-1)/reshape/broadcast): numba
    # parallel=True mis-compiles that axis reduction + broadcast.
    if h_zonal_relax > 0:
        cz = dt * h_zonal_relax
        czb = cz / ( 1.0 + cz )
        for j in prange(nj):
            s = 0.0
            for i in range(ni):
                s += h[0, j, i]
            cd = czb * ( s / ni - h_zonal_target[0, j] )
            for i in range(ni):
                h[0, j, i] -= cd
    # Pointwise (localizable) sponge restoring of h toward full-field
    # target; rate is (nk,nj,ni) array, gated by precomputed booleans.
    if h_relax_on:
        h -= ( dt * h_relax / ( 1.0 + dt * h_relax ) ) * ( h - h_target )

    # Pointwise (localizable) sponge restoring of u, v toward their full-field
    # targets; rates are (nk,nj,ni) arrays, gated by precomputed booleans.
    if u_relax_on:
        u -= ( dt * u_relax / ( 1.0 + dt * u_relax ) ) * ( u - u_target )
    if v_relax_on:
        v -= ( dt * v_relax / ( 1.0 + dt * v_relax ) ) * ( v - v_target )

    # [NP] south wall sits on the v-row j=0 (y=0): no normal flow through it.
    # Enforced last so it also overrides the east OBC / sponge at the SW & SE corners.
    if not y_periodic:                                                          # [YP]
        for k in range(nk):
            for i in range(ni):
                v[k, 0, i] = 0.0

class SSWEM:
    """(S)tacked (S)hallow (W)ater (E)quation (M)odel"""

    def __init__(self, ni, g, Ho, Lx, fo, beta, epsilon, nu_h, nu_v=0,
                 h_zonal_relax=0, h_relax=None, u_relax=None, v_relax=None,
                 h_interface_relax=0, h_interface_target=None,  # [INTERFACE_RELAX]
                 h_target=None, u_target=None, v_target=None,
                 hsub=1e-12, nj=None, Ly=None, y_periodic=False):
        """
        ni      - Number of cells in i-direction
        g       - Gravity [m s-2]; scalar (broadcast to length 1) or length-nk
                  vector. nk = len(g) sets the number of layers.
        Ho      - Nominal layer thickness [m]; scalar or length-nk vector.
                  Sum gives the nominal total water column depth.
        Lx      - Domain width [m]
        fo      - Coriolis [s-1]
        beta    - df/dy [m-1 s-1]
        epsilon - Bottom drag rate [m s-1]; bottom-boundary entry of L.
        nu_h    - Lateral (horizontal) viscosity [m2 s-1]
        nu_v    - Vertical viscosity [m2 s-1]; sets interior interfacial-stress
                  coefficients a_{k-1/2} = 2*nu_v/(h_{k-1}+h_k) for 1<k<=K. Defaults to 0.
        h_zonal_relax - Restoring rate for zonal-mean layer-0 thickness toward
                  h_zonal_target [s-1] (scalar). Acts on the zonal mean only, so
                  it nudges the mean profile without damping eddies. Set the
                  target with set_h_forcing().
        h_interface_relax - [INTERFACE_RELAX] Interface relaxation rate gamma
                  [s-1] (scalar; 0 = off, the default). Two-layer only: adds the
                  source (3-2n)*gamma*( mean_x(eta_3/2) - h_interface_target )
                  to the layer-n continuity equation, i.e. +gamma*(...) to layer
                  0 and -gamma*(...) to layer 1. Antisymmetric, so it moves
                  thickness ACROSS the interface: sum_k h and the free surface
                  are untouched, unlike h_zonal_relax. Acting on the zonal mean
                  only, it holds the mean stratification without damping eddies.
                  Applied as a continuity source (with the flux divergence),
                  NOT with the end-of-step sponges.
        h_interface_target - [INTERFACE_RELAX] Reference interface eta_3/2,Ref
                  [m], measured like eta(): negative below z=0. Scalar or
                  anything broadcastable to (nj,); default is the resting
                  interface, -Ho[0]. Typically set from the balanced initial
                  state, e.g. M.h_interface_target = M.eta(h=h0, k=1).mean(-1).
        h_relax, u_relax, v_relax - Pointwise (localizable) sponge restoring
                  rates [s-1] for h, u, v toward h_target, u_target, v_target.
                  None (default) for no restoring; otherwise a scalar or any
                  array broadcastable to (nk, nj, ni) for a spatially varying
                  rate (e.g. a sponge confined to part of the domain).
        h_target, u_target, v_target - Full-field targets for the pointwise
                  sponge. Scalar or any array broadcastable to (nk, nj, ni).
                  Default 0 (h_target defaults to the rest thickness Ho).
        hsub    - H sub-roundoff [m]
        nj      - Number of cells in j-direction. Default None.
        Ly      - Domain height [m]. Default None -> square domain (Ly=Lx,
                  nj=ni). If given, cells stay square (dy=dx=Lx/ni) and nj is
                  derived from Ly (snapped to an integer number of cells),
                  giving a rectangular domain (e.g. longer in x than y).
        """
        self.ni = ni
        self.g = np.atleast_1d(np.asarray(g, dtype=float)).copy()
        self.Ho = np.atleast_1d(np.asarray(Ho, dtype=float)).copy()
        self.nk = self.g.size
        if self.Ho.size != self.nk:
            raise ValueError(f"Ho must have length nk={self.nk}, got {self.Ho.size}")
        self.h_zonal_relax = float(h_zonal_relax)
        # [INTERFACE_RELAX] eq. (2) term 1. Two-layer only: the kernel gates on
        # nk == 2, so a nonzero rate on a 1- or 3-layer model is silently inert.
        # Warn rather than raise, so nk-generic driver notebooks stay runnable.
        self.h_interface_relax = float(h_interface_relax)
        if self.h_interface_relax > 0 and self.nk != 2:
            print(f"[INTERFACE_RELAX] WARNING: h_interface_relax={self.h_interface_relax:g} "
                  f"is ignored at nk={self.nk} (the term is defined for nk=2 only)")
        self.Lx = Lx
        self.fo = fo
        self.beta = beta
        self.epsilon = epsilon
        self.nu_h = nu_h
        self.nu_v = float(nu_v)
        self.hsub = hsub
        # [YP] Meridional boundary condition. False (default here) = free-slip solid walls
        # at y=0 and y=Ly, the zonal channel this module was written for. True = the
        # j-direction wraps (doubly-periodic), e.g. a Phillips baroclinic-instability run.
        # The i-direction is periodic either way, so sum(h) is conserved in both cases.
        self.y_periodic = bool(y_periodic)
        self.alpha_f = 0.5 # Crank-Nicholson for Coriolis
        self.alpha_nu = 1.0 # Euler backward for interfacial stresses (vertical viscosity + bottom drag)

        # Grid resolution
        self.dx = Lx / ni # Cell width [m]

        # Domain in y. Default (Ly=None) is a square domain/grid. If Ly is given,
        # the cells are kept square (dy = dx) and nj is derived from Ly (Ly is
        # snapped to an integer number of cells), allowing a rectangular domain.
        if nj is None :
            if Ly is None :
                self.nj = self.ni
                self.Ly = self.Lx
                self.dy = self.dx
            else:
                self.dy = self.dx
                self.nj = int(round(Ly / self.dx))
                self.Ly = self.nj * self.dy
        else:
            self.nj = nj
            if Ly is None : # Assume equal cell aspect ratio
                self.dy = self.dx
                self.Ly = self.dy * self.nj
            else:
                self.Ly = Ly
                self.dy = self.Ly / self.nj # Cell height [m]

        # Grid
        # xh1, yh1 are 1D coordinate for h points
        self.xh1, self.yh1 = self.dx * ( np.arange(self.ni) + 0.5 ), self.dy * ( np.arange(self.nj) + 0.5 )
        # xh, yh are @D coordinate for h points
        self.xh, self.yh = np.meshgrid(self.xh1, self.yh1)
        # xq1, yq1 are 1D coordinate for q points
        self.xq1, self.yq1 = self.dx * np.arange(self.ni + 0) , self.dy * np.arange(self.nj + 0)
        # xq, yq are @D coordinate for q points
        self.xq, self.yq = np.meshgrid(self.xq1, self.yq1)
        # xu, yu are @D coordinate for u points
        self.xu, self.yu = np.meshgrid(self.xq1, self.yh1)
        # xv, yv are @D coordinate for u points
        self.xv, self.yv = np.meshgrid(self.xh1, self.yq1)
        print("mesh: ni =",self.ni," nj =",self.nj)
        print("Grid: dx =",self.dx,"[m]"," dy =",self.dy,"[m]")

        # Resting initial conditions, no (flat) bathymetry and no forcing.
        # flat_topog must precede resting_state since rest h = D.
        self.flat_topog()
        self.resting_state()
        self.zero_forcing()
        # Zonal-mean restoring target: rest layer thickness per row (nk, nj).
        # Used by the scalar h_zonal_relax mechanism (set via set_h_forcing).
        self.h_zonal_target = np.tile(self.Ho[:, None], (1, self.nj)).astype(float)
        # [INTERFACE_RELAX] eta_3/2 reference profile (nj,). Default = the resting
        # interface: at rest h[1] = Ho[1] and D = sum(Ho), so eta[1] = h[1] - D = -Ho[0].
        # Routed through the property below, which coerces to a contiguous float64 (nj,).
        self.h_interface_target = ( -self.Ho[0] if self.nk >= 2 else 0.0 ) \
            if h_interface_target is None else h_interface_target

        # Pointwise-sponge targets and rates (full field, all layers). Accept
        # anything broadcastable to (nk, nj, ni): scalar, (ni,), (nj,ni),
        # (nk,nj,ni), ... Rates default to None (no restoring) -> a zero array,
        # so the type passed to the JIT step is always a float64 (nk,nj,ni)
        # array; the actual on/off decision is made by a boolean gate in step().
        # h_target defaults to the rest thickness Ho (sensible if a sponge is
        # enabled without setting a target); u/v targets default to 0.
        shape = (self.nk, self.nj, self.ni)
        h_t = self.Ho[:, None, None] if h_target is None else h_target
        u_t = 0.0 if u_target is None else u_target
        v_t = 0.0 if v_target is None else v_target
        self.h_target = np.broadcast_to(np.asarray(h_t, dtype=float), shape).copy()
        self.u_target = np.broadcast_to(np.asarray(u_t, dtype=float), shape).copy()
        self.v_target = np.broadcast_to(np.asarray(v_t, dtype=float), shape).copy()
        # Rates are routed through their property setters (below), which coerce
        # to a (nk,nj,ni) float64 array and cache the on/off gate once, so the
        # per-step .max() recomputation is avoided.
        self.h_relax = h_relax
        self.u_relax = u_relax
        self.v_relax = v_relax

        self._bc_cols_dummy = np.zeros(1, dtype=np.int64)
        self._bc_dummy = np.zeros((self.nk, self.nj, 1))
        self._diff_dummy = np.zeros(2)
        # [OBC - Orlanski]
        self._obc_prev_dummy = np.zeros((self.nk, self.nj, 3)) #place holder.  # [O2] slot 2 = phi^{n-1}_{b-1} (read only at obc_order==2)
        # [OBC - Orlanski - 2D - Nuding] (nk,nj) scratch for the external col-b data passed when nudging is off
        self._obc_ext_dummy = np.zeros((self.nk, self.nj))
        # [OBC - 2D - PhaseDiag] (nk,nj,2) scratch for per-row (rx,ry) when no recording array is supplied
        self._obc_rxy_dummy = np.zeros((self.nk, self.nj, 3))   # [O2] slot 2 = diag flag bitmask
        # [EPSREL] default denominator floor: uniform 1e-3 for every (field, layer) = the
        # [EPSREL] historical hard-coded value, so every existing caller is unchanged.
        self._obc_eps_default = np.ascontiguousarray(np.full((3, self.nk), 1e-3))   # [EPSREL]
        # [NNRX] (nk,nj) dummy arrays passed to _orlanski_east when use_nn_rx==0; they are
        # [NNRX] never read in that branch but numba needs consistent types across the call.
        self._nn_rx_dummy = np.zeros((self.nk, self.nj))
        self._nn_ry_dummy = np.zeros((self.nk, self.nj))
        # [CL] per-step centre-line store; stays None until a run is called with store_center=True
        self.center_line = None


        # Derived parameters
        self.f = self.fo + self.beta * self.yq # Coriolis is at q-points
        # [YP] q2u is a j-average that wraps at j=nj-1, which is right when the j-direction
        # is periodic and wrong against a wall; the analytic form is identical in the
        # interior and correct on the north row, so use it only when the walls are on.
        self.f_at_u = ( _nb_q2u( self.f ) if self.y_periodic
                        else self.fo + self.beta * self.yu )  # Coriolis interpolated to u-points
        self.f_at_v = _nb_q2v( self.f ) # Coriolis interpolated to v-points (q2v is an i-average, no j-periodicity)
        self.cg = np.sqrt( self.g[0] * self.Ho.sum() )
        if not self.fo==0:
            self.Ld = self.cg / self.fo
        else: self.Ld = None
        print("cg =", self.cg, "[m s-1]")
        print("Ld =", self.Ld, "[m]")
        if self.nk==2:
            self.cg1 = np.sqrt( self.g[1] * self.Ho.prod() / self.Ho.sum() )
            if not self.fo==0:
                self.Ld1 = self.cg1 / self.fo
            else: self.Ld1 = None
            print("cg1 =", self.cg1, "[m s-1]")
            print("Ld1 =", self.Ld1, "[m]")
        else:
            self.cg1 = None
            self.Ld1 = None
        if not self.beta==0:
            self.Ls = self.epsilon / ( self.beta * self.Ho.sum() )
        else: self.Ls = None
        print("Scales: Ls=epsilon/D/beta =", self.Ls, "[m]")
        if self.Ld is not None and self.Ld>0:
            print("Scales: Lx/Ld =", self.Lx / self.Ld)
        if self.Ld1 is not None and self.Ld1>0:
            print("Scales: Lx/Ld1 =", self.Lx / self.Ld1)
        if self.Ld is not None:
            print("Res: Ld/dx =",self.Ld / self.dx)
        if self.Ld1 is not None:
            print("Res: Ld1/dx =",self.Ld1 / self.dx)
        if not self.beta==0:
            print("Res: Ls/dx =",self.Ls / self.dx)

    def _check_eps_denom(self, eps_denom):                                   # [EPSREL]
        """Validate/normalize the per-(field,layer) denominator floor to a (3,nk) C-contiguous
        float64 array, rows ordered h, u, v.  None -> the historical uniform 1e-3.  A scalar is
        broadcast to every field and layer (also historical behaviour, explicitly)."""
        if eps_denom is None:
            return self._obc_eps_default
        # [EPSREL] asarray FIRST -- np.ascontiguousarray promotes a 0-d scalar to shape (1,),
        # [EPSREL] which would make the scalar branch below unreachable.
        E = np.asarray(eps_denom, dtype=np.float64)
        if E.ndim == 0:                             # scalar -> every field and layer
            E = np.full((3, self.nk), float(E))
        elif E.shape == (3,):                       # per field, layers pooled
            E = np.repeat(E[:, None], self.nk, axis=1)
        E = np.ascontiguousarray(E)
        if E.shape != (3, self.nk):
            raise ValueError(f'[EPSREL] eps_denom must be a scalar, (3,), or (3,nk)=(3,{self.nk}); '
                             f'got shape {E.shape}')
        if not np.all(np.isfinite(E)) or np.any(E <= 0.0):
            raise ValueError('[EPSREL] eps_denom entries must be finite and > 0 '
                             '(the floor is compared against denom = dphi_x^2 + dphi_y^2 >= 0)')
        return E

    def _set_relax(self, value):
        """Coerce a relax rate (None / scalar / broadcastable array) to a
        contiguous (nk,nj,ni) float64 array and return it together with its
        on/off gate (True iff any rate is positive). The array is marked
        read-only so the cached gate cannot silently desync: change a rate by
        assigning a new value (e.g. M.u_relax = arr), which refreshes the gate,
        not by mutating M.u_relax in place."""
        r = 0.0 if value is None else value
        arr = np.broadcast_to(np.asarray(r, dtype=float),
                              (self.nk, self.nj, self.ni)).copy()
        arr.flags.writeable = False
        return arr, bool(arr.max() > 0)

    @property
    def h_relax(self):
        return self._h_relax

    @h_relax.setter
    def h_relax(self, value):
        self._h_relax, self._h_relax_on = self._set_relax(value)

    @property
    def u_relax(self):
        return self._u_relax

    @u_relax.setter
    def u_relax(self, value):
        self._u_relax, self._u_relax_on = self._set_relax(value)

    @property
    def v_relax(self):
        return self._v_relax

    @v_relax.setter
    def v_relax(self, value):
        self._v_relax, self._v_relax_on = self._set_relax(value)

    @property
    def h_interface_target(self):
        """[INTERFACE_RELAX] Reference interface eta_3/2,Ref [m], one value per
        row, shape (nj,)."""
        return self._h_interface_target

    @h_interface_target.setter
    def h_interface_target(self, value):
        """Accept a scalar or anything broadcastable to (nj,) -- e.g. the zonal
        mean of the balanced initial interface, M.eta(h=h0, k=1).mean(axis=-1).
        Coerced to a contiguous float64 (nj,) copy so the type handed to the JIT
        step never changes and the caller's array cannot alias the model's."""
        arr = np.broadcast_to(np.asarray(value, dtype=float), (self.nj,))
        self._h_interface_target = np.ascontiguousarray(arr, dtype=float)

    def resting_state(self):
        """Set state to resting (u=v=0). Distribute layer thicknesses so that
        interfaces lie at their rest depths where the column is deep enough,
        and lower layers are clipped to the bathymetry where they would
        otherwise extend past it (h[k]=0 when blocked). Σ_k h[k] = min(D, ΣHo)
        everywhere; for nk=1 this collapses to h[0] = min(Ho[0], D) = D."""
        self.u = np.zeros((self.nk, self.nj, self.ni))
        self.v = np.zeros((self.nk, self.nj, self.ni))
        h = np.zeros((self.nk, self.nj, self.ni))
        z_top = np.zeros_like(self.D)  # depth of upper interface, positive down
        for k in range(self.nk):
            z_bot = z_top + self.Ho[k]
            h[k] = np.minimum(z_bot, self.D) - np.minimum(z_top, self.D)
            z_top = z_bot
        self.h = h
        self.time = 0
        self.iter = 0

    def eta(self, h=None, k=None):
        """Interface positions eta_{k-1/2} (relative to mean sea level z=0).
        eta[k] = -D + sum_{l=k}^{nk-1} h[l] for k = 0..nk-1 (free-surface to
        top of bottom layer); eta[nk] = -D is the bathymetry.

        h    Layer thicknesses, shape (nk, nj, ni). Defaults to self.h.
        k    Optional integer in [0, nk]. If given, returns the 2D field for
             that interface; otherwise returns the full (nk+1, nj, ni) stack.
        """
        if h is None:
            h = self.h
        self._check_3d(h, 'h')
        eta = np.empty((self.nk + 1, self.nj, self.ni))
        eta[self.nk] = -self.D
        for ki in range(self.nk - 1, -1, -1):
            eta[ki] = eta[ki + 1] + h[ki]
        if k is None:
            return eta
        return eta[k]

    def flat_topog(self):
        """Set bathymetry to flat with no boundaries"""
        self.D = self.Ho.sum() + 0 * self.xh

    def bowl_topog(self):
        """Set bathymetry to bowl shape"""
        self.D = self.Ho.sum() * np.sin( self.xh * np.pi / self.Lx ) * np.sin( self.yh * np.pi / self.Ly )
        self.D[0,:] = 0 # Ensure land along southern edge
        self.D[:,0] = 0 # Ensure land along western edge
        self.D[-1,:] = 0 # Ensure land along northern edge
        self.D[:,-1] = 0 # Ensure land along eastern edge
        self.D = np.array( self.D )

    def zero_forcing(self):
        """Set winds to zero"""
        self.taux = 0 * self.yu # zonal wind stress [m2 s-2]
        self.tauy = 0 * self.xv # meridional wind stress [m2 s-2]

    def gyre_forcing(self, mag=0.1):
        """Set winds to zonal gyre forcing"""
        self.taux = mag * np.sin( self.yu * np.pi / self.Ly ) # zonal wind stress [m2 s-2]
        self.tauy = 0 * self.xu # meridional wind stress [m2 s-2]

    def channel_forcing(self, mag=0.02):
        """Set winds to zonal channel forcing"""
        self.taux = mag * np.sin( 2 * self.yu * np.pi / self.Ly ) # zonal wind stress [m2 s-2]
        self.tauy = 0 * self.xu # meridional wind stress [m2 s-2]

    def perturb_h(self, mag, L, x0, y0=None, k=0):
        """Adds a Gaussian perturbation to the thickness h of layer k (default
        k=0, the top layer), centered at x0,y0, with magnitude mag and length
        scale L. Adding mag to h[k] raises eta_{k-1/2} (the upper interface of
        layer k) by the same amount, leaving lower interfaces unchanged."""
        if k < 0 or k >= self.nk:
            raise ValueError(f"k must be in [0,{self.nk-1}], got {k}")
        r2 = ( ( self.xh - x0 ) / L )**2
        if y0 is not None:
            r2 = r2 + ( ( self.yh - y0 ) / L )**2
        self.h[k] = self.h[k] + mag * np.exp( - 0.5 * r2 )

    def _cubint(x, xa, xb):
        """Returns f(x) with a cubic interpolating between f(xa)=0 and f(xb)=1"""
        # z is non-dimensional coordinate between xa,xb s.t. z(xa)=0 and z(xb)=1
        z = np.minimum( np.maximum( x - xa, 0 ) / ( xb - xa ), 1 )
        # We wand t gradient to be symmetric about z=1/2, positive between z=0 and z=1,
        # and zero at z=0 and z=1. The gradient is thus proportional to 1-(2z-1)^2.
        # Let g(z) = A ( 1 - ( 2 z - 1 )^2 ) = A ( 4 z - 4 z^2 ) = 4 A z ( 1 - z ) then
        # f(z) = int_dz g(z) = C + A ( 2 z^2 - 4/3 z^3 ) = C + A/3 ( 6 - 4 z) z^2
        # f(0)=0 => C=0 and f(1)=1 => A=3/2
        # Thus f(z) = ( 3 - 2 z ) z^2.
        # Note that g(1/2) = 4 . 3/2 . 1/4 = 3/2
        return ( 1 + 2 * ( 1 - z ) ) * z**2

    def _smoothint(x, xa, xb, order=5):                                            
        z = np.minimum( np.maximum( x - xa, 0 ) / ( xb - xa ), 1 )                  
        n = ( int(order) - 1 ) // 2                                                 
        if 2 * n + 1 != int(order) or not ( 1 <= n <= 3 ):                          
            raise ValueError(f"[JET] order must be 3, 5 or 7; got {order}")         
        if n == 1:                                                                  
            return ( 1 + 2 * ( 1 - z ) ) * z**2          # identical to _cubint     
        if n == 2:                                                                  
            return ( ( 6.0 * z - 15.0 ) * z + 10.0 ) * z**3                         
        return ( ( ( -20.0 * z + 70.0 ) * z - 84.0 ) * z + 35.0 ) * z**4

    def _bumpint(y, yc, hw, shape='sech2', taper=None, order=5):                    
        r = np.abs( np.asarray(y, dtype=float) - yc ) / float(hw)                   
        if shape == 'sech2':                                                        
            a = np.arccosh( np.sqrt(2.0) )        # 0.881374; sech^2(a) = 1/2        
            f = 1.0 / np.cosh( a * r )**2           
        if taper is not None:                                                        
            T = float(taper)                                                        
            if T <= 0:                                                              
                raise ValueError(f"[JET] taper must be positive; got {taper}")       
            f = f * ( 1.0 - SSWEM._smoothint(r, T, T + 1.0, order=order) )          
        return f                                                                    

    def set_h_forcing(self, mag, k=0):
        """Sets the meridional zonal-mean restoring profile for layer k. The
        zonal-mean h[k] is restored toward
        h_zonal_target[k, j] = Ho[k] + mag * profile(yh1[j]/Ly).
        Default k=0 (top layer); pass k explicitly to set a different layer.
        Note: only layer 0's target is currently used (h_zonal_relax is scalar
        and the zonal restoring branch in _step_numba acts on layer 0 only)."""
        if k < 0 or k >= self.nk:
            raise ValueError(f"k must be in [0,{self.nk-1}], got {k}")
        profile = ( SSWEM._cubint( self.yh1 / self.Ly, 0.0, 0.1 ) -
                    SSWEM._cubint( self.yh1 / self.Ly, 0.5, 0.6 ) )
        self.h_zonal_target[k, :] = self.Ho[k] + mag * profile

    def set_u_target_jet(self, mag, width=60e3):
        """Sets the u restoring target to a meridional jet profile. mag is the
        per-layer jet amplitude [m s-1]: a scalar (same amplitude in every
        layer) or a length-nk vector (one amplitude per layer). width is the
        jet half-width [m] (default 60 km); the jet is centred on the domain
        midline (y = Ly/2) and spans [Ly/2 - width, Ly/2 + width]."""
        mag = np.atleast_1d(np.asarray(mag, dtype=float))
        if mag.size == 1:
            mag = np.full(self.nk, mag[0])
        elif mag.size != self.nk:
            raise ValueError(f"mag must be a scalar or length nk={self.nk}, got {mag.size}")
        hw = width / self.Ly
        profile = ( SSWEM._cubint( self.yu / self.Ly, 0.5 - hw, 0.5 ) -
                    SSWEM._cubint( self.yu / self.Ly, 0.5, 0.5 + hw ) )
        self.u_target = np.empty_like(self.u)
        for k in range(0,self.nk):
            self.u_target[k, :] = mag[k] * profile

    def run(self, dt, samp, nsamps):
        """
        dt     - Time step [s]
        samp   - Steps between samples [steps]
        nsamps - Number of sample to integrate model [steps*samp]
        """
        # Memory-bandwidth-bound stepper: peak throughput is at the physical
        # core count, set via numba.set_num_threads() (the default may include
        # SMT threads, which can run slower than physical-core-only).
        print("numba threads =", get_num_threads())
        print("CFL: dt*f =", dt * np.abs( self.f.max() ) )
        print("CFL: dt*cg/dx =", dt * self.cg / self.dx )
        if self.cg1 is not None:
            print("CFL: dt*cg1/dx =", dt * self.cg1 / self.dx )
        print("CFL: dt*nu_h/dx^2 =", dt * self.nu_h / self.dx**2 )
        print("CFL*: dt*epsilon/h_bot =", dt * self.epsilon / self.Ho[-1] )
        print("CFL*: dt*nu_v/h_min^2 =", dt * self.nu_v / ( self.Ho.min()**2 ) )
        if self.h_zonal_relax > 0:
            print("CFL*: dt*h_zonal_relax =", dt * self.h_zonal_relax )
        if self.h_interface_relax > 0:                                          # [INTERFACE_RELAX]
            print("CFL*: dt*h_interface_relax =", dt * self.h_interface_relax )  # [INTERFACE_RELAX]
        if self.h_relax.max() > 0:
            print("CFL*: dt*h_relax =", dt * self.h_relax.max() )
        if self.u_relax.max() > 0:
            print("CFL*: dt*u_relax =", dt * self.u_relax.max() )
        if self.v_relax.max() > 0:
            print("CFL*: dt*v_relax =", dt * self.v_relax.max() )
        nsteps = nsamps * samp
        print("nsteps =", nsteps)
        Trun = nsteps * dt
        print("Time: Trun * fo =", Trun * self.fo)
        print("Time: Trun * ( cg / L ) =", Trun * self.cg / self. Lx)
        print("Running...")

        # Pre-allocatge diagnostics to be returned from run()
        u = np.zeros((nsamps+1, self.nk, self.nj, self.ni))
        v = np.zeros((nsamps+1, self.nk, self.nj, self.ni))
        h = np.zeros((nsamps+1, self.nk, self.nj, self.ni))
        time = np.zeros((nsamps+1))

        u[0] = self.u
        v[0] = self.v
        h[0] = self.h
        time[0] = self.time

        nsamp = 0
        for iter in range( 1, nsteps + 1 ):
            self.step( dt )
            if np.any( np.isnan( self.u ) ):
                print('Model has blown up!!! Stopping early')
                u = u[:nsamp]
                v = v[:nsamp]
                h = h[:nsamp]
                time = time[:nsamp]
                break
            if iter % samp == 0:
                nsamp += 1
                u[nsamp] = self.u
                v[nsamp] = self.v
                h[nsamp] = self.h
                time[nsamp] = self.time
        print("...done")
        return u, v, h, time

    # [OBC] Shared CFL/diagnostic banner, factored out so the boundary
    # [OBC] record/replace runs print the same header as run().
    def _print_run_info(self, dt, nsteps):
        print("numba threads =", get_num_threads())
        print("CFL: dt*f =", dt * np.abs( self.f.max() ) )
        print("CFL: dt*cg/dx =", dt * self.cg / self.dx )
        if self.cg1 is not None:
            print("CFL: dt*cg1/dx =", dt * self.cg1 / self.dx )
        print("CFL: dt*nu_h/dx^2 =", dt * self.nu_h / self.dx**2 )
        print("CFL*: dt*epsilon/h_bot =", dt * self.epsilon / self.Ho[-1] )
        print("CFL*: dt*nu_v/h_min^2 =", dt * self.nu_v / ( self.Ho.min()**2 ) )
        if self.h_zonal_relax > 0:
            print("CFL*: dt*h_zonal_relax =", dt * self.h_zonal_relax )
        if self.h_interface_relax > 0:                                          # [INTERFACE_RELAX]
            print("CFL*: dt*h_interface_relax =", dt * self.h_interface_relax )  # [INTERFACE_RELAX]
        if self.h_relax.max() > 0:
            print("CFL*: dt*h_relax =", dt * self.h_relax.max() )
        if self.u_relax.max() > 0:
            print("CFL*: dt*u_relax =", dt * self.u_relax.max() )
        if self.v_relax.max() > 0:
            print("CFL*: dt*v_relax =", dt * self.v_relax.max() )
        print("nsteps =", nsteps)
        Trun = nsteps * dt
        print("Time: Trun * fo =", Trun * self.fo)
        print("Time: Trun * ( cg / L ) =", Trun * self.cg / self.Lx)

    # ================= [CL] per-step centre-line recorder (grafted from the NN module) =================
    def _center_init(self, store_center, center_j, center_stride, center_fields,
                     nsteps, center_i0=0, center_i1=None, i1_default=None):
        """Allocate the per-step centre-row store on self.center_line.
        Columns are restricted to [center_i0, center_i1) -- interior only; the OBC
        drivers pass i1_default = b_obc + 1 so nothing east of the boundary is kept."""
        if not store_center:
            self.center_line = None
            return False, 0, 1, (), slice(0, 0)
        j = self.nj // 2 if center_j is None else int(center_j)               #nj//2 = y ~ Ly/2
        if not (0 <= j < self.nj):
            raise ValueError(f"[CL-REC] center_j={j} out of range [0,{self.nj})")
        i0 = int(center_i0)
        i1 = int(center_i1) if center_i1 is not None else \
             int(i1_default if i1_default is not None else self.ni)           #default = up to & incl. the OBC col
        if not (0 <= i0 < i1 <= self.ni):
            raise ValueError(f"[CL-REC] column range [{i0},{i1}) invalid for ni={self.ni}")
        sl     = slice(i0, i1)
        n_cl   = i1 - i0
        stride = max(1, int(center_stride))
        fields = tuple(f for f in ('h', 'u', 'v') if f in tuple(center_fields))
        nrec   = nsteps // stride
        self.center_line = {f: np.zeros((nrec, self.nk, n_cl), dtype=np.float32)
                            for f in fields}                                  #float32 halves RAM
        self.center_line.update(j=j, i0=i0, i1=i1, cols=np.arange(i0, i1),
                                stride=stride, x_km=self.xh1[sl] / 1e3,
                                y_km=float(self.yh1[j]) / 1e3,
                                step=(np.arange(nrec) + 1) * stride,
                                t=np.zeros(nrec))
        print(f"[CL-REC] centre line: row j={j} (y={self.center_line['y_km']:.0f} km), "
              f"cols i={i0}..{i1-1} ({n_cl} of {self.ni}, interior of the OBC), "
              f"stride={stride} -> {nrec} records, fields={fields}")
        return True, j, stride, fields, sl

    def _center_store(self, on, it, j, stride, fields, sl):
        """Copy row j (interior columns only) of the post-step state."""
        if not on or (it % stride):
            return
        n = it // stride - 1
        src = {'h': self.h, 'u': self.u, 'v': self.v}
        for f in fields:
            self.center_line[f][n] = src[f][:, j, sl]                         #interior slice
        self.center_line['t'][n] = self.time

    def _center_trim(self, on, nvalid, stride):
        """Truncate to the records written before an early (blow-up) exit."""
        if not on:
            return
        n = nvalid // stride
        for k in ('h', 'u', 'v', 't', 'step'):
            if k in self.center_line:
                self.center_line[k] = self.center_line[k][:n]

    # Records (u,v,h) control run
    def run_record_bc(self, dt, samp, nsamps, bc_cols,
                      store_downstream=False, probe_i0=None, n_probe=1,
                      store_center=False, center_j=None, center_stride=1,   # [CL]
                      center_fields=('h', 'u', 'v'),                        # [CL]
                      center_i0=0, center_i1=None):                         # [CL] interior column window
        """
        dt      - Time step [s]
        samp    - Steps between samples [steps]
        nsamps  - Number of samples [steps*samp]
        bc_cols - sequence of column indices to record (any length >= 1; e.g. the
                  two easternmost sponge columns (i_edge-1, i_edge), the default in
                  the notebook). Same i-index used for u, v, h; all rows/layers.

        [OBC] Optional downstream probe (off by default):
        store_downstream - if True, ALSO store (u,v,h) every step at a block of
                  interior columns (a diagnostic probe somewhere downstream; not
                  replaced -- purely recorded). Default False.
        probe_i0 - start column index of the probe (user-defined location).
                  Required when store_downstream=True.
        n_probe  - number of contiguous columns to store, starting at probe_i0
                  (user-defined). Default 1.

        Returns the usual sampled (u, v, h, time), the per-step boundary stores
        h_bc_all, u_bc_all, v_bc_all (each (nsteps, nk, nj, n_bc), n_bc=len(bc_cols)),
        AND a `probe` dict (None when store_downstream=False) with keys 'cols',
        'x_km', and 'h'/'u'/'v' (each (nsteps, nk, nj, n_probe)). The probe is read
        from the post-step state; downstream of the sponge the restoring is zero
        there, so that equals the pre-restoring phase used for the boundary stores.
        """
        bc_cols = np.ascontiguousarray(np.asarray(bc_cols, dtype=np.int64).ravel())  # [OBC] any length
        n_bc = bc_cols.size
        if n_bc < 1 or bc_cols.min() < 0 or bc_cols.max() >= self.ni:
            raise ValueError(f"[OBC] bc_cols {list(bc_cols)} out of range [0, {self.ni})")
        nsteps = nsamps * samp

        # [CL] centre-line setup; no OBC in this driver -> caller supplies center_i1
        _c_on, _c_j, _c_st, _c_f, _c_sl = self._center_init(
            store_center, center_j, center_stride, center_fields, nsteps,
            center_i0, center_i1)

        # [OBC] downstream-probe setup (validate + allocate only when enabled)
        if store_downstream:
            if probe_i0 is None:
                raise ValueError("[OBC] store_downstream=True requires probe_i0 "
                                 "(start column index)")
            probe_i0 = int(probe_i0); n_probe = int(n_probe)
            if n_probe < 1 or probe_i0 < 0 or probe_i0 + n_probe > self.ni:
                raise ValueError(f"[OBC] probe columns [{probe_i0}, "
                                 f"{probe_i0+n_probe}) out of range [0, {self.ni})")
            sl = slice(probe_i0, probe_i0 + n_probe)
            h_probe_all = np.zeros((nsteps, self.nk, self.nj, n_probe))
            u_probe_all = np.zeros((nsteps, self.nk, self.nj, n_probe))
            v_probe_all = np.zeros((nsteps, self.nk, self.nj, n_probe))

        self._print_run_info(dt, nsteps)
        print(f"[OBC] recording (u,v,h) at boundary columns i = {list(bc_cols)}")
        if store_downstream:
            print(f"[OBC] also storing downstream probe at columns "
                  f"i = {probe_i0}..{probe_i0+n_probe-1}")
        print("Running (record)...")

        u = np.zeros((nsamps+1, self.nk, self.nj, self.ni))
        v = np.zeros((nsamps+1, self.nk, self.nj, self.ni))
        h = np.zeros((nsamps+1, self.nk, self.nj, self.ni))
        time = np.zeros((nsamps+1))
        u[0] = self.u; v[0] = self.v; h[0] = self.h; time[0] = self.time

        # [OBC] per-step boundary stores (nsteps, nk, nj, n_bc)
        h_bc_all = np.zeros((nsteps, self.nk, self.nj, n_bc))
        u_bc_all = np.zeros((nsteps, self.nk, self.nj, n_bc))
        v_bc_all = np.zeros((nsteps, self.nk, self.nj, n_bc))

        nsamp = 0
        for iter in range( 1, nsteps + 1 ):
            # [OBC] bc_mode=1 -> _step_numba copies the boundary columns into the
            # [OBC] per-step slices; diffs unused (dummy).
            self._step_core( dt, 1, bc_cols,
                             h_bc_all[iter-1], u_bc_all[iter-1], v_bc_all[iter-1],
                             self._diff_dummy, self._diff_dummy, self._diff_dummy )
            if store_downstream:                             # [OBC] probe: post-step state
                h_probe_all[iter-1] = self.h[:, :, sl]
                u_probe_all[iter-1] = self.u[:, :, sl]
                v_probe_all[iter-1] = self.v[:, :, sl]
            if np.any( np.isnan( self.u ) ):
                print('Model has blown up!!! Stopping early')
                u = u[:nsamp]; v = v[:nsamp]; h = h[:nsamp]; time = time[:nsamp]
                h_bc_all = h_bc_all[:iter-1]                  # [OBC] drop blown step
                u_bc_all = u_bc_all[:iter-1]
                v_bc_all = v_bc_all[:iter-1]
                if store_downstream:                         # [OBC] trim probe too
                    h_probe_all = h_probe_all[:iter-1]
                    u_probe_all = u_probe_all[:iter-1]
                    v_probe_all = v_probe_all[:iter-1]
                self._center_trim(_c_on, iter - 1, _c_st)     # [CL] drop blown step
                break
            self._center_store(_c_on, iter, _c_j, _c_st, _c_f, _c_sl)   # [CL] post-step centre row
            if iter % samp == 0:
                nsamp += 1
                u[nsamp] = self.u; v[nsamp] = self.v
                h[nsamp] = self.h; time[nsamp] = self.time
        print("...done")

        # [OBC] bundle the optional probe (None when disabled) as the last return
        probe = None
        if store_downstream:
            probe = { 'cols': np.arange(probe_i0, probe_i0 + n_probe),
                      'x_km': self.xh1[probe_i0:probe_i0 + n_probe] / 1e3,
                      'h': h_probe_all, 'u': u_probe_all, 'v': v_probe_all }
        return u, v, h, time, h_bc_all, u_bc_all, v_bc_all, probe

    # Control run with switchable per-step stores: a superset of run() and
    # run_record_bc(). With store_bc=True it reproduces run_record_bc step for
    # step (same _step_core(bc_mode=1) path); with store_bc=False it is the
    # clean run() path, for a control run used only for animations.
    def run_control(self, dt, samp, nsamps, bc_cols,
                    store_bc=True, store_downstream=True, probe_cols = None,
                    store_row=False, j_row=None, row_q=False):
        """
        dt      - Time step [s]
        samp    - Steps between samples [steps]
        nsamps  - Number of samples [steps*samp]
        bc_cols - sequence of column indices to record when store_bc=True (any
                  length >= 1; e.g. the westernmost sponge columns). Same i-index
                  used for u, v, h; all rows/layers. Ignored when store_bc=False.

        store_bc         - True (default): record (u,v,h) at bc_cols every step,
                  the data a later run_replace_bc / run_obc replays. False: clean
                  run with no per-step boundary stores; h/u/v_bc_all return None.
        store_downstream - if True, ALSO store (u,v,h) every step at an arbitrary
                  set of interior columns (a diagnostic probe somewhere
                  downstream; not replaced -- purely recorded). Independent of
                  store_bc.
        probe_cols - sequence of column indices to store (any length >= 1, need
                  not be contiguous). Required when store_downstream=True.       # [OBC]
        store_row        - if True, ALSO store (u,v,h) every step along the whole
                  zonal row j_row (all i), i.e. a step-cadence Hovmoller of the
                  run. Cheap: (nsteps, nk, ni) per field. Default False.
        j_row    - row index of the stored line (default None -> nj//2, the
                  domain centre).
        row_q    - if True, ALSO store the potential vorticity q along the same
                  row (requires store_row). q is formed DURING the run from the
                  full 2D state, because q at row j needs row j-1 as well
                  (q = (f + v_x - u_y)/h_q, and both u_y and h_q straddle j-1),
                  so it cannot be reconstructed afterwards from the stored row.
                  Uses exactly the stencil of q(): h_q = u2q(h2u(h)), and the
                  same (h_q/(h_q+hsub)) dry mask, so the stored values match
                  q(k=...)[j_row, :] of the corresponding state. Adds one more
                  (nsteps, nk, ni) array. Default False.

        Returns the usual sampled (u, v, h, time), the per-step boundary stores
        h_bc_all, u_bc_all, v_bc_all (each (nsteps, nk, nj, n_bc), n_bc=len(bc_cols);
        None when store_bc=False), a `probe` dict (None when store_downstream=False)
        with keys 'cols', 'x_km', and 'h'/'u'/'v' (each (nsteps, nk, nj, n_probe),
        n_probe=len(probe_cols)),
        AND a `row` dict (None when store_row=False) with keys 'j_row', 'y_km_h',
        'y_km_v', 'y_km_q', 'x_km_h', 'x_km_u', 'x_km_q', 't' and 'h'/'u'/'v'
        (each (nsteps, nk, ni)), plus 'q' when row_q=True.
        h and u are stored on y = yh1[j_row], v and q on y = yq1[j_row]; h and v
        are on the xh1 columns, u and q on xq1. The probe and row stores are read
        from the post-step state; downstream of the sponge the restoring is zero
        there, so that equals the pre-restoring phase used for the boundary stores.
        """
        samp = int(samp); nsamps = int(nsamps); nsteps = samp * nsamps
        bc_cols = np.ascontiguousarray(np.asarray(bc_cols, dtype=np.int64).ravel())
        if store_bc and (bc_cols.size < 1 or bc_cols.min() < 0
                         or bc_cols.max() >= self.ni):
            raise ValueError(f"[OBC] bc_cols {list(bc_cols)} out of range [0, {self.ni})")

        # Downstream-probe setup (validate + allocate only when enabled)
        if store_downstream:
            if probe_cols is None:                                              # [OBC]
                raise ValueError("[OBC] store_downstream=True requires probe_cols "
                                 "(sequence of column indices)")
            probe_cols = np.asarray(probe_cols, dtype=np.int64).ravel()
            n_probe = len(probe_cols)
            if n_probe < 1 or probe_cols.min() < 0 or probe_cols.max() >= self.ni:
                raise ValueError(f"[OBC] probe columns {list(probe_cols)} out of range [0, {self.ni})")
            h_probe_all = np.zeros((nsteps, self.nk, self.nj, n_probe))
            u_probe_all = np.zeros((nsteps, self.nk, self.nj, n_probe))
            v_probe_all = np.zeros((nsteps, self.nk, self.nj, n_probe))

        # Zonal-row setup: store the whole row jr (all i) every step.
        if store_row:
            jr = self.nj // 2 if j_row is None else int(j_row)
            if not (0 <= jr < self.nj):
                raise ValueError(f"[OBC] j_row {jr} out of range [0, {self.nj})")
            h_row = np.zeros((nsteps, self.nk, self.ni))
            u_row = np.zeros((nsteps, self.nk, self.ni))
            v_row = np.zeros((nsteps, self.nk, self.ni))
            t_row = np.zeros(nsteps)
            # q at row jr needs row jr-1 too, so it is formed here, per step, from
            # the full state; jrm follows the j-periodic u2q/h2u stencil of q().
            q_row = np.zeros((nsteps, self.nk, self.ni)) if row_q else None
            jrm = jr - 1 if jr > 0 else self.nj - 1
        elif row_q:
            raise ValueError("[OBC] row_q=True requires store_row=True")

        self._print_run_info(dt, nsteps)
        if store_bc:
            print(f"[OBC] recording (u,v,h) at boundary columns i = {list(bc_cols)}")
            h_bc_all = np.zeros((nsteps, self.nk, self.nj, bc_cols.size))
            u_bc_all = np.zeros_like(h_bc_all)
            v_bc_all = np.zeros_like(h_bc_all)
        else:
            print("[OBC] clean run: no per-step boundary stores "
                  "(run_replace_bc / run_obc need store_bc=True)")
            h_bc_all = u_bc_all = v_bc_all = None
        if store_downstream:
            print(f"[OBC] also storing downstream probe at columns "
                  f"i = {list(probe_cols)}")                                     # [OBC]
        if store_row:
            print(f"[OBC] also storing (u,v,h{',q' if row_q else ''}) every step along "
                  f"row j = {jr} (y = {self.yh1[jr] / 1e3:.0f} km): "
                  f"~{(4 if row_q else 3) * nsteps * self.nk * self.ni * 8 / 1e9:.2f} GB")
        print("Running (record)..." if store_bc else "Running (clean)...")

        u = np.zeros((nsamps+1, self.nk, self.nj, self.ni))
        v = np.zeros((nsamps+1, self.nk, self.nj, self.ni))
        h = np.zeros((nsamps+1, self.nk, self.nj, self.ni))
        time = np.zeros((nsamps+1))
        u[0] = self.u; v[0] = self.v; h[0] = self.h; time[0] = self.time

        nsamp = 0
        for iter in range( 1, nsteps + 1 ):
            if store_bc:
                # bc_mode=1 -> _step_numba copies the boundary columns into the
                # per-step slices; diffs unused (dummy).
                self._step_core( dt, 1, bc_cols,
                                 h_bc_all[iter-1], u_bc_all[iter-1], v_bc_all[iter-1],
                                 self._diff_dummy, self._diff_dummy, self._diff_dummy )
            else:
                self.step( dt )
            if store_downstream:                             # probe: post-step state
                h_probe_all[iter-1] = self.h[:, :, probe_cols]
                u_probe_all[iter-1] = self.u[:, :, probe_cols]
                v_probe_all[iter-1] = self.v[:, :, probe_cols]
            if store_row:                                    # zonal row: post-step state
                h_row[iter-1] = self.h[:, jr, :]
                u_row[iter-1] = self.u[:, jr, :]
                v_row[iter-1] = self.v[:, jr, :]
                t_row[iter-1] = self.time
                if row_q:
                    # q at the q-points of row jr, same stencil as q(): h_q is the
                    # 4-point average over (jr,jrm) x (i,i-1), v_x the i-difference
                    # of v on row jr, u_y the (jr,jrm) difference of u. i-1 wraps
                    # (x-periodic channel), matching _nb_im1.
                    hq = 0.25 * ( self.h[:, jr, :] + np.roll(self.h[:, jr, :], 1, axis=-1)
                                + self.h[:, jrm, :] + np.roll(self.h[:, jrm, :], 1, axis=-1) )
                    vx = ( self.v[:, jr, :] - np.roll(self.v[:, jr, :], 1, axis=-1) ) / self.dx
                    uy = ( self.u[:, jr, :] - self.u[:, jrm, :] ) / self.dy
                    rhq = 1.0 / ( hq + self.hsub )
                    q_row[iter-1] = ( self.f[jr, :] + ( vx - uy ) ) * rhq * ( hq * rhq )
            if np.any( np.isnan( self.u ) ):
                print('Model has blown up!!! Stopping early')
                u = u[:nsamp]; v = v[:nsamp]; h = h[:nsamp]; time = time[:nsamp]
                if store_bc:
                    h_bc_all = h_bc_all[:iter-1]              # drop blown step
                    u_bc_all = u_bc_all[:iter-1]
                    v_bc_all = v_bc_all[:iter-1]
                if store_downstream:                         # trim probe too
                    h_probe_all = h_probe_all[:iter-1]
                    u_probe_all = u_probe_all[:iter-1]
                    v_probe_all = v_probe_all[:iter-1]
                if store_row:                                # trim row store too
                    h_row = h_row[:iter-1]; u_row = u_row[:iter-1]
                    v_row = v_row[:iter-1]; t_row = t_row[:iter-1]
                    if row_q: q_row = q_row[:iter-1]
                break
            if iter % samp == 0:
                nsamp += 1
                u[nsamp] = self.u; v[nsamp] = self.v
                h[nsamp] = self.h; time[nsamp] = self.time
        print("...done")

        # Bundle the optional stores (None when disabled) as the last returns
        probe = None
        if store_downstream:
            probe = { 'cols': probe_cols,                                       # [OBC]
                      'x_km': self.xh1[probe_cols] / 1e3,                       # [OBC]
                      'h': h_probe_all, 'u': u_probe_all, 'v': v_probe_all }
        row = None
        if store_row:
            row = { 'j_row': jr,
                    'y_km_h': self.yh1[jr] / 1e3, 'y_km_v': self.yq1[jr] / 1e3,
                    'y_km_q': self.yq1[jr] / 1e3,          # q shares the v rows (yq1)
                    'x_km_h': self.xh1 / 1e3, 'x_km_u': self.xq1 / 1e3,
                    'x_km_q': self.xq1 / 1e3,              # q shares the u columns (xq1)
                    't': t_row, 'h': h_row, 'u': u_row, 'v': v_row }
            if row_q:
                row['q'] = q_row
        return u, v, h, time, h_bc_all, u_bc_all, v_bc_all, probe, row

    # Replace the western sponge layers and compare it before replacing (for testing).
    def run_replace_bc(self, dt, samp, nsamps, bc_cols, h_bc_all, u_bc_all, v_bc_all):
        """
        dt, samp, nsamps - as run().
        bc_cols          - column indices (any length); MUST match run_record_bc.
        h_bc_all, u_bc_all, v_bc_all - per-step stores from run_record_bc.
        Returns sampled (u, v, h, time) PLUS a `diffs` dict of per-step misfit
        time series ('t_step', and 'h_max'/'h_rms'/'u_max'/'u_rms'/'v_max'/'v_rms').
        """
        bc_cols = np.ascontiguousarray(np.asarray(bc_cols, dtype=np.int64).ravel())  # [OBC] any length
        n_bc = bc_cols.size
        nsteps = nsamps * samp
        if h_bc_all.shape[0] < nsteps:                        # [OBC] need full coverage
            raise ValueError(f"[OBC] stored bc data has {h_bc_all.shape[0]} steps "
                             f"< nsteps={nsteps}; re-record with matching run_params")
        if h_bc_all.shape[-1] != n_bc:                        # [OBC] stores must match bc_cols
            raise ValueError(f"[OBC] stored bc data has {h_bc_all.shape[-1]} columns "
                             f"!= len(bc_cols)={n_bc}; pass the matching bc_cols")
        self._print_run_info(dt, nsteps)
        print(f"[OBC] replacing (u,v,h) at boundary columns i = {list(bc_cols)} with stored data")
        print("Running (replace)...")

        u = np.zeros((nsamps+1, self.nk, self.nj, self.ni))
        v = np.zeros((nsamps+1, self.nk, self.nj, self.ni))
        h = np.zeros((nsamps+1, self.nk, self.nj, self.ni))
        time = np.zeros((nsamps+1))
        u[0] = self.u; v[0] = self.v; h[0] = self.h; time[0] = self.time

        # [OBC] per-step misfit series [max_abs, rms] for h, u, v
        h_dmax = np.zeros(nsteps); h_drms = np.zeros(nsteps)
        u_dmax = np.zeros(nsteps); u_drms = np.zeros(nsteps)
        v_dmax = np.zeros(nsteps); v_drms = np.zeros(nsteps)
        h_diff = np.zeros(2); u_diff = np.zeros(2); v_diff = np.zeros(2)

        nsamp = 0
        nrun = nsteps
        for iter in range( 1, nsteps + 1 ):
            # [OBC] bc_mode=2 -> _step_numba writes [max_abs, rms] into *_diff then
            # [OBC] overwrites the boundary columns from the stored slices.
            self._step_core( dt, 2, bc_cols,
                             h_bc_all[iter-1], u_bc_all[iter-1], v_bc_all[iter-1],
                             h_diff, u_diff, v_diff )
            h_dmax[iter-1] = h_diff[0]; h_drms[iter-1] = h_diff[1]
            u_dmax[iter-1] = u_diff[0]; u_drms[iter-1] = u_diff[1]
            v_dmax[iter-1] = v_diff[0]; v_drms[iter-1] = v_diff[1]
            if np.any( np.isnan( self.u ) ):
                print('Model has blown up!!! Stopping early')
                u = u[:nsamp]; v = v[:nsamp]; h = h[:nsamp]; time = time[:nsamp]
                nrun = iter                                   # [OBC] steps attempted
                break
            if iter % samp == 0:
                nsamp += 1
                u[nsamp] = self.u; v[nsamp] = self.v
                h[nsamp] = self.h; time[nsamp] = self.time
        print("...done")
        # [OBC] assemble diff series (trimmed to executed steps) + summary
        diffs = { 't_step': np.arange(1, nrun+1) * dt,
                  'h_max': h_dmax[:nrun], 'h_rms': h_drms[:nrun],
                  'u_max': u_dmax[:nrun], 'u_rms': u_drms[:nrun],
                  'v_max': v_dmax[:nrun], 'v_rms': v_drms[:nrun] }
        print(f"[OBC] max over run:  |dh|={diffs['h_max'].max():.3e}  "
              f"|du|={diffs['u_max'].max():.3e}  |dv|={diffs['v_max'].max():.3e}")
        return u, v, h, time, diffs

    # [OBC - Orlanski] OBC run 
    def run_obc(self, dt, samp, nsampes, prev_cols,
                h_bc_all, u_bc_all, v_bc_all, b_obc,
                nudging=False, h_ext_all=None, u_ext_all=None, v_ext_all=None,
                alpha_in=0.5, h_bc='radiate', 
                store_downstream=False, probe_i0=None, n_probe=1, 
                inflow_persistent=False,  # [PERSIST] True -> clamp rx>=0 so inflow rows persist (phi_b^{n+1}=phi_b^n) instead of reading phi_ext
                ry_2d=True,  # [RY1D] True -> 2D Orlanski; False -> 1D Orlanski (ry == 0), everything else identical
                r_max=R_MAX_DEFAULT,  # [RMAX] outflow CFL clamp on |rx|,|ry|; default 1/50 = this module's historical value
                inflow_mode=0,  # [IM] east inflow branch: 0 = prescribe/nudge toward col-b data (unchanged); 1 = radiation-inflow scheme reading col-(b+1) data
                h_ext_out_all=None, u_ext_out_all=None, v_ext_out_all=None,  # [IM] per-step col-(b+1) exterior data @ n+1, (>=nsteps,nk,nj[,1]); REQUIRED when inflow_mode=1
                obc_order=1,   # [O2] 1 (default, bit-identical to before) | 2 = 2nd-order one-sided D_t/D_x in the phase-speed estimator (reads column b-3 and time level n-1; dphi_y and the update law stay 1st order; step 1 falls back to 1st order)
                eps_denom=None,  # [EPSREL] (3,nk) per-(field,layer) small-|grad phi|^2 floor, rows = h,u,v. None (default) -> uniform 1e-3, bit-identical to before. A uniform floor is NOT scale-invariant: it binds on ~55%/95% of u,v rows (layer 0/1) but 0% of h rows, and a floored row radiates less (|rx| shrunk toward 0 -> persistence). Build it as eps[c,k] = 1e-3 * (s[c,k]/s[h,k])**2 with s = median sqrt(dphi_x^2+dphi_y^2), which leaves h EXACTLY unchanged.
                store_center=False, center_j=None, center_stride=1,   # [CL]
                center_fields=('h', 'u', 'v'),                        # [CL]
                center_i0=0, center_i1=None):                         # [CL] None -> b_obc+1 (interior of the OBC)
        """West: prescribe (replace) `prev_cols` from stored data each step (the existing  # [OBC-E]
        sponge-edge band). East: 1D Orlanski radiation at column `b_obc`, computed only    # [OBC-E]
        from interior columns b-1,b-2 (interior-determined; periodicity kept).             # [OBC-E]
        Returns (u,v,h,time), `diffs` (west misfit, ~0), and `rx` (east phase-speed diag).

        Each field (h, u, v) estimates its OWN east-boundary phase speed r_x      # [OBC - Orlanski - 2D - IndepRx]
        independently (no shared / reused phase speed).                           # [OBC - Orlanski - 2D - IndepRx]

        nudging : INFLOW treatment knob for the east OBC column.
            False (default) -> PRESCRIBE scheme: on inflow hard-set column b to     # [OBC - Orlanski - 2D - Prescribe]
                  the recorded external data (h/u/v_ext_all), as in the 1D          # [OBC - Orlanski - 2D - Prescribe]
                  OSSWEM_obc_orlanski.                                              # [OBC - Orlanski - 2D - Prescribe]
            True  -> NUDGING scheme: on inflow soft-prescribe column b toward the
                  recorded external data (h/u/v_ext_all) with coefficient alpha_in, so
                  the ghost wall is removed (pass prev_cols WITHOUT b+1).
        h_ext_all, u_ext_all, v_ext_all : per-step external (recorded) col-b data,       
            shape (>=nsteps, nk, nj) or (>=nsteps, nk, nj, 1) (e.g. a run_record_bc     
            probe at probe_i0=b_obc). REQUIRED when nudging=True, ignored otherwise.    
        alpha_in : inflow nudging coefficient in (0,1]; 1 == hard prescribe.

        inflow_mode : [IM] east inflow-branch selector.
            0 (default) -> unchanged: prescribe/nudge toward the col-b data.
            1 -> radiation-INFLOW scheme,
                   phi_b^{n+1} = (phi_b^n - rx*phi_ext_out - ry*dy_b) / (1 - rx),  rx < 0,
                 reading the col-(b+1) exterior data (h/u/v_ext_out_all) recorded at
                 n+1; with nudging=True the result is additionally nudged toward the
                 col-b data with alpha_in. |rx| is clamped at r_max (symmetric with
                 the outflow clamp).
        h_ext_out_all, u_ext_out_all, v_ext_out_all : [IM] per-step external col-(b+1)
            data at n+1, shape (>=nsteps,nk,nj) or (...,1) -- same per-step indexing
            convention as h/u/v_ext_all (slice [it-1] is the value at the time level
            the step lands on). REQUIRED when inflow_mode=1, ignored otherwise.
        store_downstream : if True, ALSO store (u,v,h) EVERY STEP at a block of
            columns (post-step state, same convention as run_record_bc's probe).       
            E.g. probe_i0=b_obc-1, n_probe=1 gives step-cadence data at b-1 for        
            theory-vs-r_x boundary diagnostics. Returned as rx['probe'] (a dict with   
            'cols', 'x_km', 'h'/'u'/'v', each (nrun,nk,nj,n_probe); None when off),    
            so the 6-value return signature is unchanged.                              
        probe_i0, n_probe : start column index and number of contiguous columns.       
        """
        prev_cols = np.ascontiguousarray(np.asarray(prev_cols, dtype=np.int64).ravel())
        b = int(b_obc)
        nudging_mode = int(bool(nudging))  # [OBC - Orlanski - 2D - Nuding] 0=prescribe, 1=inflow nudging
        alpha_in = float(alpha_in)         # [OBC - Orlanski - 2D - Nuding]
        h_zerograd = int(h_bc == 'zerograd')   # h OBC mode: 'radiate' (default) or 'zerograd' (dh/dx=0, paper Table 1)
        inflow_persistent_mode = int(bool(inflow_persistent))  # [PERSIST] 0=inflow uses phi_ext (prescribe/nudge), 1=inflow persists
        ry_on = int(bool(ry_2d))  # [RY1D] 0 -> tangential term dropped from denominator AND update -> 1D Orlanski
        r_max = float(r_max)  # [RMAX] outflow CFL clamp
        inflow_mode = int(inflow_mode)  # [IM] 0 = unchanged prescribe/nudge; 1 = radiation-inflow scheme

        nsteps = nsampes * samp
        # Check if the shape of the stored boundary data matches the expected shape based on nsteps and bc_cols.
        if h_bc_all.shape[0] < nsteps:
            raise ValueError(f"[OBC-E] stored west data has {h_bc_all.shape[0]} < nsteps={nsteps}")  # [OBC-E]

        if h_ext_all is None or u_ext_all is None or v_ext_all is None:
            raise ValueError("[OBC - Orlanski - 2D - Nuding] run_obc needs h_ext_all/u_ext_all/"
                             "v_ext_all (recorded col-b external data, e.g. run_record_bc probe "
                             "with probe_i0=b_obc).")
        # Prepare the exterior data
        def _prep_ext(a, name):
            a = np.asarray(a, dtype=np.float64)
            if a.ndim == 4 and a.shape[-1] == 1:
                a = a[..., 0]
            if a.ndim != 3 or a.shape[1:] != (self.nk, self.nj):
                raise ValueError(f"[OBC - Orlanski - 2D - Nuding] {name} must be (>=nsteps,nk,nj) "
                                 f"or (>=nsteps,nk,nj,1); got {a.shape}")
            if a.shape[0] < nsteps:
                raise ValueError(f"[OBC - Orlanski - 2D - Nuding] {name} has {a.shape[0]} < "
                                 f"nsteps={nsteps}")
            return np.ascontiguousarray(a)
        h_ext_all = _prep_ext(h_ext_all, "h_ext_all")
        u_ext_all = _prep_ext(u_ext_all, "u_ext_all")
        v_ext_all = _prep_ext(v_ext_all, "v_ext_all")

        # [IM] col-(b+1) exterior data for the radiation-inflow scheme (recorded at n+1,
        # [IM] same per-step indexing convention as h/u/v_ext_all)
        if inflow_mode == 1:                                                           
            if h_ext_out_all is None or u_ext_out_all is None or v_ext_out_all is None: 
                raise ValueError("[IM] inflow_mode=1 needs h_ext_out_all/u_ext_out_all/"  
                                 "v_ext_out_all (recorded col-(b+1) external data at n+1, "  
                                 "e.g. the run_control probe at b_obc+1).")           
            h_ext_out_all = _prep_ext(h_ext_out_all, "h_ext_out_all")                  
            u_ext_out_all = _prep_ext(u_ext_out_all, "u_ext_out_all")                  
            v_ext_out_all = _prep_ext(v_ext_out_all, "v_ext_out_all")                  

        # optional per-step probe setup (validate + allocate only when enabled)
        if store_downstream:                                                           
            if probe_i0 is None:                                                       
                raise ValueError("[OBC - 2D - Probe] store_downstream=True requires "  
                                 "probe_i0 (start column index)")                      
            probe_i0 = int(probe_i0); n_probe = int(n_probe)                           
            if n_probe < 1 or probe_i0 < 0 or probe_i0 + n_probe > self.ni:            
                raise ValueError(f"[OBC - 2D - Probe] probe columns [{probe_i0}, "     
                                 f"{probe_i0 + n_probe}) out of range [0, {self.ni})") 
            sl_probe = slice(probe_i0, probe_i0 + n_probe)                             
            h_probe_all = np.zeros((nsteps, self.nk, self.nj, n_probe))                
            u_probe_all = np.zeros((nsteps, self.nk, self.nj, n_probe))                
            v_probe_all = np.zeros((nsteps, self.nk, self.nj, n_probe))                

        # Check if the location of the Western open boundary lies in the interior
        if not (2 <= b < self.ni - 1):
            raise ValueError(f"[OBC-B] b_obc={b} needs 2 <= b < ni-1={self.ni-1}")
        if np.any((prev_cols >= b - 2) & (prev_cols <= b)):
            raise ValueError(f"[OBC-B] prescribed cols must avoid the OBC stencil "
                             f"{{{b-2},{b-1},{b}}}; got {list(prev_cols)}")
        eps_denom = self._check_eps_denom(eps_denom)                                 # [EPSREL] validate + normalize to (3,nk) C-contiguous float64
        if int(obc_order) not in (1, 2):                                             # [O2]
            raise ValueError(f'[O2] obc_order must be 1 or 2, got {obc_order!r}')    # [O2]
        if int(obc_order) == 2:                                                      # [O2] gated on order 2: order-1 configs stay legal
            if b < 3:                                                                # [O2]
                raise ValueError(f'[O2] obc_order=2 reads column b-3: need b >= 3, got b={b}')          # [O2]
            if np.any((prev_cols >= b - 3) & (prev_cols <= b)):                      # [O2]
                raise ValueError(f'[O2] obc_order=2 widens the stencil to {{{b-3}..{b}}}; '             # [O2]
                                 f'prescribed cols {list(prev_cols)} overlap it')    # [O2]
        self._print_run_info(dt, nsteps)  

        u = np.zeros((nsampes+1, self.nk, self.nj, self.ni))
        v = np.zeros((nsampes+1, self.nk, self.nj, self.ni))
        h = np.zeros((nsampes+1, self.nk, self.nj, self.ni))
        time = np.zeros((nsampes+1))
        u[0] = self.u; v[0] = self.v; h[0] = self.h; time[0] = self.time

        # [CL] centre-line setup; default cut = up to and including the OBC column b
        _c_on, _c_j, _c_st, _c_f, _c_sl = self._center_init(
            store_center, center_j, center_stride, center_fields, nsteps,
            center_i0, center_i1, i1_default=b + 1)

        # phi_n at [b-1, b]; slot 2 = phi^{n-1}_{b-1} ([O2], read only at obc_order==2)
        h_prev = np.zeros((self.nk, self.nj, 3))   # [O2] widened 2 -> 3
        u_prev = np.zeros((self.nk, self.nj, 3))   # [O2]
        v_prev = np.zeros((self.nk, self.nj, 3))   # [O2]

        rx_h = np.zeros(2); rx_u = np.zeros(2); rx_v = np.zeros(2)  # [OBC - Orlanski] phase-speed diags at b-1,b
        h_diff = np.zeros(2); u_diff = np.zeros(2); v_diff = np.zeros(2)  # [OBC - Orlanski] diffs for diagnostics
        rxh = np.zeros((nsteps, 2)); rxu = np.zeros((nsteps, 2)); rxv = np.zeros((nsteps, 2))  # [OBC - Orlanski] phase speeds for diagnostics
        # per-row, per-step APPLIED (rx,ry) for each field: (nsteps,nk,nj,3). [...,0]=rx, [...,1]=ry,
        # [...,2]=diag bitmask (1=denom floor, 2=rx clamp, 4=ry clamp)  [O2]
        rxy_h_row = np.zeros((nsteps, self.nk, self.nj, 3))   # [O2] widened 2 -> 3
        rxy_u_row = np.zeros((nsteps, self.nk, self.nj, 3))   # [O2]
        rxy_v_row = np.zeros((nsteps, self.nk, self.nj, 3))   # [O2]
        if obc_order == 2:                                                              # [O2]
            print('[O2] east phase-speed estimator: 2ND-ORDER one-sided D_t/D_x '       # [O2]
                  '(dphi_y and the update law stay 1st order; step 1 falls back to 1st order)')  # [O2]
        if np.allclose(eps_denom, 1e-3):                                                # [EPSREL]
            print('[EPSREL] denominator floor: uniform 1e-3 (historical absolute value)')   # [EPSREL]
        else:                                                                           # [EPSREL]
            print('[EPSREL] denominator floor per (field, layer):')                      # [EPSREL]
            for _c, _fn in enumerate(('h', 'u', 'v')):                                   # [EPSREL]
                print('[EPSREL]   %s: %s' % (_fn, np.array2string(eps_denom[_c], precision=4)))  # [EPSREL]
        print("[OBC - Orlanski - 2D - IndepRx] east r_x estimated independently per field (h, u, v)")  # [OBC - Orlanski - 2D - IndepRx]
        print(f"[RY1D] east OBC scheme: {'2D Orlanski (r_y active)' if ry_on else '1D Orlanski (r_y = 0)'}")  # [RY1D]
        print(f"[OBC - Orlanski - 2D - Nuding] east inflow scheme: "  # [OBC - Orlanski - 2D - Nuding]
              f"{f'NUDGING (alpha_in={alpha_in:.3g}, ghost wall removed)' if nudging_mode else 'PRESCRIBE (hard-set col b = external)'}")  # [OBC - Orlanski - 2D - Prescribe]
        if inflow_mode == 1:                                                           
            print(f"[IM] inflow_mode=1: RADIATION-INFLOW scheme on inflow rows "      
                  f"(reads col b+1 exterior @ n+1; |rx| clamped at r_max={r_max:.3g}; "  
                  f"extra col-b nudge {'ON, alpha_in=%.3g' % alpha_in if nudging_mode else 'OFF'})")  

        nsamp = 0
        nrun = nsteps
        for it in range(1, nsteps + 1):
            # [O2] retire slot 0 (phi^{n-1}_{b-1} for the step about to run) BEFORE the
            # [O2] refresh below overwrites it -- reversed order would silently make slot 0
            # [O2] the n-1 value and corrupt the FIRST-order path (B.3b drift mode #2).
            h_prev[:, :, 2] = h_prev[:, :, 0]                                   # [O2]
            u_prev[:, :, 2] = u_prev[:, :, 0]                                   # [O2]
            v_prev[:, :, 2] = v_prev[:, :, 0]                                   # [O2]
            # snap shot phi^n before the step
            h_prev[:, :, 0:2] = self.h[:, :, b-1:b+1]   # [O2] slice (buffer now 3-wide)
            u_prev[:, :, 0:2] = self.u[:, :, b-1:b+1]   # [O2]
            v_prev[:, :, 0:2] = self.v[:, :, b-1:b+1]   # [O2]
            # [O2] no n-1 level exists at step 1: fall back to the 1st-order estimator for
            # [O2] that one step (do NOT seed slot 2 with the n value -- that would be a
            # [O2] 1st-order estimate wearing a 2nd-order label).
            _ord_eff = 1 if (obc_order == 2 and it == 1) else int(obc_order)    # [O2]
            if obc_order == 2 and it == 1:                                      # [O2]
                print('[O2] step 1: no n-1 level yet -> 1st-order estimator for this step only')  # [O2]

            # external col-b slice; read on inflow in BOTH modes
            h_ext_s = h_ext_all[it - 1]; u_ext_s = u_ext_all[it - 1]; v_ext_s = v_ext_all[it - 1]
            # [IM] col-(b+1) exterior slice @ n+1 (None -> _step_core substitutes the dummy)
            if inflow_mode == 1:                                                                       
                h_ext_out_s = h_ext_out_all[it - 1]; u_ext_out_s = u_ext_out_all[it - 1]               
                v_ext_out_s = v_ext_out_all[it - 1]                                                    
            else:                                                                                      
                h_ext_out_s = u_ext_out_s = v_ext_out_s = None                                         
            # advance
            # To turn off prescribing, set the second argument to 0.
            self._step_core(dt, 2, prev_cols,
                            h_bc_all[it - 1], u_bc_all[it - 1], v_bc_all[it - 1],
                            h_diff, u_diff, v_diff,
                            obc_on = 1, b_obc = b,
                            h_prev = h_prev, u_prev = u_prev, v_prev = v_prev,
                            rx_h = rx_h, rx_u = rx_u, rx_v = rx_v,
                            rxy_h = rxy_h_row[it-1], rxy_u = rxy_u_row[it-1], rxy_v = rxy_v_row[it-1],  # [OBC - 2D - PhaseDiag] per-row (rx,ry) into this step's slice
                            nudging_mode = nudging_mode, h_ext = h_ext_s,  # [OBC - Orlanski - 2D - Nuding]
                            u_ext = u_ext_s, v_ext = v_ext_s, alpha_in = alpha_in,  # [OBC - Orlanski - 2D - Nuding]
                            h_zerograd = h_zerograd,
                            inflow_persistent = inflow_persistent_mode,  # [PERSIST]
                            ry_on = ry_on,  # [RY1D]
                            r_max = r_max,  # [RMAX]
                            inflow_mode = inflow_mode,  # [IM]
                            h_ext_out = h_ext_out_s, u_ext_out = u_ext_out_s, v_ext_out = v_ext_out_s,  # [IM]
                            obc_order = _ord_eff,  # [O2] 1st-order fallback at step 1, obc_order after
                            eps_denom = eps_denom)  # [EPSREL]
            rxh[it-1] = rx_h; rxu[it-1] = rx_u; rxv[it-1] = rx_v  # [OBC - Orlanski] store phase speeds for diag
            if store_downstream:                                  # post-step state (same convention as run_record_bc)
                h_probe_all[it-1] = self.h[:, :, sl_probe]        
                u_probe_all[it-1] = self.u[:, :, sl_probe]        
                v_probe_all[it-1] = self.v[:, :, sl_probe]        

            if np.any(np.isnan(self.u)):
                print('Model has blown up!!! Stopping early')
                u = u[:nsamp]; v = v[:nsamp]; h = h[:nsamp]; time = time[:nsamp]
                nrun = it  # [OBC-E] steps attempted
                self._center_trim(_c_on, it - 1, _c_st)      # [CL] drop blown step
                break
            self._center_store(_c_on, it, _c_j, _c_st, _c_f, _c_sl)   # [CL] post-step centre row
            if it % samp == 0:
                nsamp += 1
                u[nsamp] = self.u; v[nsamp] = self.v
                h[nsamp] = self.h; time[nsamp] = self.time
            
        print("...done")

        diffs = {'t_step': np.arange(1, nrun+1) * dt,
                    'h_max': None}
        
        rx = {'t_step': np.arange(1, nrun+1) * dt,
                'h_mean': rxh[:nrun, 0], 'hmax': rxh[:nrun, 1],
                'u_mean': rxu[:nrun, 0], 'umax': rxu[:nrun, 1],
                'v_mean': rxv[:nrun, 0], 'vmax': rxv[:nrun, 1],
                # per-row APPLIED phase-speed Courant numbers, each (nrun,nk,nj).
                'rx_h_row': rxy_h_row[:nrun, :, :, 0], 'ry_h_row': rxy_h_row[:nrun, :, :, 1],
                'rx_u_row': rxy_u_row[:nrun, :, :, 0], 'ry_u_row': rxy_u_row[:nrun, :, :, 1],
                'rx_v_row': rxy_v_row[:nrun, :, :, 0], 'ry_v_row': rxy_v_row[:nrun, :, :, 1],
                # [O2] per-row diag bitmask (1=denom floor, 2=rx clamp, 4=ry clamp), (nrun,nk,nj)
                'flags_h_row': rxy_h_row[:nrun, :, :, 2],                       # [O2]
                'flags_u_row': rxy_u_row[:nrun, :, :, 2],                       # [O2]
                'flags_v_row': rxy_v_row[:nrun, :, :, 2]}                       # [O2]
        # bundle the optional per-step probe into rx (return signature unchanged)
        probe = None                                                                   
        if store_downstream:                                                           
            probe = { 'cols': np.arange(probe_i0, probe_i0 + n_probe),                 
                      'x_km': self.xh1[probe_i0:probe_i0 + n_probe] / 1e3,             
                      'h': h_probe_all[:nrun], 'u': u_probe_all[:nrun],                
                      'v': v_probe_all[:nrun] }                                        
        rx['probe'] = probe                                                            
        print(f"Mean r_x over run is h = {rx['h_mean'].mean():.3f}"
                f"u={rx['u_mean'].mean():.3f}  v={rx['v_mean'].mean():.3f}")
            
        return u, v, h, time, diffs, rx

    def run_obc_nn(self, dt, samp, nsampes, prev_cols,
                   h_bc_all, u_bc_all, v_bc_all, b_obc,
                   h_ext_all, u_ext_all, v_ext_all,
                   phase_fn=None, nudging=True, alpha_in=1,
                   nn_field_codes=(0,),   # [MF] field codes that receive the NN; (0,)=h-only (back-compat), (0,1,2)=all fields
                   nn_inflow_corr=False,   # [INCORR] if True, phase_fn returns (rx,ry,dphi_in) and dphi_in is added to the inflow nudge (config 3.2)
                   nn_outflow_corr=False,   # [ANCH] if True (anchored_corr), phase_fn returns (rx,ry,dphi_out); dphi_out is added to the OUTFLOW radiate
                   inflow_mode=0,   # [IM] 1 -> radiation-inflow scheme on inflow rows (reads col-(b+1) exterior @ n+1); 0 -> unchanged nudge/prescribe
                   h_ext_out_all=None, u_ext_out_all=None, v_ext_out_all=None,   # [IM] per-step col-(b+1) exterior data @ n+1; REQUIRED when inflow_mode=1
                   r_max=None,   # [RMAX] outflow (and inflow_mode=1) CFL clamp on |rx|,|ry|; None -> unclamped (historical default, unchanged)
                   alpha_from_rx=False,   # [ALPHARX] True -> mode-0 inflow nudge coeff = min(|rx|,1) per row (train==deploy with the [ROLL] rollout law); False -> fixed alpha_in (unchanged)
                   obc_order=1,   # [O2] MUST be 1: this driver's packed 8-pt stencil (and every phase_fn trained on it) has no b-3 column and no n-1 time level, so the 2nd-order estimator is DELIBERATELY UNSUPPORTED here (hard raise, no silent 1st-order fallback). Use run_obc for obc_order=2.
                   eps_denom=None,  # [EPSREL] (3,nk) per-(field,layer) denominator floor, rows = h,u,v. None -> uniform 1e-3 (historical). Used by BOTH the in-kernel estimator and this driver's _phase_inflow_np analytic fallback, so the NN run and the analytic run stay comparable.
                   store_center=False, center_j=None, center_stride=1,   # [CL]
                   center_fields=('h', 'u', 'v'),                        # [CL]
                   center_i0=0, center_i1=None):                         # [CL] None -> b_obc+1 (interior of the OBC)
        """
            samp : steps between stored output frames
            phi_bc_all : per_step stored Western prescribed value
            phi_ext_all : external forcing at open boundary
            phase_fn : The Neural Network

            [CL] store_center=True keeps row `center_j` (default nj//2) at EVERY step
            over columns [center_i0, center_i1); center_i1=None -> b_obc+1, so the
            exterior copy at b+1 is excluded. Result on `self.center_line`.

            [RMAX] r_max=None (default) applies NO CFL clamp, as before -- the phase speed
            comes either from _phase_inflow_np (unclamped by design, see its docstring) or
            straight from the network. Pass r_max=<value> (e.g. run_obc's R_MAX_DEFAULT, or
            whatever value the analytic baseline this run is compared against used) to clamp
            |rx|,|ry| the same way _orlanski_east does, on the outflow branch and on the
            inflow-radiation branch (inflow_mode=1); the plain nudge/prescribe inflow branch
            (inflow_mode=0) does not use rx/ry and is unaffected either way.

            [FRESH]/[NNRX] Timing (matches run_obc / the analytic scheme exactly):
            each step, the continuity update of columns b-1, b-2 is replicated in
            numpy BEFORE the kernel (exact -- those columns depend only on time-n
            data), the NN / analytic estimator predicts THIS step's h (rx, ry)
            from the training-identical stencil, and the kernel (obc_on=1) applies
            the h-OBC between continuity and the momentum solve with those values.
            The momentum therefore reads a correctly radiated boundary column:
            the exterior cannot influence the interior, with zero phase-speed lag
            for h.  u and v phase speeds are computed post-step from the
            post-solve stencil (also lag-free for the applied boundary value; the
            in-kernel u/v OBC application is dynamically inert because nothing
            reads those columns into the interior before the post-step overwrite).
        """
        prev_cols = np.ascontiguousarray(np.asarray(prev_cols, dtype=np.int64).ravel())
        b = int(b_obc)
        nn_field_codes = tuple(int(c) for c in np.atleast_1d(nn_field_codes))   
        alpha_eff = float(alpha_in) if nudging else 1.0

        # Total number of step                 
        nsteps = nsampes * samp
        # Check that enough boundary and external data are available. 
        if h_bc_all.shape[0] < nsteps:
            raise ValueError(f" stored west data has {h_bc_all.shape[0]} < nsteps={nsteps}")

        # Prepare the external boundary data, make sure the external data have the expected shape (nsteps, nk, nj). 
        def _prep_ext(a, name):
            a = np.asarray(a, dtype=np.float64)
            if a.ndim == 4 and a.shape[-1] == 1:
                a = a[..., 0]
            if a.ndim != 3 or a.shape[1:] != (self.nk, self.nj):
                raise ValueError(f" {name} must be (>=nsteps,nk,nj[,1]); got {a.shape}")
            if a.shape[0] < nsteps:
                raise ValueError(f" {name} has {a.shape[0]} < nsteps={nsteps}")
            return np.ascontiguousarray(a)
        h_ext_all = _prep_ext(h_ext_all, "h_ext_all")
        u_ext_all = _prep_ext(u_ext_all, "u_ext_all")
        v_ext_all = _prep_ext(v_ext_all, "v_ext_all")

        inflow_mode = int(inflow_mode)                                                 
        if inflow_mode == 1:                                                           
            if h_ext_out_all is None or u_ext_out_all is None or v_ext_out_all is None:  
                raise ValueError("[IM] inflow_mode=1 needs h_ext_out_all/u_ext_out_all/"  
                                 "v_ext_out_all (col-(b+1) exterior data @ n+1)")      
            h_ext_out_all = _prep_ext(h_ext_out_all, "h_ext_out_all")                  
            u_ext_out_all = _prep_ext(u_ext_out_all, "u_ext_out_all")                  
            v_ext_out_all = _prep_ext(v_ext_out_all, "v_ext_out_all")                  

        if not (2 <= b < self.ni - 1):
            raise ValueError(f" b_obc={b} needs 2 <= b < ni-1={self.ni-1}")
        if np.any((prev_cols >= b - 2) & (prev_cols <= b)):
            raise ValueError(f" prescribed cols must avoid {{{b-2},{b-1},{b}}}; got {list(prev_cols)}")
        eps_denom = self._check_eps_denom(eps_denom)                                 # [EPSREL]
        # [EPSREL] the driver's numpy stencils are (nk,nj,8) flattened layer-major, so the
        # [EPSREL] per-layer floor has to be repeated nj times to align row-for-row with Xf.
        _eps_flat = [np.repeat(eps_denom[_c], self.nj) for _c in range(3)]           # [EPSREL] each (nk*nj,)
        if int(obc_order) != 1:                                                      # [O2]
            raise NotImplementedError(                                               # [O2]
                '[O2] run_obc_nn supports obc_order=1 only: the packed 8-point stencil '  # [O2]
                '(indices 0-7) and every trained phase_fn lack column b-3 and time level '  # [O2]
                'n-1.  Use run_obc for the 2nd-order estimator.')                    # [O2]

        self._print_run_info(dt, nsteps)
        print(f" east phase speed from "
              f"{'analytic numpy (classical baseline)' if phase_fn is None else 'NEURAL NETWORK'}; "
              f"inflow {('NUDGE alpha_in=%.3g' % alpha_eff) if nudging else 'PRESCRIBE (hard)'}")
        if inflow_mode == 1:                                                           # [IM]
            print(f"[IM] inflow_mode=1: RADIATION-INFLOW scheme on inflow rows "       # [IM]
                  f"(reads col b+1 exterior @ n+1, UNCLAMPED like this whole driver; "  # [IM]
                  f"extra col-b nudge {'ON, alpha_in=%.3g' % alpha_eff if nudging else 'OFF'})")  # [IM]
        print(f"Running (Neural-Network-OBC, "
              f"{'doubly periodic' if self.y_periodic else 'non-periodic N/S walls'})...")   # [YP]

        # Initialize the simulation output
        u = np.zeros((nsampes+1, self.nk, self.nj, self.ni))
        v = np.zeros((nsampes+1, self.nk, self.nj, self.ni))
        h = np.zeros((nsampes+1, self.nk, self.nj, self.ni))
        time = np.zeros((nsampes+1))

        # Initial state is copied from the model's current state. 
        u[0] = self.u; v[0] = self.v; h[0] = self.h; time[0] = self.time

        # centre-line setup; default cut = up to and including the OBC column b
        _c_on, _c_j, _c_st, _c_f, _c_sl = self._center_init(
            store_center, center_j, center_stride, center_fields, nsteps,
            center_i0, center_i1, i1_default=b + 1)

        h_diff = np.zeros(2); u_diff = np.zeros(2); v_diff = np.zeros(2)   # west misfit scratch

        if self.y_periodic:                                                          # [YP]
            def jm(A): return np.roll(A, 1, axis=-1)    # (.., j-1) wrapped           # [YP]
            def jp(A): return np.roll(A, -1, axis=-1)   # (.., j+1) wrapped           # [YP]
        else:
            def jm(A):   # (.., j-1); at j=0 mirror the wall row
                out = np.empty_like(A); out[..., 1:] = A[..., :-1]; out[..., 0] = A[..., 0]; return out
            def jp(A):   # (.., j+1); at j=nj-1 mirror the wall row
                out = np.empty_like(A); out[..., :-1] = A[..., 1:]; out[..., -1] = A[..., -1]; return out
        # build (nk,nj,8) packed stencil
        def _X8(bm1_n, b_n, bm1_np1, bm2_np1):                          
            return np.stack([jm(bm1_n), bm1_n, jp(bm1_n),     # 0,1,2 (b-1, j-1/j/j+1) @ n
                             jm(b_n),   b_n,   jp(b_n),       # 3,4,5 (b,   j-1/j/j+1) @ n
                             bm1_np1, bm2_np1], axis=-1)      # 6,7   pim1, pim2 @ n+1

        # record the APPLIED (rx,ry) per field/row/step so the NN run has a phase-speed
        _rxy_rows = [np.zeros((nsteps, self.nk, self.nj, 2)) for _ in range(3)]

        _saved_rx = [None, None, None]  # [-, u, v] -> (nk,nj) or None
        _saved_ry = [None, None, None]
        # pre-allocated (nk,nj) arrays for the _step_core call (reused each step)
        _hrx = np.zeros((self.nk, self.nj)); _hry = np.zeros((self.nk, self.nj))
        _urx = np.zeros((self.nk, self.nj)); _ury = np.zeros((self.nk, self.nj))
        _vrx = np.zeros((self.nk, self.nj)); _vry = np.zeros((self.nk, self.nj))

        # [NNRX] per-step scratch: h_prev at b-1:b (time n), rx diag, rxy diag
        # [O2] widened 2 -> 3 to match the kernel's new signature; slot 2 written by the
        # [O2] refresh below but NEVER read here: run_obc_nn always runs the kernel at
        # [O2] obc_order=1 (the packed 8-pt stencil this driver and its phase_fn use has
        # [O2] no b-3 column and no n-1 level -- see the guard in the signature block).
        _h_prev = np.zeros((self.nk, self.nj, 3))   # [O2]
        _u_prev = np.zeros((self.nk, self.nj, 3))   # [O2]
        _v_prev = np.zeros((self.nk, self.nj, 3))   # [O2]
        _rx_h = np.zeros(2); _rx_u = np.zeros(2); _rx_v = np.zeros(2)
        _rxy_h = np.zeros((self.nk, self.nj, 3))    # [O2] slot 2 = diag flag bitmask
        _rxy_u = np.zeros((self.nk, self.nj, 3))    # [O2]
        _rxy_v = np.zeros((self.nk, self.nj, 3))    # [O2]
        # [NNRX] r_max for the kernel: r_max=None historically means UNCLAMPED, so pass
        # [NNRX] a huge bound that never binds (the kernel always clamps at its r_max arg)
        _r_max_val = float(r_max) if r_max is not None else 1e30
        _h_corr_mode = (phase_fn is not None and 0 in nn_field_codes
                        and (nn_inflow_corr or nn_outflow_corr))

        nsamp = 0; nrun = nsteps
        # Main timestep loop. 
        for it in range(1, nsteps + 1):
            # snapshot phi^n at columns b-1, b (all rows j) BEFORE the step
            h_bm1, h_b = self.h[:, :, b-1].copy(), self.h[:, :, b].copy()
            u_bm1, u_b = self.u[:, :, b-1].copy(), self.u[:, :, b].copy()
            v_bm1, v_b = self.v[:, :, b-1].copy(), self.v[:, :, b].copy()
            _h_prev[:, :, 0:2] = self.h[:, :, b-1:b+1]   # [O2] slice into the 3-wide buffer (slot 2 unused here, kernel runs at order 1)
            _u_prev[:, :, 0:2] = self.u[:, :, b-1:b+1]   # [O2]
            _v_prev[:, :, 0:2] = self.v[:, :, b-1:b+1]   # [O2]

            # Mimic the exact continuity updata law for 'h' field, producing 'h' field at n+1, so the Neural network then use the same stencil to predict the phase speed.
            _h_np1 = _continuity_np(self.h, self.u, self.v, dt, self.dx, self.dy,
                                    self.iter, self.y_periodic,
                                    float(self.h_interface_relax),
                                    self._h_interface_target, self.D)
            # 8 point stencil for 'h' field. 
            X8h = _X8(h_bm1, h_b, _h_np1[:, :, b-1].copy(), _h_np1[:, :, b-2].copy())
            Xfh = X8h.reshape(-1, 8)
            # calculate the classicle phase speed for 'h' field. 
            rx_hf, ry_hf, _ = _phase_inflow_np(Xfh, _eps_flat[0])          # [EPSREL] h floor
            if phase_fn is not None and 0 in nn_field_codes:
                if nn_inflow_corr:                              # [INCORR] rx,ry part only here
                    rx_hf, ry_hf, _ = phase_fn(Xfh, 0)
                elif nn_outflow_corr:                           # [ANCH] rx,ry part only here
                    rx_hf, ry_hf, _ = phase_fn(Xfh, 0)
                else:
                    rx_hf, ry_hf = phase_fn(Xfh, 0)
            if r_max is not None: 
                _inf_hf = np.asarray(rx_hf) < 0.0
                # Clipping for stability
                if inflow_mode == 1:
                    rx_hf = np.where(_inf_hf, np.maximum(rx_hf, -r_max), np.minimum(rx_hf, r_max))
                else:
                    rx_hf = np.where(_inf_hf, rx_hf, np.minimum(rx_hf, r_max))
                ry_hf = np.clip(ry_hf, -r_max, r_max)

            # The predictions get reshaped back, now the NN predicts phase velocity for 'h'
            np.copyto(_hrx, np.asarray(rx_hf, dtype=np.float64).reshape(self.nk, self.nj))
            np.copyto(_hry, np.asarray(ry_hf, dtype=np.float64).reshape(self.nk, self.nj))
            if not _h_corr_mode:
                _rxy_rows[0][it-1, :, :, 0] = _hrx
                _rxy_rows[0][it-1, :, :, 1] = _hry

            if _saved_rx[1] is not None:
                np.copyto(_urx, _saved_rx[1]); np.copyto(_ury, _saved_ry[1])
                np.copyto(_vrx, _saved_rx[2]); np.copyto(_vry, _saved_ry[2])

            h_ext_s  = h_ext_all[it-1];   u_ext_s  = u_ext_all[it-1];   v_ext_s  = v_ext_all[it-1]
            if inflow_mode == 1:
                h_ext_out_s = h_ext_out_all[it-1]; u_ext_out_s = u_ext_out_all[it-1]
                v_ext_out_s = v_ext_out_all[it-1]
            else:
                h_ext_out_s = u_ext_out_s = v_ext_out_s = self._obc_ext_dummy
            self._step_core(dt, 2, prev_cols,
                            h_bc_all[it-1], u_bc_all[it-1], v_bc_all[it-1],
                            h_diff, u_diff, v_diff,
                            obc_on=1, b_obc=b,
                            h_prev=_h_prev, u_prev=_u_prev, v_prev=_v_prev,
                            rx_h=_rx_h, rx_u=_rx_u, rx_v=_rx_v,
                            rxy_h=_rxy_h, rxy_u=_rxy_u, rxy_v=_rxy_v,
                            nudging_mode=int(bool(nudging)),
                            h_ext=h_ext_s, u_ext=u_ext_s, v_ext=v_ext_s,
                            alpha_in=alpha_eff,
                            h_zerograd=0, inflow_persistent=0, ry_on=1,
                            r_max=_r_max_val, inflow_mode=inflow_mode,
                            h_ext_out=h_ext_out_s, u_ext_out=u_ext_out_s,
                            v_ext_out=v_ext_out_s,
                            # [NNRX] h: THIS step's fresh (rx,ry) ([FRESH] above);
                            # [NNRX] u,v: previous step's post-step estimate (dynamically inert)
                            use_nn_rx=1,
                            h_rx_pre=_hrx, h_ry_pre=_hry,
                            u_rx_pre=_urx, u_ry_pre=_ury,
                            v_rx_pre=_vrx, v_ry_pre=_vry,
                            alpha_from_rx=int(bool(alpha_from_rx)),   # [ALPHARX]
                            eps_denom=eps_denom)   # [EPSREL] kernel path (use_nn_rx=1 rows ignore it; the analytic-fallback rows do not)

            # Post-step stencils for `u` and `v`. Construct their stencils using the already updated state : 
            _post_fields = [(self.u, u_bm1, u_b, u_ext_all, u_ext_out_all, 1),
                            (self.v, v_bm1, v_b, v_ext_all, v_ext_out_all, 2)]
            if _h_corr_mode:
                _post_fields.append((self.h, h_bm1, h_b, h_ext_all, h_ext_out_all, 0))
            for fld, bm1_n, b_n, ext_all, ext_out_all, fcode in _post_fields:
                X8 = _X8(bm1_n, b_n, fld[:, :, b-1].copy(), fld[:, :, b-2].copy())   # (nk,nj,8)
                Xf = X8.reshape(-1, 8)                       # (m,8) flat batch of stencils
                # Then compute the analytic estimate. 
                rx, ry, inflow = _phase_inflow_np(Xf, _eps_flat[fcode])    # [EPSREL] per-field floor

                dphi_in = None 
                dphi_out = None 
                if phase_fn is not None and fcode in nn_field_codes:
                    if nn_inflow_corr:                                     
                        rx, ry, dphi_in = phase_fn(Xf, fcode)   
                    elif nn_outflow_corr:  
                        rx, ry, dphi_out = phase_fn(Xf, fcode) 
                    else:
                        rx, ry = phase_fn(Xf, fcode)
                    inflow = rx < 0.0
                if r_max is not None:   
                    if inflow_mode == 1:
                        rx = np.where(inflow, np.maximum(rx, -r_max), np.minimum(rx, r_max))
                    else:
                        rx = np.where(inflow, rx, np.minimum(rx, r_max))
                    ry = np.clip(ry, -r_max, r_max)
                _rxy_rows[fcode][it-1, :, :, 0] = np.asarray(rx).reshape(self.nk, self.nj)
                _rxy_rows[fcode][it-1, :, :, 1] = np.asarray(ry).reshape(self.nk, self.nj) 

                _saved_rx[fcode] = np.asarray(rx, dtype=np.float64).reshape(self.nk, self.nj).copy()
                _saved_ry[fcode] = np.asarray(ry, dtype=np.float64).reshape(self.nk, self.nj).copy()

                # Direct post-step overwirte for `u` and `v`. 
                upd = _obc_update_np(Xf, rx, ry, inflow, ext_all[it-1].reshape(-1), alpha_eff,
                                     dphi_in=dphi_in, dphi_out=dphi_out,   # [INCORR]  # [ANCH]
                                     inflow_mode=inflow_mode,              # [IM]
                                     phi_ext_out=(ext_out_all[it-1].reshape(-1) if inflow_mode == 1 else None),  # [IM]
                                     inflow_nudge=bool(nudging),           # [IM]
                                     alpha_from_rx=bool(alpha_from_rx))    # [ALPHARX]
                # overwrite col b with the fresh update
                fld[:, :, b] = upd.reshape(self.nk, self.nj)
                # Also overwrite col b+1
                fld[:, :, b+1] = fld[:, :, b].copy()
                if b + 2 < self.ni:                                                  # [TRUNC]
                    fld[:, :, b+2] = fld[:, :, b].copy()                                    # [TRUNC]

            if not self.y_periodic:                                                  # [YP]
                self.v[:, 0, :] = 0.0   # [NP] south wall: the col-b overwrite can set v at j=0; re-zero it
            if np.any(np.isnan(self.u)):
                print('Model has blown up!!! Stopping early')
                u = u[:nsamp]; v = v[:nsamp]; h = h[:nsamp]; time = time[:nsamp]
                self._center_trim(_c_on, it - 1, _c_st)      # [CL] drop blown step
                nrun = it; break
            self._center_store(_c_on, it, _c_j, _c_st, _c_f, _c_sl)   # [CL] post-step centre row
            if it % samp == 0:
                nsamp += 1
                u[nsamp] = self.u; v[nsamp] = self.v
                h[nsamp] = self.h; time[nsamp] = self.time
        print("...done")

        diffs = {'t_step': np.arange(1, nrun+1) * dt, 'h_max': None,
                 # [NNRXY] per-row APPLIED (rx,ry) for the NN run, mirroring run_obc's rx dict.
                 # convert to phase speed: c_x = rx*dx/dt, c_y = ry*dy/dt.  Each (nrun,nk,nj).
                 'rx_h_row': _rxy_rows[0][:nrun, :, :, 0], 'ry_h_row': _rxy_rows[0][:nrun, :, :, 1],
                 'rx_u_row': _rxy_rows[1][:nrun, :, :, 0], 'ry_u_row': _rxy_rows[1][:nrun, :, :, 1],
                 'rx_v_row': _rxy_rows[2][:nrun, :, :, 0], 'ry_v_row': _rxy_rows[2][:nrun, :, :, 1]}
        return u, v, h, time, diffs                          # 5-tuple: (u, v, h, time, diffs)  [NNRXY] +rxy rows in diffs

    def _step_core(self, dt, bc_mode, bc_cols,\
                   # [OBC] setting
                   h_bc, u_bc, v_bc, h_diff, u_diff, v_diff,
                   # [OBC - Orlanski] Settings
                   obc_on = 0, b_obc = 0,
                   h_prev = None, u_prev = None, v_prev = None,
                   rx_h = None, rx_u = None, rx_v = None,
                   rxy_h = None, rxy_u = None, rxy_v = None,  # [OBC - 2D - PhaseDiag] per-row (nk,nj,2) applied (rx,ry) out
                   nudging_mode = 0, h_ext = None, u_ext = None, v_ext = None, alpha_in = 0.0,  # [OBC - Orlanski - 2D - Nuding]; [OBC - Orlanski - 2D - IndepRx] dropped rx_field/uv_from_h
                   h_zerograd = 0,   # 1 -> zero-gradient h at the OBC instead of radiation
                   inflow_persistent = 0,  # [PERSIST] 1 -> clamp rx>=0 (inflow becomes persistence, no nudge)
                   ry_on = 1,  # [RY1D] 1 -> 2D Orlanski; 0 -> 1D Orlanski (ry == 0)
                   r_max = R_MAX_DEFAULT,  # [RMAX] outflow CFL clamp; default preserves this module's historical 1/50
                   inflow_mode = 0,  # [IM] 1 -> radiation-inflow scheme on the east OBC (reads *_ext_out); 0 -> unchanged
                   h_ext_out = None, u_ext_out = None, v_ext_out = None,  # [IM] per-field col-(b+1) exterior data @ n+1, (nk,nj); only read when inflow_mode == 1
                   # [NNRX] pre-computed per-row (rx,ry) for h,u,v (None -> use analytic estimation).
                   # Non-None arrays are fed through to _orlanski_east via _step_numba.
                   use_nn_rx = 0,
                   h_rx_pre = None, h_ry_pre = None,
                   u_rx_pre = None, u_ry_pre = None,
                   v_rx_pre = None, v_ry_pre = None,
                   alpha_from_rx = 0,                                  # [ALPHARX] 1 -> inflow nudge coeff = min(|rx|,1); 0 (default) -> fixed alpha_in (unchanged)
                   obc_order = 1,                                      # [O2] 1 (default, bit-identical) | 2 = 2nd-order D_t/D_x estimator in _orlanski_east
                   eps_denom = None):                                  # [EPSREL] (3,nk) per-(field,layer) denominator floor, rows = h,u,v. None -> the historical uniform 1e-3.
                 
        # Sponge gates are cached by the h/u/v_relax setters (a bool keeps the
        # JIT argument types stable and lets numba skip a term when its sponge
        # is off), so no per-step recomputation is needed.

        if h_prev is None : h_prev = self._obc_prev_dummy
        if u_prev is None : u_prev = self._obc_prev_dummy
        if v_prev is None : v_prev = self._obc_prev_dummy
        if rx_h is None : rx_h = self._diff_dummy
        if rx_u is None : rx_u = self._diff_dummy
        if rx_v is None : rx_v = self._diff_dummy
        if rxy_h is None : rxy_h = self._obc_rxy_dummy
        if rxy_u is None : rxy_u = self._obc_rxy_dummy
        if rxy_v is None : rxy_v = self._obc_rxy_dummy
        # external col-b data (read on inflow; nudging_mode==1 soft, ==0 hard-prescribe)
        if h_ext is None : h_ext = self._obc_ext_dummy
        if u_ext is None : u_ext = self._obc_ext_dummy
        if v_ext is None : v_ext = self._obc_ext_dummy
        # [IM] exterior col-(b+1) data @ n+1 (read on inflow only when inflow_mode == 1;
        # [IM] the dummy keeps the numba argument types stable when the knob is off)
        if h_ext_out is None : h_ext_out = self._obc_ext_dummy                   # [IM]
        if u_ext_out is None : u_ext_out = self._obc_ext_dummy                   # [IM]
        if v_ext_out is None : v_ext_out = self._obc_ext_dummy                   # [IM]
        # [NNRX] substitute dummies when not provided so numba sees consistent (nk,nj) array types
        # [EPSREL] (3,nk) float64, C-contiguous so each row is a contiguous (nk,) view for numba
        if eps_denom is None : eps_denom = self._obc_eps_default          # [EPSREL] uniform 1e-3
        if h_rx_pre is None : h_rx_pre = self._nn_rx_dummy
        if h_ry_pre is None : h_ry_pre = self._nn_ry_dummy
        if u_rx_pre is None : u_rx_pre = self._nn_rx_dummy
        if u_ry_pre is None : u_ry_pre = self._nn_ry_dummy
        if v_rx_pre is None : v_rx_pre = self._nn_rx_dummy
        if v_ry_pre is None : v_ry_pre = self._nn_ry_dummy

        _step_numba(self.u, self.v, self.h, self.D, self.taux, self.tauy,
                    self.f, self.f_at_u, self.f_at_v,
                    dt, self.dx, self.dy, self.g, self.epsilon, self.nu_h, self.nu_v,
                    self.alpha_f, self.alpha_nu,
                    self.h_zonal_target, self.h_target, self.u_target, self.v_target,
                    self.h_zonal_relax, self._h_relax, self._u_relax, self._v_relax,
                    self._h_relax_on, self._u_relax_on, self._v_relax_on,
                    self.hsub, self.iter,
                    float(self.h_interface_relax), self._h_interface_target,  # [INTERFACE_RELAX]
                    bc_mode, bc_cols, h_bc, u_bc, v_bc, h_diff, u_diff, v_diff, #  [OBC]
                    obc_on, int(b_obc), h_prev, u_prev, v_prev, rx_h, rx_u, rx_v,  # [OBC - Orlanski]
                    rxy_h, rxy_u, rxy_v,  # [OBC - 2D - PhaseDiag] per-row (rx,ry) recording arrays
                    int(nudging_mode), h_ext, u_ext, v_ext, float(alpha_in),  # [OBC - Orlanski - 2D - Nuding]
                    int(h_zerograd),
                    int(inflow_persistent),  # [PERSIST]
                    int(ry_on),  # [RY1D]
                    self.y_periodic,  # [YP]
                    float(r_max),  # [RMAX]
                    int(inflow_mode), h_ext_out, u_ext_out, v_ext_out,  # [IM]
                    # [NNRX] pre-computed rx,ry for h,u,v (or dummies when use_nn_rx==0)
                    int(use_nn_rx),
                    h_rx_pre, h_ry_pre,
                    u_rx_pre, u_ry_pre,
                    v_rx_pre, v_ry_pre,  # [NNRX]
                    int(alpha_from_rx),  # [ALPHARX]
                    int(obc_order),      # [O2]
                    eps_denom[0], eps_denom[1], eps_denom[2])   # [EPSREL] h, u, v rows
        self.time += dt
        self.iter += 1

    def step(self, dt):
        """
        dt   - Time step [s]
        """
        self._step_core(dt, 0, self._bc_cols_dummy,
                        self._bc_dummy, self._bc_dummy, self._bc_dummy,
                        self._diff_dummy, self._diff_dummy, self._diff_dummy)

    def _check_3d(self, a, name):
        """Validate a has shape (nk, nj, ni)."""
        if a.shape != (self.nk, self.nj, self.ni):
            raise ValueError(f"{name} must have shape ({self.nk}, {self.nj}, {self.ni}), "
                             f"got {a.shape}")

    def abs_omega(self, u=None, v=None, k=None):
        """Per-layer absolute vorticity, f + vx - uy [s-1]. Returns (nk, nj, ni)
        by default; pass an integer k to return a 2D slice."""
        if u is None: u = self.u
        if v is None: v = self.v
        self._check_3d(u, 'u')
        self._check_3d(v, 'v')
        vx, uy = _nb_vxuy(u, v, 1 / self.dx, 1 / self.dy)
        omega = self.f + ( vx - uy )
        if k is None:
            return omega
        return omega[k]

    def q(self, h=None, u=None, v=None, k=None):
        """Per-layer potential vorticity, q = (f + vx - uy) / h [s-1 m-1].
        Returns (nk, nj, ni) by default; pass an integer k to return a 2D slice."""
        if u is None: u = self.u
        if v is None: v = self.v
        if h is None: h = self.h
        self._check_3d(h, 'h')
        self._check_3d(u, 'u')
        self._check_3d(v, 'v')
        hq = _nb_u2q( _nb_h2u( h ) )
        recip_hq_plus_hsub = 1.0 / ( hq + self.hsub )
        vx, uy = _nb_vxuy(u, v, 1 / self.dx, 1 / self.dy)
        q = self.f + ( vx - uy )
        q *= recip_hq_plus_hsub
        q *= ( hq * recip_hq_plus_hsub ) # Hack to mask q
        if k is None:
            return q
        return q[k]

    def KE(self, h=None, u=None, v=None, k=None):
        """Per-layer kinetic energy , KE = h /2 ( u^2 + v^2 )  [m3 s-2].
        Returns (nk, nj, ni) by default; pass an integer k to return a 2D slice."""
        if u is None: u = self.u
        if v is None: v = self.v
        if h is None: h = self.h
        u2 = _nb_q2v( u**2 )
        v2 = _nb_q2u( v**2 )
        KE = 0.5 * h * ( u2 + v2 )
        if k is None:
            return KE
        return KE[k]
