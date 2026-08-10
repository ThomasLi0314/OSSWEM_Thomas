import numpy as np
from numba import njit, prange, get_num_threads


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


# Default phase velocity clipping
R_MAX_DEFAULT = 1.0 / 50.0

@njit(cache=True)
def _orlanski_east(phi, phi_prev, b_obc, rx_out,
                   nudging_mode, phi_ext, alpha_in,
                   rxy_out,                   
                   inflow_persistent,         
                   ry_on,                    
                   y_periodic,               
                   r_max,                    
                   inflow_mode, phi_ext_out, 
                   # Directly use exterior phase velocity when the Neural network is applied to predict it. 
                   use_nn_rx,                
                   rx_nn, ry_nn):             
    """East Orlanski radiation at column b_obc, applied per layer k and row j.

    The phase-speed pair (rx, ry) (as Courant numbers) is estimated from the two
    interior columns b-1, b-2:
        rx = -dphi_t*dphi_x / (dphi_x^2 + dphi_y^2),  ry = -dphi_t*dphi_y / (same)
    with the tangential difference dphi_y upwinded along the boundary. Then

      outflow (rx >= 0): radiate,
          phi_b^{n+1} = (phi_b^n + rx*phi_{b-1}^{n+1} - ry*dy_b) / (1 + rx),
          with rx clamped to r_max and |ry| to r_max.
      inflow (rx < 0): treated according to inflow_mode (see below).

    ry_on = 0 drops the tangential difference from BOTH the denominator and the
    update, collapsing the scheme to the classic 1D Orlanski (ry == 0).

    inflow_mode :  inflow-branch selector.
        0 -> PRESCRIBE / NUDGE toward the recorded external solution phi_ext:
             nudging_mode = 0 -> phi_b^{n+1} = phi_ext;
             nudging_mode = 1 -> phi_b^{n+1} = phi_b^n + alpha_in*(phi_ext - phi_b^n)
             (alpha_in = 1 == hard prescribe).
        1 -> radiation-INFLOW scheme: implicit upwind of the radiation equation
             with the upstream point at column b+1,
               (phi_b^{n+1}-phi_b^n) + rx*(phi_ext_out - phi_b^{n+1}) + ry*dy_b = 0
               ->  phi_b^{n+1} = (phi_b^n - rx*phi_ext_out - ry*dy_b) / (1 - rx),  rx < 0.
             A convex blend of persistence and phi_ext_out (weights 1/(1+|rx|),
             |rx|/(1+|rx|)): bounded, no CFL restriction, and continuous with the
             outflow branch at rx -> 0.  |rx| and |ry| are clamped at r_max
             (symmetric with the outflow clamp).  With nudging_mode == 1 the result
             is additionally nudged toward phi_ext with alpha_in.
        2 -> persistence: phi_b^{n+1} = phi_b^n.
    phi_ext_out : (nk, nj) external value at column b_obc + 1 at time n+1; only read
        when inflow_mode == 1.  REQUIRED as an array -- numba cannot type a None
        default against phi_ext_out[k,j], so pass a dummy (nk,nj) array when unused.
    inflow_persistent : [PERSIST] 1 -> clamp rx (and ry) to 0 whenever the raw rx is
        negative, so inflow rows persist instead of reading the external data.
    use_nn_rx : 1 -> read (rx, ry) from rx_nn/ry_nn (e.g. a neural network)
        instead of estimating them analytically; inflow_persistent still applies.
        Pass dummy (nk,nj) arrays when 0.

    Outflow is ALWAYS pure radiation in every mode (no nudge applied).

    The ghost columns b_obc+1 and b_obc+2 are set equal to b_obc in every branch
    (b+2 guarded, so b = ni-2 stays legal on a truncated grid): everything east of
    b_obc is meaningless and must be cropped from diagnostics.

    phi_prev holds phi^n at columns [b-1, b] (shape (nk, nj, 2)); the interior of
    phi is already at n+1 when this runs. rx_out returns (mean, max) of rx over
    (k, j); rxy_out records the per-row applied (rx, ry) (raw, signed on inflow).
    """
    nk, nj, ni = phi.shape
    maxr = 0.0;    sumr = 0.0

    for k in range(nk):
        for j in range(nj):
            # [YP] tangential neighbours: wrap when y-periodic, mirror ghost at the walls
            jm = j - 1 if j > 0 else ( nj - 1 if y_periodic else 0 )
            jp = j + 1 if j < nj - 1 else ( 0 if y_periodic else nj - 1 )
            pim1 = phi[k, j, b_obc-1]                    # phi^{n+1}_{b-1, j}
            pim2 = phi[k, j, b_obc-2]                    # phi^{n+1}_{b-2, j}

            dphi_t = pim1 - phi_prev[k, j, 0]            # - phi^n_{b-1, j}
            dphi_x = pim1 - pim2

            # upwinded tangential difference at b-1 (selector = central difference)
            cen = phi_prev[k, jp, 0] - phi_prev[k, jm, 0]
            if dphi_t * cen > 0.0:
                dphi_y = phi_prev[k, j, 0] - phi_prev[k, jm, 0]
            else:
                dphi_y = phi_prev[k, jp, 0] - phi_prev[k, j, 0]

            # 1D orlanski loop
            if ry_on == 1:
                denom = dphi_x*dphi_x + dphi_y*dphi_y
            else:
                denom = dphi_x*dphi_x
            # when available, skipping the pointwise analytic estimation entirely.
            if use_nn_rx == 0:
                eps_denom = 1e-3
                r_d = (1.0 / eps_denom) if denom < eps_denom else (1.0 / denom)
                rx = -dphi_t * dphi_x * r_d
                ry = -dphi_t * dphi_y * r_d if ry_on == 1 else 0.0
            else:    
                rx = rx_nn[k, j]
                ry = ry_nn[k, j] 
            if inflow_persistent: 
                if rx < 0.0: 
                    rx = 0.0 
                    ry = 0.0 
            # Inflow
            if rx < 0.0: 
                if inflow_mode == 0: 
                    if nudging_mode == 1:
                        phi[k, j, b_obc] = phi_prev[k, j, 1] + alpha_in * (phi_ext[k, j] - phi_prev[k, j, 1])
                        phi[k, j, b_obc + 1] = phi[k, j, b_obc]
                        if b_obc + 2 < ni: phi[k, j, b_obc + 2] = phi[k, j, b_obc]
                    else:
                        phi[k, j, b_obc] = phi_ext[k, j]
                        phi[k, j, b_obc + 1] = phi[k, j, b_obc]
                        if b_obc + 2 < ni: phi[k, j, b_obc + 2] = phi[k, j, b_obc]

                if inflow_mode == 1: 
                    if rx < -r_max : rx = -r_max
                    if ry > r_max : ry = r_max  
                    elif ry < -r_max: ry = -r_max 
                    if ry >= 0.0:    
                        dy_b = phi_prev[k, j, 1] - phi_prev[k, jm, 1]
                    else:            
                        dy_b = phi_prev[k, jp, 1] - phi_prev[k, j, 1] 
                    phi_new = (phi_prev[k, j, 1] - rx * phi_ext_out[k, j] - ry * dy_b) / (1.0 - rx)
                    if nudging_mode == 1:       
                        phi_new = phi_new + alpha_in * (phi_ext[k, j] - phi_new) 
                    phi[k, j, b_obc] = phi_new  
                    phi[k, j, b_obc + 1] = phi[k, j, b_obc]
                    if b_obc + 2 < ni:                                            
                        phi[k, j, b_obc + 2] = phi[k, j, b_obc]                   
                elif inflow_mode == 2:          
                    phi[k, j, b_obc] = phi_prev[k, j, 1] 

            else:
                if rx > r_max: rx = r_max
                if ry > r_max: ry = r_max 
                elif ry < -r_max: ry = -r_max
                if ry >= 0.0:
                    dy_b = phi_prev[k, j, 1] - phi_prev[k, jm, 1]
                else:
                    dy_b = phi_prev[k, jp, 1] - phi_prev[k, j, 1] 

                # the 2D Orlanski update (1D when ry == 0)
                phi[k, j, b_obc] = (phi_prev[k, j, 1] + rx * pim1 - ry * dy_b) / (1.0 + rx)
                phi[k, j, b_obc + 1] = phi[k, j, b_obc]
                if b_obc + 2 < ni:                                                
                    phi[k, j, b_obc + 2] = phi[k, j, b_obc]                       

            sumr += rx
            if rx > maxr:
                maxr = rx
            rxy_out[k, j, 0] = rx                        # per-row applied rx
            rxy_out[k, j, 1] = ry                        # per-row applied ry
    rx_out[0] = sumr / (nk * nj)
    rx_out[1] = maxr


# ============ east-Orlanski OBC for the neural-network path ============
def _phase_inflow_np(X8):
    """Analytic per-point (rx, ry) + inflow flag from the packed stencil (numpy). Faithful copy
    of the estimation branch of _orlanski_east; inflow == (raw rx<0).
    [RMAX] Deliberately UNCLAMPED -- this builds NN training baselines, and clamping here would
    shift every previously recorded baseline loss. The deployed kernel clamps at r_max."""
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
    eps_denom = 1e-3                              # [OBC-eps]
    r_d = 1.0 / np.maximum(denom, eps_denom)      # [OBC-eps]
    rx = -dphi_t * dphi_x * r_d                   # [OBC-eps]
    ry = -dphi_t * dphi_y * r_d                   # [OBC-eps]
    inflow = rx < 0.0                             # inflow flag from RAW rx (before any clip)
    return rx, ry, inflow


def _obc_update_np(X8, cx, cy, inflow, phi_ext, alpha_in, dphi_in=None, dphi_out=None,   # [INCORR] +dphi_in  # [ANCH] +dphi_out
                   inflow_mode=0, phi_ext_out=None, inflow_nudge=True):                  # [IM] radiation-inflow scheme knob + col-(b+1) exterior @ n+1
    """East-Orlanski boundary update (numpy), faithful to _orlanski_east.
    OUTFLOW: phi_b = (phi_prev_b + cx*pim1 - cy*dy_b)/(1+cx) [+ dphi_out].
    INFLOW : phi_b = phi_prev_b + alpha_in*(phi_ext - phi_prev_b) [+ dphi_in]  (cx,cy unused; alpha_in=1 == hard prescribe).
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
        nudge = phi_prev_b_j + alpha_in * (phi_ext - phi_prev_b_j)
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
                # Interface relaxation 
                h_interface_relax, h_interface_target,        
                bc_mode, bc_cols, h_bc, u_bc, v_bc, h_diff, u_diff, v_diff,     # west-band record (1) / compare+replace (2)
                obc_on, b_obc, h_prev, u_prev, v_prev, rx_h, rx_u, rx_v,        # east Orlanski state (phi^n at [b-1,b] + (mean,max) rx out)
                rxy_h, rxy_u, rxy_v,                                            # per-row applied (rx,ry) out, (nk,nj,2) per field
                nudging_mode, h_ext, u_ext, v_ext, alpha_in,                    # inflow treatment + external col-b data
                inflow_persistent,   # [ORDER] ahead of ry_on, matching the helpers module (which has h_zerograd just before it)
                ry_on,
                y_periodic,
                r_max,
                inflow_mode, h_ext_out, u_ext_out, v_ext_out,
                use_nn_rx, 
                # Pre-computed r_x and r_y outside step_numba, totally from 
                h_rx_pre, h_ry_pre,                                            
                u_rx_pre, u_ry_pre,                                           
                v_rx_pre, v_ry_pre):
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

    y_periodic  [NP] True -> the j-direction is periodic,
    False -> free-slip solid walls at y=0 and y=Ly (a zonal channel). The wall
    treatment has two distinct parts, because v lives on the *southern* face of
    each h-cell: (a) the h/u/f family uses a mirrored ghost row (jm/jp clamped
    rather than wrapped), giving zero normal gradient across the wall; (b) the
    v family (v, hv, vdot) is *zeroed* on the north face via the mask wn -- the
    north wall is the phantom v-row j=nj, which the periodic code aliases to
    row 0. """
    nk, nj, ni = u.shape
    rdx = 1 / dx
    rdy = 1 / dy

    # Pre-continuity hq at q-points (used for PV in the explicit loop); kept as
    # an array because PV is needed at several q-points. Fused u2q(h2u(h)).
    hq_pre = np.empty((nk, nj, ni))
    for k in range(nk):
        for j in prange(nj):
            jm = j - 1 if j > 0 else ( nj - 1 if y_periodic else 0 ) 
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
                    jm = j - 1 if j > 0 else ( nj - 1 if y_periodic else 0 )    # [NP] wall: hv[0] uses v[0]=0 -> zero flux through south wall
                    for i in range(ni):
                        vi = v[k,j,i]
                        vp = vi if vi > 0.0 else 0.0
                        vn = vi if vi < 0.0 else 0.0
                        hv[k,j,i] = vp * h[k,jm,i] + vn * h[k,j,i]
            for k in range(nk):
                for j in prange(nj):
                    jp = j + 1 if j < nj - 1 else ( 0 if y_periodic else nj - 1 )   # [NP]
                    wn = 1.0 if ( y_periodic or j < nj - 1 ) else 0.0               # [NP] north wall: zero meridional flux, vn = 0
                    for i in range(ni):
                        h[k,j,i] -= cy * ( wn * hv[k,jp,i] - hv[k,j,i] )             # [NP]

    # Interface relaxation, controled by whether h_interface_relax > 0 and the model is 2-layer (nk==2).
    if h_interface_relax > 0 and nk == 2:
        c_ir = dt * h_interface_relax
        for j in prange(nj):
            eta1_mean = 0.0
            for i in range(ni):
                eta1_mean += h[1, j, i] - D[j, i]
            eta1_mean /= ni
            c_inter = c_ir * (eta1_mean - h_interface_target[j])
            for i in range(ni):
                h[0, j, i] += c_inter   # (3 - 2n) = +1 for n = 1, the upper layer
                h[1, j, i] -= c_inter

    # block A: record/replace h at the prescribed boundary columns (bc_cols, any
    # length), AFTER continuity (h^{n+1}) but BEFORE the momentum step reads h, so
    # a replaced h feeds the subsequent (u,v) solve. Serial loops only (explicit
    # reductions -- keep them out of prange).
    n_bc = bc_cols.shape[0]                                                     
    if bc_mode == 1:                                                            # record (control run)
        for k in range(nk):                                                     
            for j in range(nj):                                                 
                for c in range(n_bc):                                           
                    h_bc[k,j,c] = h[k,j,bc_cols[c]]                             
    elif bc_mode == 2:                                                          # compare + replace (OBC run)
        maxd = 0.0                                                              
        ss = 0.0                                                                
        for k in range(nk):                                                     
            for j in range(nj):                                                 
                for c in range(n_bc):                                           
                    d = h[k,j,bc_cols[c]] - h_bc[k,j,c]                         
                    if abs(d) > maxd: maxd = abs(d)                             
                    ss += d*d                                                   
                    h[k,j,bc_cols[c]] = h_bc[k,j,c]                             
        h_diff[0] = maxd                                                        
        h_diff[1] = ( ss / ( nk*nj*n_bc ) )**0.5                                

    # east Orlanski for h: BEFORE the Montgomery potential below, so the boundary
    # h feeds this same step's pressure gradient.
    if obc_on == 1:                                                             
        _orlanski_east(h, h_prev, b_obc, rx_h,
                       nudging_mode, h_ext, alpha_in,
                       rxy_h,
                       inflow_persistent,                                       # [PERSIST]
                       ry_on, y_periodic,
                       r_max,                                                   # [RMAX]
                       inflow_mode, h_ext_out,                                  # 
                       use_nn_rx, h_rx_pre, h_ry_pre)                          

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
            jm = j - 1 if j > 0 else ( nj - 1 if y_periodic else 0 )            # [NP] mirror ghost row (was periodic)
            jp = j + 1 if j < nj - 1 else ( 0 if y_periodic else nj - 1 )       # [NP]
            wn = 1.0 if ( y_periodic or j < nj - 1 ) else 0.0                   # [NP] v = 0 on north wall
            for i in range(ni):
                im = i - 1 if i > 0 else ni - 1
                ip = i + 1 if i < ni - 1 else 0
                ui = u[k,j,i];   up    = ui  if ui  > 0.0 else 0.0
                uip = u[k,j,ip]; un_ip = uip if uip < 0.0 else 0.0
                vi = v[k,j,i];   vp    = vi  if vi  > 0.0 else 0.0
                vjp = wn * v[k,jp,i]; vn_jp = vjp if vjp < 0.0 else 0.0         # [NP]
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
            jm = j - 1 if j > 0 else ( nj - 1 if y_periodic else 0 )            # [NP] mirror ghost for h/u/f (was periodic)
            jp = j + 1 if j < nj - 1 else ( 0 if y_periodic else nj - 1 )       # [NP] mirror ghost; the v family uses the wn-masked locals below
            wn = 1.0 if ( y_periodic or j < nj - 1 ) else 0.0                   # [NP] north wall -> v = 0, hv = 0 across the north face
            for i in range(ni):
                im = i - 1 if i > 0 else ni - 1
                ip = i + 1 if i < ni - 1 else 0
                v_jp_i   = wn * v[k,jp,i]                                       # [NP]
                v_jp_im  = wn * v[k,jp,im]                                      # [NP]
                hv_jp_i  = wn * hv[k,jp,i]                                      # [NP]
                hv_jp_im = wn * hv[k,jp,im]                                     # [NP]

                # Masked PV at q-points (j,i), (jp,i), (j,ip):
                #   q = (f + vx - uy) ; q *= r ; q *= (hqp*r),  r = 1/(hqp+hsub)
                vort_c  = f[j, i]  + ( v[k,j,i]  - v[k,j,im]  ) * rdx - ( u[k,j,i]  - u[k,jm,i]  ) * rdy
                r_c     = 1.0 / ( hq_pre[k,j,i] + hsub )
                qpv_c   = ( vort_c * r_c ) * ( hq_pre[k,j,i] * r_c )
                vort_jp = f[jp, i] + ( v_jp_i - v_jp_im ) * rdx - ( u[k,jp,i] - u[k,j,i]   ) * rdy   # [NP] v = 0 on north wall
                r_jp    = 1.0 / ( hq_pre[k,jp,i] + hsub )
                qpv_jp  = ( vort_jp * r_jp ) * ( hq_pre[k,jp,i] * r_jp )
                vort_ip = f[j, ip] + ( v[k,j,ip] - v[k,j,i]   ) * rdx - ( u[k,j,ip] - u[k,jm,ip] ) * rdy
                r_ip    = 1.0 / ( hq_pre[k,j,ip] + hsub )
                qpv_ip  = ( vort_ip * r_ip ) * ( hq_pre[k,j,ip] * r_ip )

                # Bernoulli gradient.
                Bx = ( B[k,j,i] - B[k,j,im] ) * rdx
                By = ( B[k,j,i] - B[k,jm,i] ) * rdy

                # PV Coriolis fluxes (q2u/v2q and q2v/u2q expanded).
                qhv = 0.5 * ( qpv_c  * 0.5 * ( hv[k,j,i]  + hv[k,j,im]  )
                            + qpv_jp * 0.5 * ( hv_jp_i + hv_jp_im ) )           # [NP] zero flux on north wall
                qhu = 0.5 * ( qpv_c  * 0.5 * ( hu[k,j,i]  + hu[k,jm,i]  )
                            + qpv_ip * 0.5 * ( hu[k,j,ip] + hu[k,jm,ip] ) )

                # Stress tensor: nu_h*h*D_tension at h-points {(j,i),(j,im),(jm,i)};
                # nu_h*hq*D_shear at q-points {(j,i),(jp,i),(j,ip)}.
                Dt_c  = ( u[k,j,ip] - u[k,j,i]  ) * rdx - ( v_jp_i - v[k,j,i]  ) * rdy   # [NP] v = 0 on north wall
                nuhDt_c  = nu_h * h[k,j,i]  * Dt_c
                Dt_im = ( u[k,j,i]  - u[k,j,im] ) * rdx - ( v_jp_im - v[k,j,im] ) * rdy  # [NP]
                nuhDt_im = nu_h * h[k,j,im] * Dt_im
                Dt_jm = ( u[k,jm,ip]- u[k,jm,i] ) * rdx - ( v[k,j,i]  - v[k,jm,i] ) * rdy
                nuhDt_jm = nu_h * h[k,jm,i] * Dt_jm

                Ds_c  = ( u[k,j,i]  - u[k,jm,i] ) * rdy + ( v[k,j,i]  - v[k,j,im]  ) * rdx
                hqp_c  = min( min( h[k,j,i],  h[k,j,im]  ), min( h[k,jm,i], h[k,jm,im] ) )
                nuhqDs_c  = nu_h * hqp_c  * Ds_c
                Ds_jp = ( u[k,jp,i] - u[k,j,i]  ) * rdy + ( v_jp_i - v_jp_im ) * rdx   # [NP] v = 0 on north wall
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

    if obc_on == 1:                                                             #
        udot[:, :, b_obc] = udot[:, :, b_obc - 1]                               # boundary column inherits its inland neighbour's acceleration

    # --- implicit TDMAH2 (cancellation-free; recurrence identical to reference,
    # done per column with scalar locals) ---
    # Fused interpolation of the cross-component accelerations: vdot to
    # u-points = q2u(v2q(vdot)), udot to v-points = q2v(u2q(udot)).
    vdot_at_u = np.empty((nk, nj, ni))
    udot_at_v = np.empty((nk, nj, ni))
    for k in range(nk):
        for j in prange(nj):
            jm = j - 1 if j > 0 else ( nj - 1 if y_periodic else 0 )            # [NP] mirror ghost row (was periodic)
            jp = j + 1 if j < nj - 1 else ( 0 if y_periodic else nj - 1 )       # [NP]
            wn = 1.0 if ( y_periodic or j < nj - 1 ) else 0.0                   # [NP] v-tendency = 0 on north wall
            for i in range(ni):
                im = i - 1 if i > 0 else ni - 1
                ip = i + 1 if i < ni - 1 else 0
                vdot_at_u[k,j,i] = 0.5 * ( 0.5 * ( vdot[k,j,i]  + vdot[k,j,im]  )
                                         + 0.5 * ( wn * vdot[k,jp,i] + wn * vdot[k,jp,im] ) )   # [NP]
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

    # block B: record/replace (u,v) at the prescribed boundary columns, AFTER the
    # implicit solve but BEFORE restoring -- the same pre-restoring phase point as
    # block A's h. Serial loops only.
    n_bc = bc_cols.shape[0]                                                     
    if bc_mode == 1:                                                            # record
        for k in range(nk):                                                     
            for j in range(nj):                                                 
                for c in range(n_bc):                                           
                    u_bc[k,j,c] = u[k,j,bc_cols[c]]                             
                    v_bc[k,j,c] = v[k,j,bc_cols[c]]                             
    elif bc_mode == 2:                                                          # compare + replace
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

    # east Orlanski for u, v: each field estimates its OWN (rx, ry) independently.
    if obc_on == 1:                                                             
        _orlanski_east(u, u_prev, b_obc, rx_u,
                       nudging_mode, u_ext, alpha_in,
                       rxy_u,
                       inflow_persistent,                                       # [PERSIST]
                       ry_on, y_periodic,
                       r_max,                                                   # [RMAX]
                       inflow_mode, u_ext_out,                                  # 
                       use_nn_rx, u_rx_pre, u_ry_pre)                          
        _orlanski_east(v, v_prev, b_obc, rx_v,
                       nudging_mode, v_ext, alpha_in,
                       rxy_v,
                       inflow_persistent,                                       # [PERSIST]
                       ry_on, y_periodic,
                       r_max,                                                   # [RMAX]
                       inflow_mode, v_ext_out,                                  # 
                       use_nn_rx, v_rx_pre, v_ry_pre)                          

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

    # [NP] South wall sits on the v-row j=0 (y=0): no normal flow through it.
    # Enforced last so it also overrides the east OBC at the SW/SE corners.
    if not y_periodic:
        for k in range(nk):
            for i in range(ni):
                v[k, 0, i] = 0.0

class SSWEM:
    """(S)tacked (S)hallow (W)ater (E)quation (M)odel"""

    def __init__(self, ni, g, Ho, Lx, fo, beta, epsilon, nu_h, nu_v=0,
                 h_zonal_relax=0, h_relax=None, u_relax=None, v_relax=None,
                 h_interface_relax=0, h_interface_target=None,   # [INTERFACE_RELAX]
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
        y_periodic - [YP] Meridional boundary condition. False (DEFAULT) for a
                  zonal channel with free-slip solid walls at y=0 and y=Ly: no
                  normal flow through either wall and no wall stress (du/dy = 0).
                  True for the doubly-periodic domain (e.g. a Phillips
                  baroclinic-instability run). The i-direction stays periodic
                  either way, so mass sum(h) is conserved exactly in both cases.
                  NOTE: the default was flipped from True to False to match the
                  helpers module -- existing scripts that relied on the implicit
                  doubly-periodic default must now pass y_periodic=True.
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
        self.h_zonal_target = np.tile(self.Ho[:, None], (1, self.nj)).astype(float)
        # Interface relaxation target : default is the resting interface. 
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

        # dummy variables for OBC
        self._bc_cols_dummy = np.zeros(1, dtype=np.int64)                        
        self._bc_dummy = np.zeros((self.nk, self.nj, 1))                         
        self._diff_dummy = np.zeros(2)                                          
        self._obc_prev_dummy = np.zeros((self.nk, self.nj, 2))                  
        self._obc_ext_dummy = np.zeros((self.nk, self.nj))                      
        self._obc_rxy_dummy = np.zeros((self.nk, self.nj, 2))
        self._nn_rx_dummy = np.zeros((self.nk, self.nj))                       
        self._nn_ry_dummy = np.zeros((self.nk, self.nj))
        # [CL] per-step centre-line store; stays None until a run is called with store_center=True
        self.center_line = None                                                 # [CL]

        # Derived parameters
        self.f = self.fo + self.beta * self.yq # Coriolis is at q-points
        # [NP] q2u is a j-average and wraps at j=nj-1, which is wrong against a
        # [NP] wall; the analytic form is identical in the interior and correct
        # [NP] on the north row, so use it when the walls are on.
        self.f_at_u = ( _nb_q2u( self.f ) if self.y_periodic
                        else self.fo + self.beta * self.yu ) # Coriolis interpolated to u-points
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

    # cubint shape jet
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

    # Smoothstep family; _bumpint's taper needs this, so keep the two together.
    def _smoothint(x, xa, xb, order=5):                                            # [JET]
        """Smoothstep of odd `order` interpolating f(xa)=0 -> f(xb)=1, with the first
        (order-1)/2 derivatives vanishing at both ends. order=3 is _cubint exactly;
        order=5 and 7 are the quintic / septic smootherstep polynomials."""
        z = np.minimum( np.maximum( x - xa, 0 ) / ( xb - xa ), 1 )                  # [JET]
        n = ( int(order) - 1 ) // 2                                                 # [JET]
        if 2 * n + 1 != int(order) or not ( 1 <= n <= 3 ):                          # [JET]
            raise ValueError(f"[JET] order must be 3, 5 or 7; got {order}")         # [JET]
        if n == 1:                                                                  # [JET]
            return ( 1 + 2 * ( 1 - z ) ) * z**2          # identical to _cubint     # [JET]
        if n == 2:                                                                  # [JET]
            return ( ( 6.0 * z - 15.0 ) * z + 10.0 ) * z**3                         # [JET]
        return ( ( ( -20.0 * z + 70.0 ) * z - 84.0 ) * z + 35.0 ) * z**4            # [JET]

    # arccosh shape jet.
    def _bumpint(y, yc, hw, shape='sech2', taper=None, order=5):
        """Bump profile centred at yc with half-width hw (f = 1/2 at |y-yc| = hw).
        shape='sech2' gives sech^2(a*r), r = |y-yc|/hw, a = arccosh(sqrt(2)). With
        `taper` set, the tail is smoothly cut to exactly zero over r in [taper,
        taper+1] using _smoothint of the given order, giving compact support."""
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
        midline (y = Ly/2) and spans [Ly/2 - width, Ly/2 + width].

        [JET] The half-width used to be hard-coded at 0.1*Ly (the jet spanned
        y/Ly in [0.4, 0.6]); it is now an explicit parameter, so the previous
        profile is reproduced exactly by passing width = 0.1 * Ly."""
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
        nsteps = nsamps * samp
        self._print_run_info(dt, nsteps)
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

    # Per-step centre-row recorder, shared by the OBC drivers.
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

    # Clean control run driver, run the control run while storing the data for OBC runs, also support
    # storing center line for plotting Hovmoller.
    def run_control(self, dt, samp, nsamps, bc_cols,
                    store_bc=True, store_downstream=True, probe_cols=None,      # [OBC] probe_cols replaces probe_i0/n_probe
                    store_row=False, j_row=None, row_q=False):
        """
        dt      - Time step [s]
        samp    - Steps between samples [steps]
        nsamps  - Number of samples [steps*samp]
        bc_cols - sequence of column indices to record when store_bc=True (any
                  length >= 1; e.g. the westernmost sponge columns). Same i-index
                  used for u, v, h; all rows/layers. Ignored when store_bc=False.

        store_bc         - True (default): record (u,v,h) at bc_cols every step,
                  the data a later run_obc replays. False: clean run with no
                  per-step boundary stores; h/u/v_bc_all return None.
        store_downstream - if True, ALSO store (u,v,h) every step at an arbitrary
                  set of interior columns (a diagnostic probe somewhere
                  downstream; not replaced -- purely recorded). Independent of
                  store_bc.
        probe_cols - sequence of column indices to store (any length >= 1, need
                  not be contiguous). Required when store_downstream=True.       # [OBC]
                  Replaces the old probe_i0 / n_probe pair: the contiguous block
                  [i0, i0+n) is now written explicitly as range(i0, i0+n).
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
            # [OBC] guard against the old positional call run_control(..., probe_i0, n_probe):
            # a bare int would bind to probe_cols (silently a 1-column probe) and n_probe
            # would land on store_row. Demand a sequence so that mistake raises instead.
            if np.ndim(probe_cols) == 0:                                        # [OBC]
                raise ValueError(f"[OBC] probe_cols must be a SEQUENCE of column indices, "
                                 f"got the scalar {probe_cols}. The old probe_i0/n_probe API "
                                 f"is gone: pass e.g. probe_cols=range(i0, i0+n) or [i0].")
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
                  "(run_obc needs store_bc=True)")
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
                h_probe_all[iter-1] = self.h[:, :, probe_cols]                   # [OBC]
                u_probe_all[iter-1] = self.u[:, :, probe_cols]                   # [OBC]
                v_probe_all[iter-1] = self.v[:, :, probe_cols]                   # [OBC]
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
            probe = { 'cols': probe_cols,                                        # [OBC]
                      'x_km': self.xh1[probe_cols] / 1e3,                        # [OBC]
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

    # OBC run driver. 
    def run_obc(self, dt, samp, nsamps, prev_cols,
                h_bc_all, u_bc_all, v_bc_all, b_obc,
                nudging=False, h_ext_all=None, u_ext_all=None, v_ext_all=None,
                alpha_in=0.5,
                store_downstream=False, probe_i0=None, n_probe=1,
                ry_2d=True,
                inflow_persistent=False,   # [PERSIST] True -> clamp rx>=0 so inflow rows persist instead of reading phi_ext
                r_max=R_MAX_DEFAULT,       # [RMAX] outflow CFL clamp on |rx|,|ry|
                inflow_mode=0,             #  0 = prescribe/nudge toward col-b data; 1 = radiation-inflow reading col-(b+1) data; 2 = persistence
                h_ext_out_all=None, u_ext_out_all=None, v_ext_out_all=None,   #  per-step col-(b+1) exterior data @ n+1; REQUIRED when inflow_mode=1
                store_center=False, center_j=None, center_stride=1,   # [CL]
                center_fields=('h', 'u', 'v'),                        # [CL]
                center_i0=0, center_i1=None):                         # [CL] None -> b_obc+1 (interior of the OBC)
        """West: prescribe (replace) `prev_cols` from stored data each step (the sponge
        band recorded by run_control). East: Orlanski radiation at column `b_obc`,
        computed only from the interior columns b-1, b-2; the ghost column b_obc+1
        copies b_obc, so everything east of b_obc must be cropped from diagnostics.

        Runs on THIS instance: the meridional boundary condition is this instance's
        y_periodic, so build the truncated model with the SAME y_periodic (and grid,
        bathymetry, sponge, targets) as the control whose stores it replays.

        Returns (u, v, h, time), `diffs` (time axis only) and `rx` (east phase-speed
        diagnostics: per-step mean/max r_x per field, per-row applied (r_x, r_y) per
        field, and the optional probe).

        Each field (h, u, v) estimates its OWN east-boundary phase speed independently.
        On outflow the Courant numbers are clamped to rx <= r_max and |ry| <= r_max;
        see _orlanski_east.

        r_max : [RMAX] outflow (and inflow_mode=1) CFL clamp on |rx| and |ry|.
            Default R_MAX_DEFAULT = 1/50, matching the helpers module. This used to be
            derived internally as rx_max = dt/dx, ry_max = dt/dy; those coincide with
            1/50 only on the standard grid (dx = dy = 10 km, dt = 200 s). On any other
            (dt, dx, dy) pass r_max explicitly to reproduce the old behaviour.
        inflow_persistent : [PERSIST] True -> clamp rx (and ry) to 0 whenever the raw
            rx is negative, so inflow rows persist (phi_b^{n+1} = phi_b^n) and the
            external data is never read.
        inflow_mode :  east inflow-branch selector.
            0 (default) -> unchanged: prescribe/nudge toward the col-b data.
            1 -> radiation-INFLOW scheme,
                   phi_b^{n+1} = (phi_b^n - rx*phi_ext_out - ry*dy_b) / (1 - rx),  rx < 0,
                 reading the col-(b+1) exterior data (h/u/v_ext_out_all) recorded at
                 n+1; with nudging=True the result is additionally nudged toward the
                 col-b data with alpha_in. |rx|,|ry| are clamped at r_max.
            2 -> persistence on inflow rows.
        h_ext_out_all, u_ext_out_all, v_ext_out_all :  per-step external col-(b+1)
            data at n+1, shape (>=nsteps,nk,nj) or (...,1) -- same per-step indexing
            convention as h/u/v_ext_all. REQUIRED when inflow_mode=1, ignored otherwise.

        nudging : INFLOW treatment at the east OBC column.
            False (default) -> PRESCRIBE: on inflow hard-set column b to the recorded
                  external data (h/u/v_ext_all).
            True  -> NUDGE: on inflow relax column b toward the external data with
                  coefficient alpha_in (alpha_in = 1 == hard prescribe).
        h_ext_all, u_ext_all, v_ext_all : per-step external (recorded) col-b data,
            shape (>=nsteps, nk, nj) or (>=nsteps, nk, nj, 1) (e.g. a run_control
            probe at probe_i0=b_obc). ALWAYS required -- the inflow branch reads them
            in BOTH modes.
        alpha_in : inflow nudging coefficient in (0,1]; only used when nudging=True.
        store_downstream : if True, ALSO store (u,v,h) EVERY STEP at a block of
            columns (post-step state, same convention as run_control's probe).
            Returned as rx['probe'] (None when off).
        probe_i0, n_probe : start column index and number of contiguous columns.
        ry_2d : True -> full 2D Orlanski; False -> 1D Orlanski limit (r_y == 0),
            everything else identical.
        store_center : [CL] True keeps row `center_j` (default nj//2) at EVERY step
            over columns [center_i0, center_i1); center_i1=None -> b_obc+1, so the
            exterior copy at b+1 is excluded. Result on `self.center_line` (a dict
            of float32 (nrec,nk,n_cl) arrays plus the axes), NOT in the return tuple.
        center_stride : [CL] keep every Nth step (default 1 = every step).
        center_fields : [CL] which of ('h','u','v') to record.
        """
        samp = int(samp); nsamps = int(nsamps)
        nsteps = samp * nsamps
        prev_cols = np.ascontiguousarray(np.asarray(prev_cols, dtype=np.int64).ravel())
        if prev_cols.size < 1:
            raise ValueError("[OBC] prev_cols must contain at least one column to replay")
        b = int(b_obc)
        nudging_mode = int(bool(nudging))                # 0 = prescribe, 1 = inflow nudging
        alpha_in = float(alpha_in)
        if nudging_mode == 1 and not (0.0 < alpha_in <= 1.0):
            raise ValueError(f"[OBC] alpha_in must be in (0, 1] when nudging=True; got {alpha_in}")
        ry_on = int(bool(ry_2d))                         # [RY1D] 0 -> tangential term dropped -> 1D Orlanski
        inflow_persistent_mode = int(bool(inflow_persistent))  # [PERSIST] 0 = inflow uses phi_ext (prescribe/nudge), 1 = inflow persists
        if h_bc_all is None or u_bc_all is None or v_bc_all is None:            # run_control(store_bc=False) returns None
            raise ValueError("[OBC] run_obc needs the control's per-step west-band stores "
                             "(h/u/v_bc_all): re-run run_control with store_bc=True")
        if h_bc_all.shape[0] < nsteps:
            raise ValueError(f"[OBC] stored west data has {h_bc_all.shape[0]} < nsteps={nsteps}")
        if h_bc_all.shape[-1] != prev_cols.size:                                # silent index corruption guard
            raise ValueError(f"[OBC] stored west data has {h_bc_all.shape[-1]} columns "
                             f"!= len(prev_cols)={prev_cols.size}; pass the matching prev_cols")

        # external col-b data is read on inflow in BOTH modes, so it is always required
        if h_ext_all is None or u_ext_all is None or v_ext_all is None:
            raise ValueError("[OBC] run_obc needs h_ext_all/u_ext_all/v_ext_all (recorded "
                             "col-b external data, e.g. a run_control probe at b_obc)")
        def _prep_ext(a, name):
            a = np.asarray(a, dtype=np.float64)
            if a.ndim == 4 and a.shape[-1] == 1:
                a = a[..., 0]
            if a.ndim != 3 or a.shape[1:] != (self.nk, self.nj):
                raise ValueError(f"[OBC] {name} must be (>=nsteps,nk,nj) "
                                 f"or (>=nsteps,nk,nj,1); got {a.shape}")
            if a.shape[0] < nsteps:
                raise ValueError(f"[OBC] {name} has {a.shape[0]} < nsteps={nsteps}")
            return np.ascontiguousarray(a)
        h_ext_all = _prep_ext(h_ext_all, "h_ext_all")
        u_ext_all = _prep_ext(u_ext_all, "u_ext_all")
        v_ext_all = _prep_ext(v_ext_all, "v_ext_all")

        #  col-(b+1) exterior data for the radiation-inflow scheme (recorded at n+1)
        if inflow_mode == 1:            
            if h_ext_out_all is None or u_ext_out_all is None or v_ext_out_all is None:  # 
                raise ValueError(" inflow_mode=1 needs h_ext_out_all/u_ext_out_all/"  # 
                                 "v_ext_out_all (recorded col-(b+1) external data at n+1, "  # 
                                 "e.g. the run_control probe at b_obc+1).")             # 
            h_ext_out_all = _prep_ext(h_ext_out_all, "h_ext_out_all")                   # 
            u_ext_out_all = _prep_ext(u_ext_out_all, "u_ext_out_all")                   # 
            v_ext_out_all = _prep_ext(v_ext_out_all, "v_ext_out_all")                   # 

        # optional per-step probe setup (validate + allocate only when enabled)
        if store_downstream:
            if probe_i0 is None:
                raise ValueError("[OBC] store_downstream=True requires probe_i0 "
                                 "(start column index)")
            probe_i0 = int(probe_i0); n_probe = int(n_probe)
            if n_probe < 1 or probe_i0 < 0 or probe_i0 + n_probe > self.ni:
                raise ValueError(f"[OBC] probe columns [{probe_i0}, "
                                 f"{probe_i0 + n_probe}) out of range [0, {self.ni})")
            sl_probe = slice(probe_i0, probe_i0 + n_probe)
            h_probe_all = np.zeros((nsteps, self.nk, self.nj, n_probe))
            u_probe_all = np.zeros((nsteps, self.nk, self.nj, n_probe))
            v_probe_all = np.zeros((nsteps, self.nk, self.nj, n_probe))

        # the OBC column must be interior: the scheme reads b-2, b-1 and writes b, b+1
        if not (2 <= b < self.ni - 1):
            raise ValueError(f"[OBC] b_obc={b} needs 2 <= b < ni-1={self.ni-1}")
        if np.any((prev_cols >= b - 2) & (prev_cols <= b)):
            raise ValueError(f"[OBC] prescribed cols must avoid the OBC stencil "
                             f"{{{b-2},{b-1},{b}}}; got {list(prev_cols)}")
        self._print_run_info(dt, nsteps)

        u = np.zeros((nsamps+1, self.nk, self.nj, self.ni))
        v = np.zeros((nsamps+1, self.nk, self.nj, self.ni))
        h = np.zeros((nsamps+1, self.nk, self.nj, self.ni))
        time = np.zeros((nsamps+1))
        u[0] = self.u; v[0] = self.v; h[0] = self.h; time[0] = self.time

        # [CL] centre-line setup; default cut = up to and including the OBC column b
        _c_on, _c_j, _c_st, _c_f, _c_sl = self._center_init(
            store_center, center_j, center_stride, center_fields, nsteps,
            center_i0, center_i1, i1_default=b + 1)

        # phi^n at columns [b-1, b], snapshotted before each step
        h_prev = np.zeros((self.nk, self.nj, 2))
        u_prev = np.zeros((self.nk, self.nj, 2))
        v_prev = np.zeros((self.nk, self.nj, 2))

        rx_h = np.zeros(2); rx_u = np.zeros(2); rx_v = np.zeros(2)   # per-step (mean, max) rx per field
        h_diff = np.zeros(2); u_diff = np.zeros(2); v_diff = np.zeros(2)
        rxh = np.zeros((nsteps, 2)); rxu = np.zeros((nsteps, 2)); rxv = np.zeros((nsteps, 2))
        # per-row, per-step APPLIED (rx,ry) for each field: (nsteps,nk,nj,2). [...,0]=rx, [...,1]=ry
        rxy_h_row = np.zeros((nsteps, self.nk, self.nj, 2))
        rxy_u_row = np.zeros((nsteps, self.nk, self.nj, 2))
        rxy_v_row = np.zeros((nsteps, self.nk, self.nj, 2))
        print("[OBC] east r_x estimated independently per field (h, u, v)")
        print(f"[OBC] east OBC scheme: {'2D Orlanski (r_y active)' if ry_on else '1D Orlanski (r_y = 0)'}")   # [RY1D]
        print(f"[OBC] east inflow scheme: "
              f"{f'NUDGING (alpha_in={alpha_in:.3g})' if nudging_mode else 'PRESCRIBE (hard-set col b = external)'}")
        print(f"[RMAX] outflow CFL clamp: |rx|,|ry| <= r_max = {r_max:.4g} "        # [RMAX]
              f"(dt/dx = {dt / self.dx:.4g}, dt/dy = {dt / self.dy:.4g})")          # [RMAX]
        if inflow_persistent_mode:                                                  # [PERSIST]
            print("[PERSIST] inflow rows PERSIST (rx,ry clamped to 0); the external "  # [PERSIST]
                  "col-b data is NOT read on inflow")                                  # [PERSIST]

        if inflow_mode == 1:
            print(f" inflow_mode=1: RADIATION-INFLOW scheme on inflow rows "    # 
                  f"(reads col b+1 exterior @ n+1; |rx| clamped at r_max={r_max:.3g}; "  # 
                  f"extra col-b nudge {'ON, alpha_in=%.3g' % alpha_in if nudging_mode else 'OFF'})")  # 
        elif inflow_mode == 2:      
            print(" inflow_mode=2: PERSISTENCE on inflow rows")                 # 

        nsamp = 0
        nrun = nsteps
        for it in range(1, nsteps + 1):
            # snapshot phi^n at [b-1, b] before the step
            h_prev[:] = self.h[:, :, b-1:b+1]
            u_prev[:] = self.u[:, :, b-1:b+1]
            v_prev[:] = self.v[:, :, b-1:b+1]
            # external col-b slice; read on inflow in BOTH modes
            h_ext_s = h_ext_all[it - 1]; u_ext_s = u_ext_all[it - 1]; v_ext_s = v_ext_all[it - 1]
            #  col-(b+1) exterior slice @ n+1 (None -> _step_core substitutes the dummy)
            if inflow_mode == 1:    
                h_ext_out_s = h_ext_out_all[it - 1]; u_ext_out_s = u_ext_out_all[it - 1]   
                v_ext_out_s = v_ext_out_all[it - 1]                                  
            else:                   
                h_ext_out_s = u_ext_out_s = v_ext_out_s = None                       
            # advance: bc_mode=2 replays the west band, obc_on=1 radiates at b
            self._step_core(dt, 2, prev_cols,
                            h_bc_all[it - 1], u_bc_all[it - 1], v_bc_all[it - 1],
                            h_diff, u_diff, v_diff,
                            obc_on = 1, b_obc = b,
                            h_prev = h_prev, u_prev = u_prev, v_prev = v_prev,
                            rx_h = rx_h, rx_u = rx_u, rx_v = rx_v,
                            rxy_h = rxy_h_row[it-1], rxy_u = rxy_u_row[it-1], rxy_v = rxy_v_row[it-1],
                            nudging_mode = nudging_mode, h_ext = h_ext_s,
                            u_ext = u_ext_s, v_ext = v_ext_s, alpha_in = alpha_in,
                            inflow_persistent = inflow_persistent_mode,          # [PERSIST]
                            ry_on = ry_on,
                            r_max = r_max,
                            inflow_mode = inflow_mode,                           
                            h_ext_out = h_ext_out_s, u_ext_out = u_ext_out_s,    
                            v_ext_out = v_ext_out_s)                             
            rxh[it-1] = rx_h; rxu[it-1] = rx_u; rxv[it-1] = rx_v
            if store_downstream:                              # post-step state (probe convention)
                h_probe_all[it-1] = self.h[:, :, sl_probe]
                u_probe_all[it-1] = self.u[:, :, sl_probe]
                v_probe_all[it-1] = self.v[:, :, sl_probe]

            if np.any(np.isnan(self.u)):
                print('Model has blown up!!! Stopping early')
                u = u[:nsamp]; v = v[:nsamp]; h = h[:nsamp]; time = time[:nsamp]
                nrun = it                                     # steps attempted
                self._center_trim(_c_on, it - 1, _c_st)       # [CL] drop blown step
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
              'rx_v_row': rxy_v_row[:nrun, :, :, 0], 'ry_v_row': rxy_v_row[:nrun, :, :, 1]}
        # bundle the optional per-step probe into rx (return signature unchanged)
        probe = None
        if store_downstream:
            probe = { 'cols': np.arange(probe_i0, probe_i0 + n_probe),
                      'x_km': self.xh1[probe_i0:probe_i0 + n_probe] / 1e3,
                      'h': h_probe_all[:nrun], 'u': u_probe_all[:nrun],
                      'v': v_probe_all[:nrun] }
        rx['probe'] = probe
        print(f"Mean r_x over run:  h = {rx['h_mean'].mean():.3f}  "
              f"u = {rx['u_mean'].mean():.3f}  v = {rx['v_mean'].mean():.3f}")
        return u, v, h, time, diffs, rx

    # Neural-network east-OBC driver
    def run_obc_nn(self, dt, samp, nsamps, prev_cols,
                   h_bc_all, u_bc_all, v_bc_all, b_obc,
                   h_ext_all, u_ext_all, v_ext_all,
                   phase_fn=None, nudging=True, alpha_in=1,
                   nn_field_codes=(0,),   # [MF] field codes that receive the NN; (0,)=h-only (back-compat), (0,1,2)=all fields
                   nn_inflow_corr=False,   # [INCORR] if True, phase_fn returns (rx,ry,dphi_in) and dphi_in is added to the inflow nudge (config 3.2)
                   nn_outflow_corr=False,   # [ANCH] if True (anchored_corr), phase_fn returns (rx,ry,dphi_out); dphi_out is added to the OUTFLOW radiate
                   inflow_mode=0,   # [IM] 1 -> radiation-inflow scheme on inflow rows (reads col-(b+1) exterior @ n+1); 0 -> unchanged nudge/prescribe
                   h_ext_out_all=None, u_ext_out_all=None, v_ext_out_all=None,   # [IM] per-step col-(b+1) exterior data @ n+1; REQUIRED when inflow_mode=1
                   r_max=None,   # [RMAX] outflow (and inflow_mode=1) CFL clamp on |rx|,|ry|; None -> unclamped (historical default, unchanged)
                   store_center=False, center_j=None, center_stride=1,   # [CL]
                   center_fields=('h', 'u', 'v'),                        # [CL]
                   center_i0=0, center_i1=None):                         # [CL] None -> b_obc+1 (interior of the OBC)
        """
            samp : steps between stored output frames
            phi_bc_all : per_step stored Western prescribed value
            phi_ext_all : external forcing at open boundary
            phase_fn : The Neural Network. Called as phase_fn(X, fcode) with X the
                (m,8) packed stencil and fcode 0/1/2 for h/u/v; returns (rx, ry), or
                (rx, ry, dphi) when nn_inflow_corr / nn_outflow_corr is set.

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

            [FRESH] Timing (matches run_obc / the analytic scheme exactly):
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

            Returns the 5-tuple (u, v, h, time, diffs); diffs carries the per-row
            APPLIED (rx, ry) for each field, mirroring run_obc's `rx` dict.
        """
        prev_cols = np.ascontiguousarray(np.asarray(prev_cols, dtype=np.int64).ravel())
        b = int(b_obc)
        nn_field_codes = tuple(int(c) for c in np.atleast_1d(nn_field_codes))
        alpha_eff = float(alpha_in) if nudging else 1.0

        # Total number of steps
        nsteps = nsamps * samp
        # Check that enough boundary and external data are available.
        if h_bc_all is None or u_bc_all is None or v_bc_all is None:            # run_control(store_bc=False) returns None
            raise ValueError("[OBC] run_obc_nn needs the control's per-step west-band stores "
                             "(h/u/v_bc_all): re-run run_control with store_bc=True")
        if h_bc_all.shape[0] < nsteps:
            raise ValueError(f"[OBC] stored west data has {h_bc_all.shape[0]} < nsteps={nsteps}")
        if h_bc_all.shape[-1] != prev_cols.size:                                # silent index corruption guard
            raise ValueError(f"[OBC] stored west data has {h_bc_all.shape[-1]} columns "
                             f"!= len(prev_cols)={prev_cols.size}; pass the matching prev_cols")

        # Prepare the external boundary data: make sure it has the expected shape (nsteps, nk, nj).
        def _prep_ext(a, name):
            a = np.asarray(a, dtype=np.float64)
            if a.ndim == 4 and a.shape[-1] == 1:
                a = a[..., 0]
            if a.ndim != 3 or a.shape[1:] != (self.nk, self.nj):
                raise ValueError(f"[OBC] {name} must be (>=nsteps,nk,nj[,1]); got {a.shape}")
            if a.shape[0] < nsteps:
                raise ValueError(f"[OBC] {name} has {a.shape[0]} < nsteps={nsteps}")
            return np.ascontiguousarray(a)
        h_ext_all = _prep_ext(h_ext_all, "h_ext_all")
        u_ext_all = _prep_ext(u_ext_all, "u_ext_all")
        v_ext_all = _prep_ext(v_ext_all, "v_ext_all")

        inflow_mode = int(inflow_mode)
        if inflow_mode == 1:                                                            # [IM]
            if h_ext_out_all is None or u_ext_out_all is None or v_ext_out_all is None:  # [IM]
                raise ValueError("[IM] inflow_mode=1 needs h_ext_out_all/u_ext_out_all/"  # [IM]
                                 "v_ext_out_all (col-(b+1) exterior data @ n+1)")       # [IM]
            h_ext_out_all = _prep_ext(h_ext_out_all, "h_ext_out_all")                   # [IM]
            u_ext_out_all = _prep_ext(u_ext_out_all, "u_ext_out_all")                   # [IM]
            v_ext_out_all = _prep_ext(v_ext_out_all, "v_ext_out_all")                   # [IM]

        if not (2 <= b < self.ni - 1):
            raise ValueError(f"[OBC] b_obc={b} needs 2 <= b < ni-1={self.ni-1}")
        if np.any((prev_cols >= b - 2) & (prev_cols <= b)):
            raise ValueError(f"[OBC] prescribed cols must avoid {{{b-2},{b-1},{b}}}; got {list(prev_cols)}")

        self._print_run_info(dt, nsteps)
        print(f" east phase speed from "
              f"{'analytic numpy (classical baseline)' if phase_fn is None else 'NEURAL NETWORK'}; "
              f"inflow {('NUDGE alpha_in=%.3g' % alpha_eff) if nudging else 'PRESCRIBE (hard)'}")
        if inflow_mode == 1:                                                           # [IM]
            print(f"[IM] inflow_mode=1: RADIATION-INFLOW scheme on inflow rows "       # [IM]
                  f"(reads col b+1 exterior @ n+1, "                                   # [IM]
                  f"{'clamped at r_max=%.3g' % r_max if r_max is not None else 'UNCLAMPED like this whole driver'}; "  # [IM]
                  f"extra col-b nudge {'ON, alpha_in=%.3g' % alpha_eff if nudging else 'OFF'})")  # [IM]
        print(f"Running (Neural-Network-OBC, "
              f"{'doubly periodic' if self.y_periodic else 'non-periodic N/S walls'})...")   # [YP]

        # Initialize the simulation output
        u = np.zeros((nsamps+1, self.nk, self.nj, self.ni))
        v = np.zeros((nsamps+1, self.nk, self.nj, self.ni))
        h = np.zeros((nsamps+1, self.nk, self.nj, self.ni))
        time = np.zeros((nsamps+1))

        # Initial state is copied from the model's current state.
        u[0] = self.u; v[0] = self.v; h[0] = self.h; time[0] = self.time

        # centre-line setup; default cut = up to and including the OBC column b
        _c_on, _c_j, _c_st, _c_f, _c_sl = self._center_init(
            store_center, center_j, center_stride, center_fields, nsteps,
            center_i0, center_i1, i1_default=b + 1)

        h_diff = np.zeros(2); u_diff = np.zeros(2); v_diff = np.zeros(2)   # west misfit scratch

        # tangential neighbours: free-slip mirror-clamp against walls, np.roll when the
        # j-direction wraps. y_periodic=False reproduces the [NP] mirror of the kernel.
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

        _rxy_rows = [np.zeros((nsteps, self.nk, self.nj, 2)) for _ in range(3)]

        _saved_rx = [None, None, None]  # [h, u, v] -> (nk,nj) or None
        _saved_ry = [None, None, None]
        # pre-allocated (nk,nj) arrays for the _step_core call (reused each step)
        _hrx = np.zeros((self.nk, self.nj)); _hry = np.zeros((self.nk, self.nj))
        _urx = np.zeros((self.nk, self.nj)); _ury = np.zeros((self.nk, self.nj))
        _vrx = np.zeros((self.nk, self.nj)); _vry = np.zeros((self.nk, self.nj))

        _h_prev = np.zeros((self.nk, self.nj, 2))
        _u_prev = np.zeros((self.nk, self.nj, 2))
        _v_prev = np.zeros((self.nk, self.nj, 2))
        _rx_h = np.zeros(2); _rx_u = np.zeros(2); _rx_v = np.zeros(2)
        _rxy_h = np.zeros((self.nk, self.nj, 2))
        _rxy_u = np.zeros((self.nk, self.nj, 2))
        _rxy_v = np.zeros((self.nk, self.nj, 2))
        # r_max for the kernel: r_max=None historically means UNCLAMPED, so pass
        # a huge bound that never binds (the kernel always clamps at its r_max arg)
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
            _h_prev[:] = self.h[:, :, b-1:b+1]
            _u_prev[:] = self.u[:, :, b-1:b+1]
            _v_prev[:] = self.v[:, :, b-1:b+1]

            # Mimic the exact continuity update law for h, producing h at n+1, so the
            # network then uses the same stencil (and the same timing) to predict the phase speed.
            _h_np1 = _continuity_np(self.h, self.u, self.v, dt, self.dx, self.dy,
                                    self.iter, self.y_periodic,
                                    float(self.h_interface_relax),
                                    self._h_interface_target, self.D)
            # 8-point stencil for the h field
            X8h = _X8(h_bm1, h_b, _h_np1[:, :, b-1].copy(), _h_np1[:, :, b-2].copy())
            Xfh = X8h.reshape(-1, 8)
            # classical (analytic) phase speed for h; overwritten below when the NN drives h
            rx_hf, ry_hf, _ = _phase_inflow_np(Xfh)
            if phase_fn is not None and 0 in nn_field_codes:
                if nn_inflow_corr:                              # [INCORR] rx,ry part only here
                    rx_hf, ry_hf, _ = phase_fn(Xfh, 0)
                elif nn_outflow_corr:                           # [ANCH] rx,ry part only here
                    rx_hf, ry_hf, _ = phase_fn(Xfh, 0)
                else:
                    rx_hf, ry_hf = phase_fn(Xfh, 0)
            if r_max is not None:                               # [RMAX] opt-in clamp
                _inf_hf = np.asarray(rx_hf) < 0.0
                if inflow_mode == 1:
                    rx_hf = np.where(_inf_hf, np.maximum(rx_hf, -r_max), np.minimum(rx_hf, r_max))
                else:
                    rx_hf = np.where(_inf_hf, rx_hf, np.minimum(rx_hf, r_max))
                ry_hf = np.clip(ry_hf, -r_max, r_max)

            # reshape the predictions back to (nk,nj) for the kernel
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
                            inflow_persistent=0, ry_on=1,
                            r_max=_r_max_val, inflow_mode=inflow_mode,
                            h_ext_out=h_ext_out_s, u_ext_out=u_ext_out_s,
                            v_ext_out=v_ext_out_s,
                            # h: THIS step's fresh (rx,ry) ([FRESH] above);
                            # u,v: previous step's post-step estimate (dynamically inert)
                            use_nn_rx=1,
                            h_rx_pre=_hrx, h_ry_pre=_hry,
                            u_rx_pre=_urx, u_ry_pre=_ury,
                            v_rx_pre=_vrx, v_ry_pre=_vry)

            # Post-step stencils for u and v, built from the already-updated state.
            _post_fields = [(self.u, u_bm1, u_b, u_ext_all, u_ext_out_all, 1),
                            (self.v, v_bm1, v_b, v_ext_all, v_ext_out_all, 2)]
            if _h_corr_mode:
                _post_fields.append((self.h, h_bm1, h_b, h_ext_all, h_ext_out_all, 0))
            for fld, bm1_n, b_n, ext_all, ext_out_all, fcode in _post_fields:
                X8 = _X8(bm1_n, b_n, fld[:, :, b-1].copy(), fld[:, :, b-2].copy())   # (nk,nj,8)
                Xf = X8.reshape(-1, 8)                       # (m,8) flat batch of stencils
                # analytic estimate first; the NN overrides it for its own field codes
                rx, ry, inflow = _phase_inflow_np(Xf)

                dphi_in = None
                dphi_out = None
                if phase_fn is not None and fcode in nn_field_codes:
                    if nn_inflow_corr:                                     # [INCORR]
                        rx, ry, dphi_in = phase_fn(Xf, fcode)
                    elif nn_outflow_corr:                                  # [ANCH]
                        rx, ry, dphi_out = phase_fn(Xf, fcode)
                    else:
                        rx, ry = phase_fn(Xf, fcode)
                    inflow = rx < 0.0
                if r_max is not None:                                      # [RMAX]
                    if inflow_mode == 1:
                        rx = np.where(inflow, np.maximum(rx, -r_max), np.minimum(rx, r_max))
                    else:
                        rx = np.where(inflow, rx, np.minimum(rx, r_max))
                    ry = np.clip(ry, -r_max, r_max)
                _rxy_rows[fcode][it-1, :, :, 0] = np.asarray(rx).reshape(self.nk, self.nj)
                _rxy_rows[fcode][it-1, :, :, 1] = np.asarray(ry).reshape(self.nk, self.nj)

                _saved_rx[fcode] = np.asarray(rx, dtype=np.float64).reshape(self.nk, self.nj).copy()
                _saved_ry[fcode] = np.asarray(ry, dtype=np.float64).reshape(self.nk, self.nj).copy()

                # direct post-step overwrite of column b
                upd = _obc_update_np(Xf, rx, ry, inflow, ext_all[it-1].reshape(-1), alpha_eff,
                                     dphi_in=dphi_in, dphi_out=dphi_out,   # [INCORR]  # [ANCH]
                                     inflow_mode=inflow_mode,              # [IM]
                                     phi_ext_out=(ext_out_all[it-1].reshape(-1) if inflow_mode == 1 else None),  # [IM]
                                     inflow_nudge=bool(nudging))           # [IM]
                fld[:, :, b] = upd.reshape(self.nk, self.nj)
                # the ghost columns follow column b, as in _orlanski_east
                fld[:, :, b+1] = fld[:, :, b].copy()
                if b + 2 < self.ni:                                        # [TRUNC]
                    fld[:, :, b+2] = fld[:, :, b].copy()                   # [TRUNC]

            if not self.y_periodic:                                        # [YP]
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

    def _step_core(self, dt, bc_mode, bc_cols,
                   h_bc, u_bc, v_bc, h_diff, u_diff, v_diff,
                   obc_on=0, b_obc=0,
                   h_prev=None, u_prev=None, v_prev=None,
                   rx_h=None, rx_u=None, rx_v=None,
                   rxy_h=None, rxy_u=None, rxy_v=None,
                   nudging_mode=0, h_ext=None, u_ext=None, v_ext=None, alpha_in=0.0,
                   ry_on=1,
                   inflow_persistent=0,                                          # [PERSIST] 1 -> inflow rows persist (rx clamped >= 0)
                   r_max=R_MAX_DEFAULT,                                          # [RMAX] outflow CFL clamp on |rx|,|ry|
                   inflow_mode=0,
                   h_ext_out=None, u_ext_out=None, v_ext_out=None,               #  per-field col-(b+1) exterior data @ n+1, (nk,nj)
                   use_nn_rx=0,                                                  
                   h_rx_pre=None, h_ry_pre=None,                               
                   u_rx_pre=None, u_ry_pre=None,                                
                   v_rx_pre=None, v_ry_pre=None):                               
        """One time step with optional west-band record/replace (bc_mode 1/2 on
        bc_cols) and optional east Orlanski OBC (obc_on=1 at column b_obc).
        bc_mode=0 with obc_on=0 reduces to the plain step(). The dummy fills keep
        _step_numba's argument types stable across all call paths."""
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
        # External forcing at b+1, for the new inflow radiation scheme (inflow_mode=1). The scheme reads the col-(b+1) exterior
        if h_ext_out is None : h_ext_out = self._obc_ext_dummy                   # 
        if u_ext_out is None : u_ext_out = self._obc_ext_dummy                   # 
        if v_ext_out is None : v_ext_out = self._obc_ext_dummy                   # 
        # Dummies for the neural-network rx,ry arrays (use_nn_rx=1) when they are not provided
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
                    float(self.h_interface_relax), self._h_interface_target,      # [INTERFACE_RELAX]
                    bc_mode, bc_cols, h_bc, u_bc, v_bc, h_diff, u_diff, v_diff,
                    obc_on, int(b_obc), h_prev, u_prev, v_prev, rx_h, rx_u, rx_v,
                    rxy_h, rxy_u, rxy_v,
                    int(nudging_mode), h_ext, u_ext, v_ext, float(alpha_in),
                    int(inflow_persistent),                                       # [PERSIST] [ORDER] before ry_on, as in the helpers module
                    int(ry_on),                                                   #[RY1D]
                    self.y_periodic,
                    float(r_max),                                                 # [RMAX]
                    int(inflow_mode), h_ext_out, u_ext_out, v_ext_out,            # 
                    int(use_nn_rx),                                              
                    h_rx_pre, h_ry_pre,                                          
                    u_rx_pre, u_ry_pre,                                          
                    v_rx_pre, v_ry_pre)                                          
        self.time += dt
        self.iter += 1

    def step(self, dt):
        """
        dt   - Time step [s]
        """
        # Sponge gates are cached by the h/u/v_relax setters (a bool keeps the
        # JIT argument types stable and lets numba skip a term when its sponge
        # is off), so no per-step recomputation is needed.
        self._step_core(dt, 0, self._bc_cols_dummy,                             # plain step = no record/replace, no OBC
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
