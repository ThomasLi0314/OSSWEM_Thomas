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

######################################## Orlanski OBC ########################################
# OBC Orlanski Implementation [OBC - Orlanski]
# This is a 1D SCHEME
@njit(cache=True)
def _orlanski_east(phi, phi_prev, b_obc, rx_out):
    """
        b here is the column for implementing the orlanski. 
    """
    nk, nj, ni = phi.shape
    # Initial two variables for diagonise, sum is used for average, max is used for maximum.
    maxr = 0.0
    sumr = 0.0

    for k in range(nk):
        for j in range(nj):
            pim1 = phi[k, j, b_obc-1] #phi^{n+1}_{b-1}
            pim2 = phi[k, j, b_obc-2] #phi^{n+1}_{b-2}
            dphi_t = pim1 - phi_prev[k,j, 1] # - phi^n_{b-1}
            dphi_x = pim1 - pim2
            if dphi_x == 0.0:
                rx = 0.0
            else:
                rx = -dphi_t / dphi_x

            # Have question, how to handle the case, why?
            if rx < 0.0:
                rx = 0.0
            else:
                if rx > 1.0:
                    rx = 1.0
                phi[k,j,b_obc] = (phi_prev[k,j,2] + rx * pim1) / (1.0 + rx)

            sumr += rx
            if rx > maxr:
                maxr = rx
    rx_out[0] = sumr / (nk * nj)
    rx_out[1] = maxr


@njit(parallel=True, cache=True)
def _step_numba(u, v, h, D, taux, tauy, f, f_at_u, f_at_v,
                dt, dx, dy, g, epsilon, nu_h, nu_v, alpha_f, alpha_nu,
                h_zonal_target, h_target, u_target, v_target,
                h_zonal_relax, h_relax, u_relax, v_relax,
                h_relax_on, u_relax_on, v_relax_on, hsub, iter_num,
                bc_mode, bc_cols, h_bc, u_bc, v_bc, h_diff, u_diff, v_diff,
                obc_on, b_obc, h_prev, u_prev, v_prev, rx_h, rx_u, rx_v
                ):  # Updated the last column of input for Orlanski [OBC - Orlanski]
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
            jm = j - 1 if j > 0 else nj - 1
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
                    jm = j - 1 if j > 0 else nj - 1
                    for i in range(ni):
                        vi = v[k,j,i]
                        vp = vi if vi > 0.0 else 0.0
                        vn = vi if vi < 0.0 else 0.0
                        hv[k,j,i] = vp * h[k,jm,i] + vn * h[k,j,i]
            for k in range(nk):
                for j in prange(nj):
                    jp = j + 1 if j < nj - 1 else 0
                    for i in range(ni):
                        h[k,j,i] -= cy * ( hv[k,jp,i] - hv[k,j,i] )

    # [OBC] block A: record/replace h at the prescribed boundary columns (bc_cols,
    # [OBC] any length), done AFTER continuity (h^{n+1}) but BEFORE the momentum step
    # [OBC] reads h, so a replaced h feeds the subsequent (u,v) solve. Serial range
    # [OBC] loops only (no prange): matches the file's rule that reductions stay explicit.
    n_bc = bc_cols.shape[0]                                   # [OBC] number of columns
    if bc_mode == 1:                                          # [OBC] record
        for k in range(nk):
            for j in range(nj):
                for c in range(n_bc):
                    h_bc[k,j,c] = h[k,j,bc_cols[c]]
    elif bc_mode == 2:                                        # [OBC] compare + replace
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
    
    # [OBC - Orlanski]
    if obc_on == 1:
        _orlanski_east(h, h_prev, b_obc, rx_h)

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
            jm = j - 1 if j > 0 else nj - 1
            jp = j + 1 if j < nj - 1 else 0
            for i in range(ni):
                im = i - 1 if i > 0 else ni - 1
                ip = i + 1 if i < ni - 1 else 0
                ui = u[k,j,i];   up    = ui  if ui  > 0.0 else 0.0
                uip = u[k,j,ip]; un_ip = uip if uip < 0.0 else 0.0
                vi = v[k,j,i];   vp    = vi  if vi  > 0.0 else 0.0
                vjp = v[k,jp,i]; vn_jp = vjp if vjp < 0.0 else 0.0
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
            jm = j - 1 if j > 0 else nj - 1
            jp = j + 1 if j < nj - 1 else 0
            for i in range(ni):
                im = i - 1 if i > 0 else ni - 1
                ip = i + 1 if i < ni - 1 else 0

                # Masked PV at q-points (j,i), (jp,i), (j,ip):
                #   q = (f + vx - uy) ; q *= r ; q *= (hqp*r),  r = 1/(hqp+hsub)
                vort_c  = f[j, i]  + ( v[k,j,i]  - v[k,j,im]  ) * rdx - ( u[k,j,i]  - u[k,jm,i]  ) * rdy
                r_c     = 1.0 / ( hq_pre[k,j,i] + hsub )
                qpv_c   = ( vort_c * r_c ) * ( hq_pre[k,j,i] * r_c )
                # qpv_c   = ( vort_c * r_c )
                vort_jp = f[jp, i] + ( v[k,jp,i] - v[k,jp,im] ) * rdx - ( u[k,jp,i] - u[k,j,i]   ) * rdy
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
                            + qpv_jp * 0.5 * ( hv[k,jp,i] + hv[k,jp,im] ) )
                qhu = 0.5 * ( qpv_c  * 0.5 * ( hu[k,j,i]  + hu[k,jm,i]  )
                            + qpv_ip * 0.5 * ( hu[k,j,ip] + hu[k,jm,ip] ) )
            
    
                # qhv = 0.5 * ( vort_c * 0.5 * ( v[k,j,i]  + v[k,j,im]  )
                #             + vort_jp * 0.5 * ( v[k,jp,i] + v[k,jp,im] ) )
                # qhu = 0.5 * ( vort_c  * 0.5 * ( u[k,j,i]  + u[k,jm,i]  )
                #             + vort_ip * 0.5 * ( u[k,j,ip] + u[k,jm,ip] ) )

                # Stress tensor: nu_h*h*D_tension at h-points {(j,i),(j,im),(jm,i)};
                # nu_h*hq*D_shear at q-points {(j,i),(jp,i),(j,ip)}.
                Dt_c  = ( u[k,j,ip] - u[k,j,i]  ) * rdx - ( v[k,jp,i] - v[k,j,i]  ) * rdy
                nuhDt_c  = nu_h * h[k,j,i]  * Dt_c
                Dt_im = ( u[k,j,i]  - u[k,j,im] ) * rdx - ( v[k,jp,im]- v[k,j,im] ) * rdy
                nuhDt_im = nu_h * h[k,j,im] * Dt_im
                Dt_jm = ( u[k,jm,ip]- u[k,jm,i] ) * rdx - ( v[k,j,i]  - v[k,jm,i] ) * rdy
                nuhDt_jm = nu_h * h[k,jm,i] * Dt_jm

                Ds_c  = ( u[k,j,i]  - u[k,jm,i] ) * rdy + ( v[k,j,i]  - v[k,j,im]  ) * rdx
                hqp_c  = min( min( h[k,j,i],  h[k,j,im]  ), min( h[k,jm,i], h[k,jm,im] ) )
                nuhqDs_c  = nu_h * hqp_c  * Ds_c
                Ds_jp = ( u[k,jp,i] - u[k,j,i]  ) * rdy + ( v[k,jp,i] - v[k,jp,im] ) * rdx
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

    # --- implicit TDMAH2 (cancellation-free; recurrence identical to reference,
    # done per column with scalar locals) ---
    # Fused interpolation of the cross-component accelerations: vdot to
    # u-points = q2u(v2q(vdot)), udot to v-points = q2v(u2q(udot)).
    vdot_at_u = np.empty((nk, nj, ni))
    udot_at_v = np.empty((nk, nj, ni))
    for k in range(nk):
        for j in prange(nj):
            jm = j - 1 if j > 0 else nj - 1
            jp = j + 1 if j < nj - 1 else 0
            for i in range(ni):
                im = i - 1 if i > 0 else ni - 1
                ip = i + 1 if i < ni - 1 else 0
                vdot_at_u[k,j,i] = 0.5 * ( 0.5 * ( vdot[k,j,i]  + vdot[k,j,im]  )
                                         + 0.5 * ( vdot[k,jp,i] + vdot[k,jp,im] ) )
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
        _orlanski_east(u, u_prev, b_obc, rx_u)
        _orlanski_east(v, v_prev, b_obc, rx_v)

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

class SSWEM:
    """(S)tacked (S)hallow (W)ater (E)quation (M)odel"""

    def __init__(self, ni, g, Ho, Lx, fo, beta, epsilon, nu_h, nu_v=0,
                 h_zonal_relax=0, h_relax=None, u_relax=None, v_relax=None,
                 h_target=None, u_target=None, v_target=None,
                 hsub=1e-12, nj=None, Ly=None):
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
        self.Lx = Lx
        self.fo = fo
        self.beta = beta
        self.epsilon = epsilon
        self.nu_h = nu_h
        self.nu_v = float(nu_v)
        self.hsub = hsub
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

        # [OBC] cached dummy arrays so the normal step path passes the boundary
        # [OBC] record/replace args to _step_numba with bc_mode=0 (a no-op) without
        # [OBC] per-step allocation. Real (nk,nj,n_bc) / (n_bc,) arrays are passed only
        # [OBC] by run_record_bc / run_replace_bc; here a single-column placeholder is
        # [OBC] enough since the blocks are skipped when bc_mode=0.
        self._bc_cols_dummy = np.zeros(1, dtype=np.int64)
        self._bc_dummy = np.zeros((self.nk, self.nj, 1))
        self._diff_dummy = np.zeros(2)
        # [OBC - Orlanski]
        self._obc_prev_dummy = np.zeros((self.nk, self.nj, 3)) #place holder.


        # Derived parameters
        self.f = self.fo + self.beta * self.yq # Coriolis is at q-points
        self.f_at_u = _nb_q2u( self.f ) # Coriolis interpolated to u-points
        self.f_at_v = _nb_q2v( self.f ) # Coriolis interpolated to v-points
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

    # [OBC] Like run(), but records (u,v,h) at the two prescribed boundary columns
    # [OBC] EVERY step: h at the end-of-continuity phase, (u,v) at the end-of-solve
    # [OBC] phase (both pre-restoring; see blocks A/B in _step_numba). These stores
    # [OBC] are what run_replace_bc later injects to drive an identical second run.
    def run_record_bc(self, dt, samp, nsamps, bc_cols,
                      store_downstream=False, probe_i0=None, n_probe=1):
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
                break
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

    # [OBC] Like run(), but at EVERY step compares the freely-evolved boundary
    # [OBC] columns against the stored data and overwrites them: h right after
    # [OBC] continuity, (u,v) right after the implicit solve (see blocks A/B).
    # [OBC] With the sponge active and stored data from the same IC, the diffs
    # [OBC] should be ~0 (the overwrite is a no-op) -> identical dynamics.
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

    # [OBC - Orlanski] New Driver for OBC with Orlanski
    def run_obc(self, dt, samp, nsampes, prev_cols, 
                h_bc_all, u_bc_all, v_bc_all, b_obc):
        """West: prescribe (replace) `prev_cols` from stored data each step (the existing  # [OBC-E]
        sponge-edge band). East: 1D Orlanski radiation at column `b_obc`, computed only    # [OBC-E]
        from interior columns b-1,b-2 (interior-determined; periodicity kept).             # [OBC-E]
        Returns (u,v,h,time), `diffs` (west misfit, ~0), and `rx` (east phase-speed diag)."""  # [OBC-E]
        prev_cols = np.ascontiguousarray(np.asarray(prev_cols, dtype=np.int64).ravel()) 
        b = int(b_obc)

        nsteps = nsampes * samp
        # Check if the shape of the stored boundary data matches the expected shape based on nsteps and bc_cols.
        if h_bc_all.shape[0] < nsteps:
            raise ValueError(f"[OBC-E] stored west data has {h_bc_all.shape[0]} < nsteps={nsteps}")  # [OBC-E]
        
        # Check if the location of the Western open boundary lies in the interior
        if not (2 <= b < self.ni - 1):
            raise ValueError(f"[OBC-B] b_obc={b} needs 2 <= b < ni-1={self.ni-1}")
        if np.any((prev_cols >= b - 2) & (prev_cols <= b)):
            raise ValueError(f"[OBC-B] prescribed cols must avoid the OBC stencil "
                             f"{{{b-2},{b-1},{b}}}; got {list(prev_cols)}")
        self._print_run_info(dt, nsteps)  

        u = np.zeros((nsampes+1, self.nk, self.nj, self.ni))
        v = np.zeros((nsampes+1, self.nk, self.nj, self.ni))
        h = np.zeros((nsampes+1, self.nk, self.nj, self.ni))
        time = np.zeros((nsampes+1))
        u[0] = self.u; v[0] = self.v; h[0] = self.h; time[0] = self.time

        # phi_n at {b-2, b-1, b}
        h_prev = np.zeros((self.nk, self.nj, 3))  # [OBC - Orlanski]
        u_prev = np.zeros((self.nk, self.nj, 3))  # [OBC - Orlanski]
        v_prev = np.zeros((self.nk, self.nj, 3))  # [OBC - Orlanski]

        rx_h = np.zeros(2); rx_u = np.zeros(2); rx_v = np.zeros(2)  # [OBC - Orlanski] phase-speed diags at b-1,b
        h_diff = np.zeros(2); u_diff = np.zeros(2); v_diff = np.zeros(2)  # [OBC - Orlanski] diffs for diagnostics
        rxh = np.zeros((nsteps, 2)); rxu = np.zeros((nsteps, 2)); rxv = np.zeros((nsteps, 2))  # [OBC - Orlanski] phase speeds for diagnostics

        nsamp = 0
        nrun = nsteps
        for it in range(1, nsteps + 1):
            # snap shot phi^n before the step
            h_prev[:] = self.h[:, :, b-2:b+1]  
            u_prev[:] = self.u[:, :, b-2:b+1] 
            v_prev[:] = self.v[:, :, b-2:b+1]

            # advance
            # To turn off prescribing, set the second argument to 0. 
            self._step_core(dt, 2, prev_cols, 
                            h_bc_all[it - 1], u_bc_all[it - 1], v_bc_all[it - 1],
                            h_diff, u_diff, v_diff,
                            obc_on = 1, b_obc = b,
                            h_prev = h_prev, u_prev = u_prev, v_prev = v_prev,
                            rx_h = rx_h, rx_u = rx_u, rx_v = rx_v)
            rxh[it-1] = rx_h; rxu[it-1] = rx_u; rxv[it-1] = rx_v  # [OBC - Orlanski] store phase speeds for diag

            if np.any(np.isnan(self.u)):
                print('Model has blown up!!! Stopping early')
                u = u[:nsamp]; v = v[:nsamp]; h = h[:nsamp]; time = time[:nsamp]
                nrun = it  # [OBC-E] steps attempted
                break
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
                'v_mean': rxv[:nrun, 0], 'vmax': rxv[:nrun, 1]}
        print(f"Mean r_x over run is h = {rx['h_mean'].mean():.3f}"
                f"u={rx['u_mean'].mean():.3f}  v={rx['v_mean'].mean():.3f}")
            
        return u, v, h, time, diffs, rx
        
    # [OBC] single _step_numba call site, shared by step() (bc_mode=0) and the
    # [OBC] boundary record/replace runs. Advances time/iter after the JIT step.
    def _step_core(self, dt, bc_mode, bc_cols,\
                   # [OBC] setting
                   h_bc, u_bc, v_bc, h_diff, u_diff, v_diff,
                   # [OBC - Orlanski] Settings
                   obc_on = 0, b_obc = 0, 
                   h_prev = None, u_prev = None, v_prev = None,
                   rx_h = None, rx_u = None, rx_v = None):
        # Sponge gates are cached by the h/u/v_relax setters (a bool keeps the
        # JIT argument types stable and lets numba skip a term when its sponge
        # is off), so no per-step recomputation is needed.

        # [OBC - Orlanski]
        if h_prev is None : h_prev = self._obc_prev_dummy
        if u_prev is None : u_prev = self._obc_prev_dummy
        if v_prev is None : v_prev = self._obc_prev_dummy
        if rx_h is None : rx_h = self._diff_dummy
        if rx_u is None : rx_u = self._diff_dummy
        if rx_v is None : rx_v = self._diff_dummy

        _step_numba(self.u, self.v, self.h, self.D, self.taux, self.tauy,
                    self.f, self.f_at_u, self.f_at_v,
                    dt, self.dx, self.dy, self.g, self.epsilon, self.nu_h, self.nu_v,
                    self.alpha_f, self.alpha_nu,
                    self.h_zonal_target, self.h_target, self.u_target, self.v_target,
                    self.h_zonal_relax, self._h_relax, self._u_relax, self._v_relax,
                    self._h_relax_on, self._u_relax_on, self._v_relax_on,
                    self.hsub, self.iter,
                    bc_mode, bc_cols, h_bc, u_bc, v_bc, h_diff, u_diff, v_diff, #  [OBC]
                    obc_on, int(b_obc), h_prev, u_prev, v_prev, rx_h, rx_u, rx_v)  # [OBC - Orlanski]
        self.time += dt
        self.iter += 1

    def step(self, dt):
        """
        dt   - Time step [s]
        """
        # [OBC] bc_mode=0 + cached dummies -> the boundary blocks are skipped, so
        # [OBC] this is bit-identical to the pre-OBC step.
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
