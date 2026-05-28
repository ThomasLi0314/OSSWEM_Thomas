import numpy as np
from numba import njit, prange


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
def _nb_v2q(a):
    """Averages from v- to q- points."""
    return 0.5 * ( a + _nb_im1(a) )

@njit(cache=True)
def _nb_u2h(a):
    """Averages from u- to h- points. Also does q- to v-."""
    return 0.5 * ( a + _nb_ip1(a) )

@njit(cache=True)
def _nb_q2v(a):
    """Averages from q- to v- points."""
    return 0.5 * ( a + _nb_ip1(a) )

@njit(cache=True)
def _nb_h2v(a):
    """Averages from h- to v- points. Also does u- to q-."""
    return 0.5 * ( a + _nb_jm1(a) )

@njit(cache=True)
def _nb_u2q(a):
    """Averages from u- to q- points."""
    return 0.5 * ( a + _nb_jm1(a) )

@njit(cache=True)
def _nb_v2h(a):
    """Averages from v- to h- points. Also does q- to u-."""
    return 0.5 * ( a + _nb_jp1(a) )

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
def _nb_diu(a):
    """Difference u- points to h- points. Also does q- to v-."""
    return _nb_ip1(a) - a

@njit(cache=True)
def _nb_djh(a):
    """Difference h- points to v- points. Also does u- to q-."""
    return a - _nb_jm1(a)

@njit(cache=True)
def _nb_djv(a):
    """Difference v- points to h- points. Also does q- to u-."""
    return _nb_jp1(a) - a

# --- Numba-JIT minimum helpers ---

@njit(cache=True)
def _nb_minh2u(a):
    """Minimum from h- to u- points. Also does v- to q-."""
    return np.minimum( a, _nb_im1(a) )

@njit(cache=True)
def _nb_minh2v(a):
    """Minimum from h- to v- points. Also does u- to q-."""
    return np.minimum( a, _nb_jm1(a) )

# --- Numba-JIT physical functions ---

@njit(cache=True)
def _nb_vxuy(u, v, rdx, rdy):
    """Calculate components of relative vorticity"""
    return _nb_dih( v ) * rdx, _nb_djh( u ) * rdy

# --- Numba-JIT step function ---

@njit(cache=True)
def _step_numba(u, v, h, D, taux, tauy, f, f_at_u, f_at_v,
                dt, dx, dy, g, epsilon, nu, nu_v, alpha_f, alpha_nu,
                h_target, u_target, v_target,
                h_relax, u_relax, v_relax, u_relax_on, v_relax_on, hsub, iter_num):
    """JIT-compiled step function. Modifies u, v, h in-place. Layer thickness h
    is the prognostic; eta = h - D is diagnosed where needed (pressure gradient).
    State arrays u, v, h have shape (nk, nj, ni). g is a length-nk vector.
    h_target has shape (nk, nj) and gives the target zonal-mean thickness per
    layer per row. h_relax is a scalar; restoring acts on layer 0 only. Inverse
    layer thicknesses are taken from the prognostic h (with +hsub to guard
    division), not from any nominal/reference thickness.
    Interior vertical viscosity (nu_v) and bottom drag (epsilon) together form a
    per-column tridiagonal vertical-diffusion operator L; both are time-weighted
    by alpha_nu (1 = Euler backward, 0 = explicit). The implicit step solves the
    K-element coupled tridiagonal+Coriolis system per (i,j) via Hallberg's
    cancellation-free TDMAH2 recurrence: rows are first scaled by their layer
    thickness h_k to symmetrize the matrix (removing all 1/h_k factors) and to
    bundle the Coriolis term as h_k(1+ic). The forward sweep maintains running
    ratios q_k = a_{k+1/2}*beta and Q_k = (h_k(1+ic)+a_{k-1/2}*Q_{k-1})*beta
    with beta_k = 1/(h_k(1+ic) + a_{k-1/2}*Q_{k-1} + a_{k+1/2}); every
    denominator is a sum of nonnegative-real plus complex terms with positive
    real part, so the algorithm is robust as h_k -> 0. Re(Δw) gives Δu at
    u-points; Im(Δw) gives Δv at v-points."""
    nk, nj, ni = u.shape

    rdx = 1 / dx
    rdy = 1 / dy

    # Restoring on layer 0: relax zonal-mean h[0] toward h_target[0, :].
    if h_relax > 0:
        h_dev = h[0].sum(axis=-1) / ni - h_target[0, :]
        h[0] -= ( dt * h_relax ) * h_dev.reshape(nj, 1)

    # Restoring of velocities toward prescribed target patterns (full field,
    # all layers). Explicit (operator-split) Euler step applied before the
    # dynamics so the rest of the step sees the relaxed u, v. Stable while
    # dt*u_relax <= 1. u_relax, v_relax, u_target, v_target all have the same
    # shape as u, v; the u_relax_on / v_relax_on booleans (precomputed on the
    # Python side) gate the term so it is skipped entirely when restoring is
    # off, and avoid an ambiguous truth test on the array rate.
    if u_relax_on:
        u -= ( dt * u_relax ) * ( u - u_target )
    if v_relax_on:
        v -= ( dt * v_relax ) * ( v - v_target )

    # Cache upwind-signed velocities (u,v are unchanged until end of step)
    u_pos = np.maximum( u, 0.0 )
    u_neg = np.minimum( u, 0.0 )
    v_pos = np.maximum( v, 0.0 )
    v_neg = np.minimum( v, 0.0 )

    # Pre-continuity hq (used for PV in the momentum step below)
    hq = _nb_u2q( _nb_h2u( h ) )

    # Continuity: integrate h directly via alternating directional split
    if iter_num % 2 == 0:
        hu = u_pos * _nb_im1( h ) + u_neg * h # Upwinded h*u on western edge
        h -= ( dt * rdx ) * _nb_diu( hu )
        hv = v_pos * _nb_jm1( h ) + v_neg * h # Upwinded h*v on southern edge
        h -= ( dt * rdy ) * _nb_djv( hv )
    else:
        hv = v_pos * _nb_jm1( h ) + v_neg * h # Upwinded h*v on southern edge
        h -= ( dt * rdy ) * _nb_djv( hv )
        hu = u_pos * _nb_im1( h ) + u_neg * h # Upwinded h*u on western edge
        h -= ( dt * rdx ) * _nb_diu( hu )

    # Explicit accelerations
    uip1_neg = _nb_ip1( u_neg )
    vjp1_neg = _nb_jp1( v_neg )
    # Enquist-Oscher 1/2 ( u^2 + v^2 )
    K = 0.5 * ( u_pos**2 + uip1_neg**2 )
    K += 0.5 * ( v_pos**2 + vjp1_neg**2 )
    # Interface positions eta[k] = -D + sum_{l=k}^{nk-1} h[l]  (cumulative from bottom)
    eta = np.empty_like(h)
    eta[nk-1] = h[nk-1] - D
    for k in range(nk-2, -1, -1):
        eta[k] = eta[k+1] + h[k]
    # Montgomery potential M[k] = sum_{l=0}^{k} g[l] * eta[l]  (cumulative from top)
    M = np.empty_like(h)
    M[0] = g[0] * eta[0]
    for k in range(1, nk):
        M[k] = M[k-1] + g[k] * eta[k]
    B = M + K # Bernoulli = potential + KE

    # Gradient of Bernoulli
    Bx = _nb_dih( B ) * rdx
    By = _nb_djh( B ) * rdy

    # Components of relative vorticity and stress tensor
    # vx = _nb_dih( v ) * rdx
    # uy = _nb_djh( u ) * rdy
    vx, uy = _nb_vxuy(u, v, rdx, rdy)
    vy = _nb_djv( v ) * rdy
    ux = _nb_diu( u ) * rdx

    # Potential vorticity
    q = f + ( vx - uy )
    recip_hq_plus_hsub = 1.0 / ( hq + hsub )
    q *= recip_hq_plus_hsub
    q *= ( hq * recip_hq_plus_hsub ) # Hack to mask q
    # q * h at u- and v-points
    qhv = _nb_q2u( q * _nb_v2q( hv ) )
    qhu = _nb_q2v( q * _nb_u2q( hu ) )
    # For the stress tensor
    D_tension = ux - vy
    D_shear = uy + vx
    # Use latest h here, but not in q (still using pre-continuity hq above)
    # h at q points
    hq = _nb_minh2v( _nb_minh2u( h ) )
    # Components of stress tensor
    nu_h_Dt = nu * h * D_tension
    nu_hq_Ds = nu * hq * D_shear
    # h at u- and v-points
    rhu = 1.0 / ( _nb_h2u( h ) + hsub )
    rhv = 1.0 / ( _nb_h2v( h ) + hsub )
    # Divergence of stress tensor
    uxxyy = _nb_dih( nu_h_Dt ) * rdx + _nb_djv( nu_hq_Ds ) * rdy
    uxxyy = uxxyy * rhu
    vxxyy = _nb_diu( nu_hq_Ds ) * rdx - _nb_djh( nu_h_Dt ) * rdy
    vxxyy = vxxyy * rhv

    # Wind forcing on top layer (explicit).
    udot =   ( qhv - Bx ) + uxxyy
    vdot = - ( qhu + By ) + vxxyy
    udot[0,:,:] += taux * rhu[0,:,:]
    vdot[0,:,:] += tauy * rhv[0,:,:]

    # Interfacial-stress coefficients at u- and v-points (a_{k-1/2} = a_top, a_{k+1/2} = a_bot).
    # Top:    a_top[0]    = 0 (wind is the explicit forcing applied above).
    # Bottom: a_bot[nk-1] = epsilon (bottom drag).
    # Interior: a_{k-1/2} = 2*nu_v/(h_{k-1}+h_k) appears as a_top[k] AND a_bot[k-1].
    h_at_u = _nb_h2u( h )
    h_at_v = _nb_h2v( h )
    a_top_u = np.zeros((nk, nj, ni))
    a_bot_u = np.zeros((nk, nj, ni))
    a_top_v = np.zeros((nk, nj, ni))
    a_bot_v = np.zeros((nk, nj, ni))
    for k in range(1, nk):
        a_int_u = 2.0 * nu_v / ( h_at_u[k-1] + h_at_u[k] + hsub )
        a_top_u[k]   = a_int_u
        a_bot_u[k-1] = a_int_u
        a_int_v = 2.0 * nu_v / ( h_at_v[k-1] + h_at_v[k] + hsub )
        a_top_v[k]   = a_int_v
        a_bot_v[k-1] = a_int_v
    a_bot_u[nk-1] = epsilon
    a_bot_v[nk-1] = epsilon

    # Explicit -(L u^n), -(L v^n): (L u)_k = ((a_top+a_bot) u_k - a_top u_{k-1} - a_bot u_{k+1}) / h_k.
    # Vectorized over (j,i); k-loop only.
    for k in range(nk):
        Lu_k = (a_top_u[k] + a_bot_u[k]) * u[k]
        Lv_k = (a_top_v[k] + a_bot_v[k]) * v[k]
        if k > 0:
            Lu_k -= a_top_u[k] * u[k-1]
            Lv_k -= a_top_v[k] * v[k-1]
        if k < nk - 1:
            Lu_k -= a_bot_u[k] * u[k+1]
            Lv_k -= a_bot_v[k] * v[k+1]
        udot[k] -= Lu_k * rhu[k]
        vdot[k] -= Lv_k * rhv[k]

    # Implicit step: TDMAH2 (cancellation-free symmetric Thomas, Hallberg).
    # Row-scale the system by h_k to symmetrize: the matrix becomes diag
    # h_k(1+ic) + a^*_{k-1/2} + a^*_{k+1/2} with off-diagonals -a^*_{k±1/2},
    # where a^*_{k±1/2} = alpha_nu*dt*a_{k±1/2} (real, ≥0) and ic = alpha_f*dt*f.
    # RHS is h_k * dt*(udot + i vdot_at_u). Forward sweep keeps running q_k and
    # Q_k built from sums of nonneg-real plus complex-with-positive-real-part —
    # no catastrophic cancellation; bounded as h_k -> 0.
    vdot_at_u = _nb_q2u( _nb_v2q( vdot ) )
    udot_at_v = _nb_q2v( _nb_u2q( udot ) )

    q       = np.empty((nk, nj, ni), dtype=np.complex128)
    y_prime = np.empty((nk, nj, ni), dtype=np.complex128)

    # u-point pass: take Re(Δw) for Δu.
    ic_u = alpha_f * dt * f_at_u   # 2D real (nj, ni)
    hc   = h_at_u[0] * ( 1.0 + 1j * ic_u )
    a_t  = ( alpha_nu * dt ) * a_top_u[0]   # = 0 at the surface
    a_b  = ( alpha_nu * dt ) * a_bot_u[0]
    beta = 1.0 / ( hc + a_t + a_b )
    q[0] = a_b * beta
    Q    = hc * beta
    y_prime[0] = h_at_u[0] * ( dt * udot[0] + 1j * ( dt * vdot_at_u[0] ) ) * beta
    for k in range(1, nk):
        a_t  = ( alpha_nu * dt ) * a_top_u[k]
        a_b  = ( alpha_nu * dt ) * a_bot_u[k]
        hc   = h_at_u[k] * ( 1.0 + 1j * ic_u )
        beta = 1.0 / ( hc + a_t * Q + a_b )
        q[k] = a_b * beta
        Q    = ( hc + a_t * Q ) * beta
        y_k  = h_at_u[k] * ( dt * udot[k] + 1j * ( dt * vdot_at_u[k] ) )
        y_prime[k] = ( y_k + a_t * y_prime[k-1] ) * beta
    delta_w = y_prime[nk-1]
    u[nk-1] += delta_w.real
    for k in range(nk-2, -1, -1):
        delta_w = y_prime[k] + q[k] * delta_w
        u[k] += delta_w.real

    # v-point pass: take Im(Δw) for Δv.
    ic_v = alpha_f * dt * f_at_v
    hc   = h_at_v[0] * ( 1.0 + 1j * ic_v )
    a_t  = ( alpha_nu * dt ) * a_top_v[0]
    a_b  = ( alpha_nu * dt ) * a_bot_v[0]
    beta = 1.0 / ( hc + a_t + a_b )
    q[0] = a_b * beta
    Q    = hc * beta
    y_prime[0] = h_at_v[0] * ( dt * udot_at_v[0] + 1j * ( dt * vdot[0] ) ) * beta
    for k in range(1, nk):
        a_t  = ( alpha_nu * dt ) * a_top_v[k]
        a_b  = ( alpha_nu * dt ) * a_bot_v[k]
        hc   = h_at_v[k] * ( 1.0 + 1j * ic_v )
        beta = 1.0 / ( hc + a_t * Q + a_b )
        q[k] = a_b * beta
        Q    = ( hc + a_t * Q ) * beta
        y_k  = h_at_v[k] * ( dt * udot_at_v[k] + 1j * ( dt * vdot[k] ) )
        y_prime[k] = ( y_k + a_t * y_prime[k-1] ) * beta
    delta_w = y_prime[nk-1]
    v[nk-1] += delta_w.imag
    for k in range(nk-2, -1, -1):
        delta_w = y_prime[k] + q[k] * delta_w
        v[k] += delta_w.imag


def _step_fused_impl(u, v, h, D, taux, tauy, f, f_at_u, f_at_v,
                     dt, dx, dy, g, epsilon, nu, nu_v, alpha_f, alpha_nu,
                     h_target, u_target, v_target,
                     h_relax, u_relax, v_relax, u_relax_on, v_relax_on, hsub, iter_num):
    """Fused implementation of the time step (EXPERIMENTAL). Shared body for
    the serial (_step_numba_fused) and threaded (_step_numba_fused_par)
    kernels: njit-compiled with parallel=False the prange loops run serially,
    with parallel=True they run across threads (parallelized over j-rows,
    which are independent so the loops are race-free).

    Same algorithm as _step_numba, but the helper-based array operations are
    replaced by fused loops that recompute stencil quantities inline,
    eliminating their temporary allocations: pre-continuity hq, continuity,
    kinetic energy + Bernoulli + h-at-u/v reciprocals, the explicit momentum
    accelerations (PV Coriolis fluxes, Bernoulli gradient, viscous stress
    divergence), the TDMAH2 input averaging, and the solver tail (interfacial
    coefficients computed inline as scalars; the cancellation-free TDMAH2
    recurrence run per column with scalar locals). Per-cell expressions keep
    the reference's arithmetic grouping, so results agree to roundoff."""
    nk, nj, ni = u.shape
    rdx = 1 / dx
    rdy = 1 / dy

    # --- restoring (identical to reference) ---
    if h_relax > 0:
        h_dev = h[0].sum(axis=-1) / ni - h_target[0, :]
        h[0] -= ( dt * h_relax ) * h_dev.reshape(nj, 1)
    if u_relax_on:
        u -= ( dt * u_relax ) * ( u - u_target )
    if v_relax_on:
        v -= ( dt * v_relax ) * ( v - v_target )

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
                vort_jp = f[jp, i] + ( v[k,jp,i] - v[k,jp,im] ) * rdx - ( u[k,jp,i] - u[k,j,i]   ) * rdy
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
                            + qpv_jp * 0.5 * ( hv[k,jp,i] + hv[k,jp,im] ) )
                qhu = 0.5 * ( qpv_c  * 0.5 * ( hu[k,j,i]  + hu[k,jm,i]  )
                            + qpv_ip * 0.5 * ( hu[k,j,ip] + hu[k,jm,ip] ) )

                # Stress tensor: nu*h*D_tension at h-points {(j,i),(j,im),(jm,i)};
                # nu*hq*D_shear at q-points {(j,i),(jp,i),(j,ip)}.
                Dt_c  = ( u[k,j,ip] - u[k,j,i]  ) * rdx - ( v[k,jp,i] - v[k,j,i]  ) * rdy
                nuhDt_c  = nu * h[k,j,i]  * Dt_c
                Dt_im = ( u[k,j,i]  - u[k,j,im] ) * rdx - ( v[k,jp,im]- v[k,j,im] ) * rdy
                nuhDt_im = nu * h[k,j,im] * Dt_im
                Dt_jm = ( u[k,jm,ip]- u[k,jm,i] ) * rdx - ( v[k,j,i]  - v[k,jm,i] ) * rdy
                nuhDt_jm = nu * h[k,jm,i] * Dt_jm

                Ds_c  = ( u[k,j,i]  - u[k,jm,i] ) * rdy + ( v[k,j,i]  - v[k,j,im]  ) * rdx
                hqp_c  = min( min( h[k,j,i],  h[k,j,im]  ), min( h[k,jm,i], h[k,jm,im] ) )
                nuhqDs_c  = nu * hqp_c  * Ds_c
                Ds_jp = ( u[k,jp,i] - u[k,j,i]  ) * rdy + ( v[k,jp,i] - v[k,jp,im] ) * rdx
                hqp_jp = min( min( h[k,jp,i], h[k,jp,im] ), min( h[k,j,i],  h[k,j,im]  ) )
                nuhqDs_jp = nu * hqp_jp * Ds_jp
                Ds_ip = ( u[k,j,ip] - u[k,jm,ip]) * rdy + ( v[k,j,ip] - v[k,j,i]   ) * rdx
                hqp_ip = min( min( h[k,j,ip], h[k,j,i]   ), min( h[k,jm,ip],h[k,jm,i]  ) )
                nuhqDs_ip = nu * hqp_ip * Ds_ip

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


# Two compilations of the shared fused body. Serial: prange acts as range.
# Parallel: prange loops over j-rows run across numba threads (set the count
# with numba.set_num_threads(); this is a memory-bandwidth-bound stencil, so
# the sweet spot is the physical core count, not the SMT thread count).
# parallel=True is left uncached because two Dispatchers compiled from one
# py_func would otherwise share a cache key.
_step_numba_fused     = njit(cache=True)(_step_fused_impl)
_step_numba_fused_par = njit(parallel=True)(_step_fused_impl)


class SSWEM:
    """(S)tacked (S)hallow (W)ater (E)quation (M)odel"""

    def __init__(self, ni, g, Ho, Lx, fo, beta, epsilon, nu, nu_v=0,
                 h_relax=0, u_relax=None, v_relax=None, u_target=None, v_target=None,
                 hsub=1e-12, fused=False):
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
        nu      - Lateral (horizontal) viscosity [m2 s-1]
        nu_v    - Vertical viscosity [m2 s-1]; sets interior interfacial-stress
                  coefficients a_{k-1/2} = 2*nu_v/(h_{k-1}+h_k) for 1<k<=K. Defaults to 0.
        h_relax - Restoring rate for zonal-mean surface eta [s-1] (scalar).
        u_relax - Restoring rate for u toward u_target [s-1]. None (default) for
                  no restoring; otherwise a scalar or any array broadcastable to
                  (nk, nj, ni) for a spatially varying rate.
        v_relax - Restoring rate for v toward v_target [s-1]. None (default) for
                  no restoring; otherwise a scalar or any array broadcastable to
                  (nk, nj, ni) for a spatially varying rate.
        u_target - Target zonal velocity field [m s-1] relaxed to when u_relax>0.
                  Scalar or any array broadcastable to (nk, nj, ni). Defaults to 0.
        v_target - Target meridional velocity field [m s-1] relaxed to when v_relax>0.
                  Scalar or any array broadcastable to (nk, nj, ni). Defaults to 0.
        hsub    - H sub-roundoff [m]
        fused   - EXPERIMENTAL time-step kernel selector (default False).
                  False      -> reference vectorized kernel (_step_numba).
                  True       -> serial fused single-loop kernel.
                  'parallel' -> threaded fused kernel (prange over j-rows).
                  The fused kernels agree with the reference to roundoff; the
                  parallel one scales best at the physical core count (set via
                  numba.set_num_threads()).
        """
        self.ni = ni
        self.g = np.atleast_1d(np.asarray(g, dtype=float)).copy()
        self.Ho = np.atleast_1d(np.asarray(Ho, dtype=float)).copy()
        self.nk = self.g.size
        if self.Ho.size != self.nk:
            raise ValueError(f"Ho must have length nk={self.nk}, got {self.Ho.size}")
        self.h_relax = float(h_relax)
        # EXPERIMENTAL kernel selection: False -> reference _step_numba;
        # True -> serial fused; 'parallel' -> threaded fused (prange over
        # j-rows; set thread count with numba.set_num_threads()).
        if fused not in (False, True, 'parallel'):
            raise ValueError("fused must be False, True, or 'parallel'")
        self.fused = fused
        self.Lx = Lx
        self.fo = fo
        self.beta = beta
        self.epsilon = epsilon
        self.nu = nu
        self.nu_v = float(nu_v)
        self.hsub = hsub
        self.alpha_f = 0.5 # Crank-Nicholson for Coriolis
        self.alpha_nu = 1.0 # Euler backward for interfacial stresses (vertical viscosity + bottom drag)

        # Grid resolution
        self.dx = Lx / ni # Cell width [m]

        # Limit to square domains/grids for now
        self.nj = self.ni
        self.Ly = self.Lx
        self.dy = self.dx

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
        print("Grid: dx =",self.dx,"[m]")

        # Resting initial conditions, no (flat) bathymetry and no forcing.
        # flat_topog must precede resting_state since rest h = D.
        self.flat_topog()
        self.resting_state()
        self.zero_forcing()
        # Default h_target = rest layer thickness per row (no perturbation)
        self.h_target = np.tile(self.Ho[:, None], (1, self.nj)).astype(float)

        # Velocity restoring targets and rates (full field, all layers). Accept
        # anything broadcastable to (nk, nj, ni): scalar, (ni,), (nj,ni),
        # (nk,nj,ni), ... Rates default to None (no restoring) -> a zero array,
        # so the type passed to the JIT step is always a float64 (nk,nj,ni)
        # array; the actual on/off decision is made by a boolean gate in step().
        shape = (self.nk, self.nj, self.ni)
        u_t = 0.0 if u_target is None else u_target
        v_t = 0.0 if v_target is None else v_target
        self.u_target = np.broadcast_to(np.asarray(u_t, dtype=float), shape).copy()
        self.v_target = np.broadcast_to(np.asarray(v_t, dtype=float), shape).copy()
        # Rates are routed through their property setters (below), which coerce
        # to a (nk,nj,ni) float64 array and cache the on/off gate once, so the
        # per-step .max() recomputation is avoided.
        self.u_relax = u_relax
        self.v_relax = v_relax

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
        """Sets the meridional restoring profile for layer k. The zonal-mean
        h[k] is restored toward h_target[k, j] = Ho[k] + mag * profile(yh1[j]/Ly).
        Default k=0 (top layer); pass k explicitly to set a different layer.
        Note: only layer 0's target is currently used (h_relax is scalar and
        the restoring branch in _step_numba acts on layer 0 only)."""
        if k < 0 or k >= self.nk:
            raise ValueError(f"k must be in [0,{self.nk-1}], got {k}")
        profile = ( SSWEM._cubint( self.yh1 / self.Ly, 0.0, 0.1 ) -
                    SSWEM._cubint( self.yh1 / self.Ly, 0.5, 0.6 ) )
        self.h_target[k, :] = self.Ho[k] + mag * profile

    def set_u_target_jet(self, mag):
        """Sets the u restoring target to a meridional jet profile. mag is the
        per-layer jet amplitude [m s-1]: a scalar (same amplitude in every
        layer) or a length-nk vector (one amplitude per layer)."""
        mag = np.atleast_1d(np.asarray(mag, dtype=float))
        if mag.size == 1:
            mag = np.full(self.nk, mag[0])
        elif mag.size != self.nk:
            raise ValueError(f"mag must be a scalar or length nk={self.nk}, got {mag.size}")
        profile = ( SSWEM._cubint( self.yu / self.Ly, 0.25, 0.5 ) -
                    SSWEM._cubint( self.yu / self.Ly, 0.5, 0.75 ) )
        self.u_target = np.empty_like(self.u)
        for k in range(0,self.nk):
            self.u_target[k, :] = mag[k] * profile

    def run(self, dt, samp, nsamps):
        """
        dt     - Time step [s]
        samp   - Steps between samples [steps]
        nsamps - Number of sample to integrate model [steps*samp]
        """
        print("CFL: dt*epsilon/h_bot =", dt * self.epsilon / self.Ho[-1] )
        print("CFL: dt*nu_v/h_min^2 =", dt * self.nu_v / ( self.Ho.min()**2 ) )
        print("CFL: dt*f =", dt * np.abs( self.f.max() ) )
        print("CFL: dt*cg/dx =", dt * self.cg / self.dx )
        if self.cg1 is not None:
            print("CFL: dt*cg1/dx =", dt * self.cg1 / self.dx )
        print("CFL: dt*nu/dx^2 =", dt * self.nu / self.dx**2 )
        if self.h_relax > 0:
            print("CFL: dt*h_relax =", dt * self.h_relax )
        if self.u_relax.max() > 0:
            print("CFL: dt*u_relax =", dt * self.u_relax.max() )
        if self.v_relax.max() > 0:
            print("CFL: dt*v_relax =", dt * self.v_relax.max() )
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
                break
            if iter % samp == 0:
                nsamp += 1
                u[nsamp] = self.u
                v[nsamp] = self.v
                h[nsamp] = self.h
                time[nsamp] = self.time
        print("...done")
        return u, v, h, time

    def step(self, dt):
        """
        dt   - Time step [s]
        """
        # Velocity-restoring gates are cached by the u_relax/v_relax setters
        # (a bool keeps the JIT argument types stable and lets numba skip the
        # term when restoring is off), so no per-step recomputation is needed.
        if self.fused == 'parallel':
            kernel = _step_numba_fused_par
        elif self.fused:
            kernel = _step_numba_fused
        else:
            kernel = _step_numba
        kernel(self.u, self.v, self.h, self.D, self.taux, self.tauy,
               self.f, self.f_at_u, self.f_at_v,
               dt, self.dx, self.dy, self.g, self.epsilon, self.nu, self.nu_v,
               self.alpha_f, self.alpha_nu,
               self.h_target, self.u_target, self.v_target,
               self.h_relax, self._u_relax, self._v_relax,
               self._u_relax_on, self._v_relax_on, self.hsub, self.iter)
        self.time += dt
        self.iter += 1

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
