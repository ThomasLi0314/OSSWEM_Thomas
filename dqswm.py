import numpy as np


class DQSWE:

    def __init__(self, ni, g, Do, Lx, fo, beta, epsilon, nu, h_relax=0, hsub=1e-12):
        """
        ni      - Number of cells in i-direction
        g       - Gravity [m s-2]
        Do      - Max depth [m]
        Lx      - Domain width [m]
        fo      - Coriolis [s-1]
        beta    - df/dy [m-1 s-1]
        epsilon - Drag rate [m-1 s-1]
        nu      - Lateral viscosity [m s-2]
        h_relax - Relaxation rate of zonal average eta to profile [s-1]
        hsub    - H sub-roundoff [m]
        """
        self.ni = ni
        self.g = g
        self.Do = Do
        self.Lx = Lx
        self.fo = fo
        self.beta = beta
        self.epsilon = epsilon
        self.nu = nu
        self.h_relax = h_relax
        self.hsub = hsub
        self.alpha_f = 0.5 # Crank-Nicholson for Coriolis
        self.alpha_e = 1.0 # Euler backward for dissipation

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

        # Resting initial conditions, no (flat) bathymetry and no forcing
        self.resting_state()
        self.flat_topog()
        self.zero_forcing()
        self.set_eta_forcing(0)

        # Derived parameters
        self.f = self.fo + self.beta * self.yq # Coriolis is at q-points
        self.cg = np.sqrt( self.g * self.Do )
        if not self.fo==0:
            self.Ld = self.cg / self.fo
        else: self.Ld = None
        print("cg =", self.cg, "[m s-1]")
        print("Ld =", self.Ld, "[m]")
        if not self.beta==0:
            self.Ls = self.epsilon / ( self.beta * self.Do )
        else: self.Ls = None
        print("Scales: Ls=epsilon/D/beta =", self.Ls, "[m]")
        if self.Ld is not None and self.Ld>0:
            print("Scales: Lx/Ld =", self.Lx / self.Ld)
        if not self.fo==0:
            print("Res: Ld/dx =",self.Ld / self.dx)
        if not self.beta==0:
            print("Res: Ls/dx =",self.Ls / self.dx)

    def resting_state(self):
        """Set state to resting (u=v=eta=0)"""
        self.u = np.zeros((self.nj,self.ni))
        self.v = np.zeros((self.nj,self.ni))
        self.eta = np.zeros((self.nj,self.ni))
        self.time = 0
        self.iter = 0

    def flat_topog(self):
        """Set bathymetry to flat with no boundaries"""
        self.D = self.Do + 0 * self.xh

    def bowl_topog(self):
        """Set bathymetry to bowl shape"""
        self.D = self.Do * np.sin( self.xh * np.pi / self.Lx ) * np.sin( self.yh * np.pi / self.Ly )
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

    def perturb_eta(self, eta_mag, L, x0, y0=None):
        """Adds a Gaussion perturbation to eta, centered at x0,y0, with magnitude eta_mag
        length scale L"""
        r2 = ( ( self.xh - x0 ) / L )**2
        if y0 is not None:
            r2 = r2 + ( ( self.yh - y0 ) / L )**2
        self.eta = self.eta + eta_mag * np.exp( - 0.5 * r2 )

    def set_eta_forcing(self, eta_mag):
        """Sets the forcing profile to which the zonal averagin eta is restored"""
        self.eta_target = eta_mag * np.cos( 2 * np.pi * np.mean( self.yh, axis=1, keepdims=True ) / self.Ly)

    def run(self, dt, samp, nsamps):
        """
        dt     - Time step [s]
        samp   - Steps between samples [steps]
        nsamps - Number of sample to integrate model [steps*samp]
        """
        print("CFL: dt*epsilon/D =", dt * self.epsilon / self.Do )
        print("CFL: dt*f =", dt * np.abs( self.f.max() ) )
        print("CFL: dt*cg/dx =", dt * self.cg / self.dx )
        print("CFL: dt*nu/dx^2 =", dt * self.nu / self.dx**2 )
        if self.h_relax>0:
            print("CFL: dt*h_relax =", dt * self.h_relax )
        nsteps = nsamps * samp
        print("nsteps =", nsteps)
        Trun = nsteps * dt
        print("Time: Trun * fo =", Trun * self.fo)
        print("Time: Trun * ( cg / L ) =", Trun * self.cg / self. Lx)
        print("Running...")

        # Pre-allocatge diagnostics to be returned from run()
        u = np.zeros((nsamps+1, self.nj, self.ni))
        v = np.zeros((nsamps+1, self.nj, self.ni))
        eta = np.zeros((nsamps+1, self.nj, self.ni))
        time = np.zeros((nsamps+1))

        u[0] = self.u.copy()
        v[0] = self.v.copy()
        eta[0] = self.eta.copy()
        time[0] = self.time

        nsamp = 0
        for iter in range( 1, nsteps + 1 ):
            self.step( dt )
            if iter % samp ==0:
                nsamp += 1
                u[nsamp] = self.u.copy()
                v[nsamp] = self.v.copy()
                eta[nsamp] = self.eta.copy()
                time[nsamp] = self.time
        print("...done")
        return u, v, eta, time

    def _h2u(a):
        """Averages from h- to u- points. Also does v- to q-."""
        return 0.5 * ( a + np.concatenate((a[:, -1:], a[:, :-1]), axis=1) )
    def _v2q(a):
        """Averages from v- to q- points."""
        return 0.5 * ( a + np.concatenate((a[:, -1:], a[:, :-1]), axis=1) )
    def _u2h(a):
        """Averages from u- to h- points. Also does q- to v-."""
        return 0.5 * ( a + np.concatenate((a[:, 1:], a[:, :1]), axis=1) )
    def _q2v(a):
        """Averages from q- to v- points."""
        return 0.5 * ( a + np.concatenate((a[:, 1:], a[:, :1]), axis=1) )
    def _h2v(a):
        """Averages from h- to v- points. Also does u- to q-."""
        return 0.5 * ( a + np.concatenate((a[-1:, :], a[:-1, :]), axis=0) )
    def _u2q(a):
        """Averages from u- to q- points."""
        return 0.5 * ( a + np.concatenate((a[-1:, :], a[:-1, :]), axis=0) )
    def _v2h(a):
        """Averages from v- to h- points. Also does q- to u-."""
        return 0.5 * ( a + np.concatenate((a[1:, :], a[:1, :]), axis=0) )
    def _q2u(a):
        """Averages from q- to u- points."""
        return 0.5 * ( a + np.concatenate((a[1:, :], a[:1, :]), axis=0) )
    def _dih(a):
        """Difference h- points to u- points. Also does v- to q-."""
        return a - np.concatenate((a[:, -1:], a[:, :-1]), axis=1)
    def _diu(a):
        """Difference u- points to h- points. Also does q- to v-."""
        return np.concatenate((a[:, 1:], a[:, :1]), axis=1) - a
    def _djh(a):
        """Difference h- points to v- points. Also does u- to q-."""
        return a - np.concatenate((a[-1:, :], a[:-1, :]), axis=0)
    def _djv(a):
        """Difference v- points to h- points. Also does q- to u-."""
        return np.concatenate((a[1:, :], a[:1, :]), axis=0) - a
    def _minh2u(a):
        """Minimum from h- to u- points. Also does v- to q-."""
        return np.minimum( a, np.concatenate((a[:, -1:], a[:, :-1]), axis=1) )
    def _minh2v(a):
        """Minimum from h- to v- points. Also does u- to q-."""
        return np.minimum( a, np.concatenate((a[-1:, :], a[:-1, :]), axis=0) )

    def step(self, dt):
        """
        dt   - Time step [s]
        """

        # Forcing on eta
        if self.h_relax>0:
            eta_dev = np.mean( self.eta, axis=1, keepdims=True ) - self.eta_target
            self.eta = self.eta - ( dt * self.h_relax ) * eta_dev

        # Continuity equation (uses u,v at [n])
        h = self.D + self.eta # Total thickness
        hq = DQSWE._u2q( DQSWE._h2u( h ) )
        if self.iter % 2==0:
            hu = ( np.maximum( self.u, 0 ) * np.concatenate((h[:, -1:], h[:, :-1]), axis=1) + np.minimum( self.u, 0 ) * h ) # Upwinded h*u on western edge
            self.eta = self.eta - ( dt / self.dx ) * DQSWE._diu( hu )
            h = self.D + self.eta
            hv = ( np.maximum( self.v, 0 ) * np.concatenate((h[-1:, :], h[:-1, :]), axis=0) + np.minimum( self.v, 0 ) * h ) # Upwinded h*v on southern edge
            self.eta = self.eta - ( dt / self.dy ) * DQSWE._djv( hv )
        else:
            hv = ( np.maximum( self.v, 0 ) * np.concatenate((h[-1:, :], h[:-1, :]), axis=0) + np.minimum( self.v, 0 ) * h ) # Upwinded h*v on southern edge
            self.eta = self.eta - ( dt / self.dy ) * DQSWE._djv( hv )
            h = self.D + self.eta
            hu = ( np.maximum( self.u, 0 ) * np.concatenate((h[:, -1:], h[:, :-1]), axis=1) + np.minimum( self.u, 0 ) * h ) # Upwinded h*u on western edge
            self.eta = self.eta - ( dt / self.dx ) * DQSWE._diu( hu )
        # h = self.D + self.eta # Needed?

        # Explicit accelerations
        uip1 = np.concatenate((self.u[:, 1:], self.u[:, :1]), axis=1)
        vjp1 = np.concatenate((self.v[1:, :], self.v[:1, :]), axis=0)
        # Enquist-Oscher u^2 + v^2
        K = np.maximum( self.u, 0 )**2 + np.minimum( uip1, 0 )**2
        K = K + np.maximum( self.v, 0 )**2 + np.minimum( vjp1, 0 )**2
        B = self.g * self.eta + 0.5 * K # Potential + KE

        Bx = DQSWE._dih( B ) / self.dx
        By = DQSWE._djh( B ) / self.dy

        vx = DQSWE._dih( self.v ) / self.dx
        uy = DQSWE._djh( self.u ) / self.dy
        vy = DQSWE._djv( self.v ) / self.dy
        ux = DQSWE._diu( self.u ) / self.dx

        q = self.f + ( vx - uy )
        q = q / ( hq + self.hsub )
        q = q * ( hq / ( hq + self.hsub ) ) # Hack to mask q ##################################################
        # qhv = DQSWE._q2u( q ) * DQSWE._q2u( DQSWE._v2q( hv ) )  # issues with vanishing layers
        # qhu = DQSWE._q2v( q ) * DQSWE._q2v( DQSWE._u2q( hu ) )
        qhv = DQSWE._q2u( q * DQSWE._v2q( hv ) )
        qhu = DQSWE._q2v( q * DQSWE._u2q( hu ) )
        D_tension = ( ux - vy )
        D_shear = ( uy + vx )
        # Use latest h here, but not in q !!!
        h = self.D + self.eta # Total thickness
        hq = DQSWE._minh2v( DQSWE._minh2u( h ) )
        # h, hq = 1+0*h, 1+0*hq
        uxxyy = DQSWE._dih( self.nu * h * D_tension ) / self.dx + DQSWE._djv( self.nu * hq * D_shear ) / self.dy
        uxxyy = uxxyy / DQSWE._h2u( h + self.hsub )
        vxxyy = DQSWE._diu( self.nu * hq * D_shear ) / self.dx - DQSWE._djh( self.nu * h * D_tension ) / self.dy
        vxxyy = vxxyy / DQSWE._h2v( h + self.hsub )

        # rDu = 1 / ( DQSWE._h2u( self.D ) + self.hsub )
        rDu = 1 / ( self.Do + self.hsub ) ##############################################################################
        udot = ( self.taux - self.epsilon * self.u ) * rDu + ( qhv - Bx ) + uxxyy
        # rDv = 1 / ( DQSWE._h2v( self.D ) + self.hsub )
        rDv = 1 / ( self.Do + self.hsub ) ##############################################################################
        vdot = ( self.tauy - self.epsilon * self.v ) * rDv - ( qhu + By ) + vxxyy

        # Update momentum components with implicit terms
        edtp1 = 1. + self.alpha_e * dt * self.epsilon * rDu
        afdt = self.alpha_f * dt * DQSWE._q2u( self.f )
        du = ( edtp1 * udot + afdt * DQSWE._q2u( DQSWE._v2q( vdot ) ) ) / ( afdt**2 + edtp1**2 )
        self.u = self.u + dt * du
        edtp1 = 1. + self.alpha_e * dt * self.epsilon * rDv
        afdt = self.alpha_f * dt * DQSWE._q2v( self.f )
        dv = ( edtp1 * vdot - afdt * DQSWE._q2v( DQSWE._u2q( udot ) ) ) / ( afdt**2 + edtp1**2 )
        self.v = self.v + dt * dv

        # self.Ro = np.max( ( vx - uy ) / self.f )
        self.time = self.time + dt
        self.iter = self.iter + 1
