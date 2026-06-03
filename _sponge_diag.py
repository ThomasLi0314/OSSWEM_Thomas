"""Standalone sponge diagnostic (does NOT touch the notebook).

Question: extending the western sponge in x does not bring the profile at the
*end* of the sponge any closer to the target. Why, and what helps?

Strategy: replicate the 1-layer-jet-modify-sponge setup, then sweep three knobs
independently and measure the time-mean RMS misfit (model - target) on the
meridional profile at the sponge's last column:
  (A) sponge WIDTH  (the extra extension `d`)     -- the user's current lever
  (B) sponge RATE   (restoring strength)          -- the hypothesised real lever
  (C) edge TAPER    (hard cut vs smooth ramp-down) -- the handoff quality

Prints a table; saves an edge-profile comparison figure to Testings/output/.
"""
import os, sys, io, contextlib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # find OSSWEM.py
from OSSWEM import SSWEM

# ---- fixed configuration (matches the notebook) ----------------------------
NJ          = 60
RUN_PARAMS  = (200., 500, 25)          # dt, steps/segment, segments
G, HO, LX   = 0.08, 800., 900.e3
FO, BETA    = 1.e-4, 3.0e-11
EPS, NU_H   = 2.5e-5, 2e2
LW, YB      = 150.e3, 100.e3
RATE0       = 1.0e-5
JJ          = slice(3, -3)             # interior rows (skip dry walls)

ramp  = lambda x, x0, L: np.clip((x - x0) / (L - x0), 0, 1) ** 2
ramp2 = lambda x, x0, L: np.clip(1.5 * (x - x0) / (L - x0), 0, 1) ** 2


def build_and_run(rate=RATE0, extra_width=50.e3, taper='hard'):
    """Build the model with the given sponge, run it, return (M, u, v, h, time, i_edge)."""
    M = SSWEM(NJ + NJ // 2, G, HO, LX, FO, BETA, EPS, NU_H, nj=NJ)
    M.D[0, :] = 0
    M.D[-1, :] = 0

    # target jet + geostrophically balanced thickness (identical to the notebook)
    M.set_u_target_jet(1.0)
    uy = M.u_target[0].mean(axis=1)
    f_h = M.fo + M.beta * M.yh1
    eta = -(1.0 / M.g[0]) * np.cumsum(f_h * uy) * M.dy
    eta -= eta.mean()
    hbal = np.maximum(M.D + eta[:, None], 0.0)

    def west(x):
        up = ramp2(x, 0., LW)                                  # 0 -> 1 rise over [0, Lw]
        if taper == 'hard':
            return up * (x < LW + extra_width)                 # hold, then drop to 0
        # smooth: hold to Lw, then quadratic taper down to 0 at Lw+extra_width
        d = max(extra_width, 1e-9)
        return up * np.clip((LW + extra_width - x) / d, 0, 1) ** 2

    def relax(xx, yy):
        return rate * np.maximum(np.maximum(west(xx),
                                            ramp(yy, M.Ly - YB, M.Ly)),
                                 ramp(M.Ly - yy, M.Ly - YB, M.Ly))

    M.u_relax = relax(M.xu, M.yu)
    M.v_relax = relax(M.xv, M.yv)
    M.h_relax = relax(M.xh, M.yh)
    M.h_target = hbal[None, :, :].copy()

    M.u[0] = M.u_target[0].copy()
    M.h[0] = hbal.copy()
    M.perturb_h(20., 0.05 * M.Lx, 0.5 * M.Lx, 0.55 * M.Ly)

    with contextlib.redirect_stdout(io.StringIO()):             # silence run()'s CFL spam
        u, v, h, time = M.run(*RUN_PARAMS)

    i_edge = int(np.where(M.xh1 < LW + extra_width)[0].max())
    return M, u, v, h, time, i_edge


def edge_misfit(M, u, v, h, time, i_edge, frac=0.5):
    """Time-mean (2nd half) RMS misfit of the interior profile at column i_edge."""
    n0 = int(len(time) * frac)
    eta_t = M.h_target[0] - M.D
    u_e = u[n0:, 0, JJ, i_edge] - M.u_target[0, JJ, i_edge]
    v_e = v[n0:, 0, JJ, i_edge] - M.v_target[0, JJ, i_edge]
    h_e = (h[n0:, 0, JJ, i_edge] - M.D[JJ, i_edge]) - eta_t[JJ, i_edge]
    rms = lambda a: float(np.sqrt(np.mean(a ** 2)))
    return rms(u_e), rms(v_e), rms(h_e)


def mean_edge_profiles(M, u, v, h, time, i_edge, frac=0.5):
    n0 = int(len(time) * frac)
    return (u[n0:, 0, :, i_edge].mean(0),
            (h[n0:, 0, :, i_edge] - M.D[:, i_edge]).mean(0))


# ---------------------------------------------------------------------------
print("=" * 78)
print(f"Sponge diagnostic  (nj={NJ}, ni={NJ+NJ//2}, run_params={RUN_PARAMS}, "
      f"Lw={LW/1e3:.0f} km, base rate={RATE0:.1e} 1/s)")
print("Metric: time-mean (2nd half) RMS misfit of the interior profile at the")
print("        sponge's LAST column.  u,v in m/s (jet peak = 1);  eta in m.")
print("=" * 78)

records = []   # (label, rate, width_km, taper, x_edge_km, u_rms, v_rms, h_rms)
profiles = {}  # label -> (yh1_km, u_target, u_model, x_edge_km)

def run_case(label, rate, extra_width, taper):
    M, u, v, h, time, i_edge = build_and_run(rate, extra_width, taper)
    ur, vr, hr = edge_misfit(M, u, v, h, time, i_edge)
    xkm = M.xh1[i_edge] / 1e3
    records.append((label, rate, extra_width / 1e3, taper, xkm, ur, vr, hr))
    um, _ = mean_edge_profiles(M, u, v, h, time, i_edge)
    profiles[label] = (M.yh1 / 1e3, M.u_target[0, :, i_edge], um, xkm)
    print(f"  {label:28s} edge x={xkm:6.1f} km | "
          f"u={ur:.4f}  v={vr:.4f}  eta={hr:.4f}")
    return ur, vr, hr

print("\n(A) WIDTH sweep  -- rate fixed at base, hard edge:")
for d in (0., 50.e3, 150.e3, 300.e3):
    run_case(f"width d={d/1e3:.0f}km", RATE0, d, 'hard')

print("\n(B) RATE sweep   -- width fixed at d=50 km, hard edge:")
for r in (3.e-5, 1.e-4, 3.e-4):
    run_case(f"rate={r:.0e}", r, 50.e3, 'hard')

print("\n(C) TAPER         -- smooth ramp-down over d instead of a hard cut:")
run_case("smooth taper, rate=1e-5", RATE0, 50.e3, 'smooth')
run_case("smooth taper, rate=1e-4", 1.e-4, 50.e3, 'smooth')

# ---- summary tables --------------------------------------------------------
print("\n" + "=" * 78)
print("SUMMARY (RMS misfit at the sponge's last column; smaller = closer to target)")
print("-" * 78)
print(f"{'case':30s} {'edge x':>8s} {'u':>9s} {'v':>9s} {'eta':>9s}")
for lab, r, wkm, tp, xkm, ur, vr, hr in records:
    print(f"{lab:30s} {xkm:7.1f}k {ur:9.4f} {vr:9.4f} {hr:9.4f}")
print("=" * 78)

# ---- figure: time-mean edge u-profile, target vs model ---------------------
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Testings', 'output')
    os.makedirs(outdir, exist_ok=True)
    sel = ["width d=0km", "width d=300km", "rate=3e-04", "smooth taper, rate=1e-4"]
    fig, axs = plt.subplots(1, len(sel), figsize=(4 * len(sel), 4), sharey=True)
    for ax, lab in zip(axs, sel):
        ykm, utgt, umod, xkm = profiles[lab]
        ax.plot(utgt, ykm, 'k--', label='target')
        ax.plot(umod, ykm, 'b-', label='model (time-mean)')
        ax.set_title(f"{lab}\nedge x={xkm:.0f} km"); ax.set_xlabel('u [m/s]')
        ax.legend(fontsize=8)
    axs[0].set_ylabel('y [km]')
    fig.suptitle('Edge u-profile vs target')
    fig.tight_layout()
    figpath = os.path.join(outdir, 'sponge_diag_edge_profiles.png')
    fig.savefig(figpath, dpi=110)
    print(f"saved figure: {figpath}")
except Exception as e:
    print("figure skipped:", e)
