"""[EPSREL] Field-relative denominator floor for the east Orlanski phase-speed estimator.

WHY
---
The estimator is   rx = -dphi_t*dphi_x / (dphi_x^2 + dphi_y^2),  denom = dphi_x^2 + dphi_y^2.
Numerator and denominator both scale as phi^2, so rx is SCALE-INVARIANT in phi.  The only
thing that breaks that invariance is the small-denominator floor, which was historically a
single ABSOLUTE constant (1e-3) shared by h, u and v.  Because thickness increments are
~250x (layer 0) to ~1400x (layer 1) larger than velocity increments, that one constant sits
in a completely different place in each field's denom distribution:

    measured on the 2-layer baroclinic run, fraction of rows with denom < 1e-3
        h: 0.0% (L0) / 0.0% (L1)      u: 55% (L0) / 95% (L1)      v: 54% (L0) / 93% (L1)

A floored row has |rx| shrunk toward zero, and rx -> 0 turns the outflow update
    phi_b = (phi_prev_b + rx*pim1 - ry*dy_b)/(1+rx)   into   phi_b = phi_prev_b - ry*dy_b,
i.e. PERSISTENCE.  So for u and v -- and almost totally for a quiescent deep layer -- the
open boundary was not radiating, it was persisting.  (The floor can never flip the
inflow/outflow branch: sign(rx) = -sign(dphi_t*dphi_x), independent of the floor.)

THE FIX
-------
Give the floor the same phi^2 scaling the ratio has:

    eps[c,k] = EPS_ABS * ( s[c,k] / s[h,k] )^2 ,   s[c,k] = median sqrt(dphi_x^2 + dphi_y^2)

Normalising each field against **h of the same layer** makes eps[h,k] == EPS_ABS EXACTLY, so
the h field is bit-identical to the historical scheme and acts as a built-in null control;
only u and v change.  The floor then sits at the same RELATIVE position in every field's
denom distribution, so all fields floor at h's rate (~15% here) instead of h at 0% and u,v
at 55-95%.  This is NOT "u,v stop flooring" -- it is "u,v floor like h does".

NOTE denom is PURELY SPATIAL (dphi_t enters only the upwind selector, not denom), so s can be
estimated from stored snapshots -- no per-step probe is needed.
"""
import numpy as np

EPS_ABS_DEFAULT = 1e-3          # [EPSREL] the historical absolute floor; h keeps exactly this
FIELDS = ('h', 'u', 'v')        # [EPSREL] row order of the (3, nk) table the drivers expect


def increment_scales(frames, b_obc, y_periodic=False, reduce='median'):
    """Characteristic increment scale s[c,k] per (field, layer) from stored snapshots.

    frames   : dict with keys 'h','u','v', each (nt, nk, nj, ni) -- e.g. the control run's
               sampled output.  Only columns b-2, b-1 and rows j+/-1 are touched.
    b_obc    : OBC column index in the SAME column indexing as `frames`.
    y_periodic : tangential neighbour convention at the walls (mirror when False), matching
               _orlanski_east.
    reduce   : 'median' (default, robust) or 'rms'.

    Returns (3, nk) array of s, rows ordered h, u, v.

    The tangential difference uses the |.| of BOTH one-sided candidates averaged, rather than
    the kernel's dphi_t-dependent upwind pick: the pick needs a time increment, and for a
    magnitude SCALE the two candidates are statistically interchangeable.  This approximation
    affects only the calibration constant, never the scheme.
    """
    out = []
    for name in FIELDS:
        A = np.asarray(frames[name])
        if A.ndim != 4:
            raise ValueError(f'[EPSREL] frames[{name!r}] must be (nt,nk,nj,ni), got {A.shape}')
        bm1 = A[:, :, :, int(b_obc) - 1]                         # (nt, nk, nj) at column b-1
        bm2 = A[:, :, :, int(b_obc) - 2]                         # (nt, nk, nj) at column b-2
        dphi_x = bm1 - bm2                                       # exactly the kernel's dphi_x
        if y_periodic:
            jm, jp = np.roll(bm1, 1, axis=2), np.roll(bm1, -1, axis=2)
        else:                                                    # mirror ghost at the walls
            jm = np.concatenate([bm1[:, :, :1], bm1[:, :, :-1]], axis=2)
            jp = np.concatenate([bm1[:, :, 1:], bm1[:, :, -1:]], axis=2)
        dy_a, dy_b = bm1 - jm, jp - bm1
        dphi_y2 = 0.5 * (dy_a ** 2 + dy_b ** 2)                  # mean of the two candidates
        den = dphi_x ** 2 + dphi_y2                              # (nt, nk, nj)
        den = np.moveaxis(den, 1, 0).reshape(den.shape[1], -1)   # (nk, nt*nj)
        s = np.sqrt(np.median(den, axis=1) if reduce == 'median' else np.mean(den, axis=1))
        out.append(s)
    return np.ascontiguousarray(np.array(out, dtype=np.float64))


def _denom(frames, name, b_obc, y_periodic):
    """(nk, nt*nj) sample of denom = dphi_x^2 + dphi_y^2 for one field."""
    A = np.asarray(frames[name])
    bm1, bm2 = A[:, :, :, int(b_obc) - 1], A[:, :, :, int(b_obc) - 2]
    dphi_x = bm1 - bm2
    if y_periodic:
        jm, jp = np.roll(bm1, 1, axis=2), np.roll(bm1, -1, axis=2)
    else:
        jm = np.concatenate([bm1[:, :, :1], bm1[:, :, :-1]], axis=2)
        jp = np.concatenate([bm1[:, :, 1:], bm1[:, :, -1:]], axis=2)
    den = dphi_x ** 2 + 0.5 * ((bm1 - jm) ** 2 + (jp - bm1) ** 2)
    return np.moveaxis(den, 1, 0).reshape(den.shape[1], -1)


def floor_fraction(frames, b_obc, eps, y_periodic=False):
    """(3, nk) fraction of (t, j) samples where the floor would bind, for a given eps table.

    This is the number to look at -- far more meaningful than the scale itself, and directly
    comparable to the run-time `flags_*_row` bit 1.  NOTE it depends on WHICH ROWS are
    included: the kernel runs on every j, so all rows are sampled here.  Statistics taken over
    a sponge-masked subset (e.g. the NN training pack) exclude the quiescent near-wall rows and
    therefore understate the floor rate substantially.
    """
    E = np.asarray(eps, dtype=np.float64)
    out = np.zeros_like(E)
    for c, fn in enumerate(FIELDS):
        den = _denom(frames, fn, b_obc, y_periodic)
        for k in range(den.shape[0]):
            out[c, k] = float((den[k] < E[c, k]).mean())
    return out


def eps_table(scales, eps_abs=EPS_ABS_DEFAULT, ref_field='h'):
    """(3, nk) denominator-floor table from (3, nk) increment scales.

    eps[c,k] = eps_abs * (s[c,k] / s[ref,k])^2 -> the reference field's row is EXACTLY
    eps_abs, so that field is bit-identical to the historical uniform-floor scheme.
    """
    S = np.asarray(scales, dtype=np.float64)
    if S.ndim != 2 or S.shape[0] != 3:
        raise ValueError(f'[EPSREL] scales must be (3, nk), got {S.shape}')
    if not np.all(np.isfinite(S)) or np.any(S <= 0):
        raise ValueError('[EPSREL] increment scales must be finite and > 0')
    ref = S[FIELDS.index(ref_field)]
    return np.ascontiguousarray(float(eps_abs) * (S / ref[None, :]) ** 2)


def uniform_table(nk, eps_abs=EPS_ABS_DEFAULT):
    """(3, nk) historical uniform floor -- the bit-identical default."""
    return np.ascontiguousarray(np.full((3, int(nk)), float(eps_abs)))


def describe(eps, scales=None, eps_abs=EPS_ABS_DEFAULT):
    """Human-readable summary; returns the printed string."""
    E = np.asarray(eps, dtype=np.float64)
    lines = ['[EPSREL] denominator floor table (rows h,u,v; columns = layers):']
    for c, fn in enumerate(FIELDS):
        row = '   %s: ' % fn + '  '.join('L%d %.4g' % (k, E[c, k]) for k in range(E.shape[1]))
        if scales is not None:
            row += '   | s = ' + ' '.join('%.4g' % s for s in np.asarray(scales)[c])
        row += '   | x%.3g vs %g' % (E[c].mean() / eps_abs, eps_abs)
        lines.append(row)
    if np.allclose(E, eps_abs):
        lines.append('   -> UNIFORM (historical absolute floor): bit-identical to pre-[EPSREL] runs')
    else:
        lines.append('   -> RELATIVE: h row is exactly %g (bit-identical control); u,v rescaled' % eps_abs)
    txt = '\n'.join(lines)
    print(txt)
    return txt
