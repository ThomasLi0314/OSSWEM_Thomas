# Prescribed-boundary ("stored data") replay — implementation notes

This document explains the `# [OBC]`-tagged changes in
[`../OSSWEM_obc.py`](../OSSWEM_obc.py) and the new cells in
[`1-layer-jet-obc.ipynb`](1-layer-jet-obc.ipynb): **what** was changed, **why**, and the
**formulas** each change is tied to.

---

## 1. Goal and the core idea

We are building toward an **Open Boundary Condition (OBC)** at the downstream (eastern)
edge of the western sponge in the barotropic-jet channel experiment. Before implementing a
real OBC, we first verify the *plumbing* of "prescribing" a boundary:

1. **Sim 1 (record):** run the model normally and, **every time step**, save the state
   `(u, v, h)` at the **last `n_bc` columns of the western sponge** (`n_bc` is
   user-chosen, **default 2**) — the columns where the sponge hands off to the free
   interior.
2. **Sim 2 (replace):** re-run from the **same initial condition** and, every step,
   **overwrite** those columns with the stored data, *after computing the free update
   but before that update can influence the next phase*. Compare free-vs-stored first
   (the "comparison result"), then replace.

Because Sim 2 keeps the full sponge active and the stored values are exactly what those
columns would have evolved to, **the overwrite is a no-op and Sim 2 reproduces Sim 1
bit-for-bit** (misfits = `0.0`). That is the success criterion at this stage — see
[§5](#5-why-the-misfit-is-exactly-zero). A future OBC will keep this same machinery but
supply the boundary columns from a boundary scheme instead of from stored data.

The boundary columns (by index, same `i` for `u`, `v`, `h`, all rows `j`, all layers `k`):

```
n_bc    = 2                                          # number of boundary columns (>=1; default 2)
i_edge  = np.where(M.xh1 < Lw_uv + width)[0].max()   # eastern edge of the western sponge
bc_cols = np.arange(i_edge - n_bc + 1, i_edge + 1)   # the last n_bc columns
```

---

## 2. The model and its time step (context for the formulas)

The single layer solves the rotating shallow-water equations in **vector-invariant
form**. With layer thickness $h$, free surface $\eta = h - D$, Coriolis $f$, relative
vorticity $\zeta = \partial_x v - \partial_y u$, potential vorticity $q=(f+\zeta)/h$,
Montgomery potential $M = g\,\eta$ and Bernoulli function $B = M + \tfrac12(u^2+v^2)$:

$$\partial_t h + \nabla\!\cdot(h\,\mathbf{u}) = 0$$

$$\partial_t u = \;\; q\,(h v) - \partial_x B + \nu_h\,(\nabla\!\cdot\boldsymbol\tau)_x + \tfrac{\tau^x}{h} - (\mathbf{L}u)$$

$$\partial_t v = -\,q\,(h u) - \partial_y B + \nu_h\,(\nabla\!\cdot\boldsymbol\tau)_y + \tfrac{\tau^y}{h} - (\mathbf{L}v)$$

where $\mathbf{L}$ is the vertical-stress/bottom-drag operator. One time step in
[`_step_numba`](../OSSWEM_obc.py#L81) executes these phases **in this order**:

| Phase | What it computes | Updates | Code |
|---|---|---|---|
| **P0** | PV thickness `hq_pre` at q-points from $h^n$ | — | L123–129 |
| **P1 — continuity** | upwind fluxes `hu,hv`; directional split | **`h`** → $h^{n+1}$ | L132–169 |
| **P2** | `eta`, `M`, `B`, `h_at_u/v`, reciprocals | — | L196–231 |
| **P3 — explicit momentum** | PV-Coriolis flux, $\nabla B$, viscous stress | `udot,vdot` | L233–303 |
| **P4** | subtract $\mathbf{L}u^n,\mathbf{L}v^n$ | `udot,vdot` | L305–327 |
| **P5 — implicit solve** | complex TDMAH2 (Coriolis Crank–Nicolson) | **`u`,`v`** → $u^{n+1},v^{n+1}$ | L329–399 |
| **P6 — restoring** | backward-Euler sponge nudging | `h`,`u`,`v` | L434–end |

Two facts drive the design:

- **`h` is fully determined by the end of P1**, and the momentum phases **P2–P5 read that
  post-continuity `h`** (they run *before* the restoring in P6).
- The restoring in **P6 is applied last**, so the values `h,u,v` carry into the next step
  *after* nudging.

The numerical scheme thus matches the user's description exactly: **first update `h` from
neighbouring `(u,v)` (continuity), then update `(u,v)` together (the implicit solve,
needed because Coriolis couples `u` and `v`).**

---

## 3. The two record/replace blocks in `_step_numba`

### 3a. Extended signature
[`../OSSWEM_obc.py:81–87`](../OSSWEM_obc.py#L81)

```python
def _step_numba(..., hsub, iter_num,
                bc_mode, bc_cols, h_bc, u_bc, v_bc, h_diff, u_diff, v_diff):  # [OBC]
```

- `bc_mode`: `0` = off (existing behaviour, blocks skipped), `1` = record, `2` = replace.
- `bc_cols`: an `int64` array of boundary column indices, **any length `n_bc ≥ 1`**.
- `h_bc, u_bc, v_bc`: per-step stores, shape `(nk, nj, n_bc)`.
- `h_diff, u_diff, v_diff`: per-field outputs, shape `(2,)` holding `[max_abs, rms]`.

### 3b. Block A — `h` (after continuity, before momentum reads it)
[`../OSSWEM_obc.py:171–194`](../OSSWEM_obc.py#L171) — inserted between **P1** and **P2**.

This is the placement that makes "update `h`, replace, *then* update `(u,v)`" literal: the
replaced `h` feeds `eta/M/B/h_at_u` and therefore the `(u,v)` solve.

```python
n_bc = bc_cols.shape[0]
if bc_mode == 1:                       # record h^{n+1} at the boundary columns
    for k in range(nk):
        for j in range(nj):
            for c in range(n_bc):
                h_bc[k,j,c] = h[k,j,bc_cols[c]]
elif bc_mode == 2:                     # compare, then replace
    maxd = 0.0; ss = 0.0
    for k in range(nk):
        for j in range(nj):
            for c in range(n_bc):
                d = h[k,j,bc_cols[c]] - h_bc[k,j,c]
                if abs(d) > maxd: maxd = abs(d)
                ss += d*d
                h[k,j,bc_cols[c]] = h_bc[k,j,c]   # overwrite with stored
    h_diff[0] = maxd
    h_diff[1] = ( ss / ( nk*nj*n_bc ) )**0.5
```

**Comparison formulas** (over the `n_bc` columns × all rows × all layers, $N = n_k n_j \cdot n_{bc}$):

$$\text{max\_abs} = \max_{k,j,c}\;\bigl|\,h^{\text{free}}_{k,j,c} - h^{\text{stored}}_{k,j,c}\,\bigr|,
\qquad
\text{rms} = \sqrt{\frac1N \sum_{k,j,c}\bigl(h^{\text{free}}_{k,j,c}-h^{\text{stored}}_{k,j,c}\bigr)^2}.$$

### 3c. Block B — `(u,v)` (after the implicit solve, before restoring)
[`../OSSWEM_obc.py:401–434`](../OSSWEM_obc.py#L401) — inserted between **P5** and **P6**.

Identical pattern for `u` and `v`, writing `[max_abs, rms]` into `u_diff`/`v_diff` and then
overwriting `u`/`v` at the boundary columns. It sits *before* the restoring (P6) so that block A
(`h`) and block B (`u,v`) record/replace at the **same pre-restoring phase point**, which
is the key to identical dynamics ([§5](#5-why-the-misfit-is-exactly-zero)).

### 3d. Why these are plain serial loops (not `prange`)
The blocks use serial `range` loops over `(k,j)` (only two `i` columns). The function is
`@njit(parallel=True)`; the file's own docstring warns that Numba's `ParallelAccelerator`
**mis-compiles array reductions/broadcasts** (it produced blow-ups), so every reduction in
this file is written as an explicit loop. The `max`/sum reductions here follow that rule.
The blocks also run in the **sequential region between** the parallel `prange` sections, so
the column writes are race-free.

---

## 4. Plumbing so existing code is untouched

### 4a. Cached dummies + single call site
- [`__init__`](../OSSWEM_obc.py#L591): `self._bc_dummy = np.zeros((nk,nj,2))`,
  `self._diff_dummy = np.zeros(2)` — passed in `bc_mode=0` so the normal step never
  allocates.
- New [`_step_core`](../OSSWEM_obc.py#L994): the **only** `_step_numba` call site; it
  forwards the boundary args and then advances `self.time`/`self.iter`.
- [`step`](../OSSWEM_obc.py#L1011) now just calls `_step_core(dt, 0, 0, 0, dummies…)`.
  With `bc_mode=0`, blocks A/B are skipped, so **`M.run()` and every existing notebook are
  bit-identical to before** the change.

### 4b. The two new run methods
- [`run_record_bc(dt, samp, nsamps, bc_cols, store_downstream=False, probe_i0=None, n_probe=1)`](../OSSWEM_obc.py#L882):
  mirrors `run()` (same CFL banner via the factored
  [`_print_run_info`](../OSSWEM_obc.py#L856), same sampling and NaN-blowup trim) but calls
  `_step_core(dt, 1, …)` each step, writing into per-step slices
  `h_bc_all[n], u_bc_all[n], v_bc_all[n]` of shape `(nsteps, nk, nj, n_bc)`.
  Returns the usual sampled `u,v,h,time`, the three boundary stores, **and** a `probe`
  (8th return).
  - **Optional downstream probe** (off by default): set `store_downstream=True` and give a
    user-defined start column `probe_i0` and column count `n_probe` to *also* store
    `(u,v,h)` every step at `n_probe` contiguous interior columns — a pure diagnostic, not
    replaced. It is read from the **post-step** state in Python (no `_step_numba` change);
    downstream of the sponge the restoring is zero there, so that equals the pre-restoring
    phase used for the boundary stores. Returned as `probe` — a dict with `cols`, `x_km`
    and `h`/`u`/`v` of shape `(nsteps, nk, nj, n_probe)` — or `None` when disabled.
- [`run_replace_bc(dt, samp, nsamps, bc_cols, h_bc_all, u_bc_all, v_bc_all)`](../OSSWEM_obc.py#L936):
  calls `_step_core(dt, 2, …)` each step, copies the `[max_abs, rms]` outputs into per-step
  time series, prints the run-max summary, and returns `u,v,h,time` **plus** a `diffs`
  dict (`t_step`, `h_max/h_rms/u_max/u_rms/v_max/v_rms`). It raises if the stored data does
  not cover `nsteps`.

**Memory:** each store is `nsteps·nk·nj·2·8` bytes. For the default
`run_params=[200,500,100]` (`nsteps=50 000`, `nk=1`, `nj=120`) that is ~96 MB × 3 ≈ 290 MB;
it scales linearly with `nsteps`.

### 4c. Notebook flow (setup cell unchanged)
1. Import → `from OSSWEM_obc import SSWEM`.
2. **New cell after setup:** compute `bc_cols` and snapshot the IC
   `ic = (M.u.copy(), M.v.copy(), M.h.copy(), M.time, M.iter)`. **`M.iter` is included**
   because the continuity directional split alternates on `iter % 2`
   ([L138](../OSSWEM_obc.py#L138)) — Sim 2 must restart the parity from the same value.
3. **Run cell → Sim 1:** `u,v,h,time, h_bc_all,u_bc_all,v_bc_all, probe = M.run_record_bc(*run_params, bc_cols)`
   (`probe` is `None` unless the optional downstream store is enabled).
4. **New cell → Sim 2:** restore `M` from `ic`, then `run_replace_bc(...)`; prints
   `max|h2-h|`, `max|u2-u|`, `max|v2-v|` over the sampled snapshots (expected `0.0`).
5. **New cell:** plot the per-step boundary misfit (max-abs and RMS for `h,u,v`).
6. **New cell:** side-by-side PV animation, original (Sim 1) vs prescribed-boundary
   (Sim 2) — visually indistinguishable.

---

## 5. Why the misfit is exactly zero

**Claim.** With the sponge active in both runs and the stored data taken from Sim 1's own
IC, Sim 2 is bit-identical to Sim 1, so every misfit is `0.0`.

**Stencil reach.** Both the continuity update (P1) and the momentum update (P2–P5) of any
column $i$ read only columns $\{i-1, i, i+1\}$ — a reach of $\pm 1$ in $i$. (Check the index
shifts `im=i-1`, `ip=i+1` throughout, e.g. the flux divergence
[L153–169](../OSSWEM_obc.py#L153) and the accelerations [L237–296](../OSSWEM_obc.py#L237).)
Given this $\pm 1$ reach, a single prescribed column already shields the first interior
column, so `n_bc = 1` is sufficient for the identical-dynamics argument; `n_bc ≥ 2`
(the default 2) just prescribes a thicker band — handy as a safety margin and for a future
wider/higher-order boundary stencil. The argument below holds for any `n_bc ≥ 1`.

**Phase-matched record/replace.** Block A replaces `h` at the boundary at the *same*
algorithmic instant Sim 1 recorded it (end of P1, before P2 reads `h`); block B does the
same for `(u,v)` (end of P5, before P6). So the *only* difference Sim 2 could introduce is
to overwrite the boundary columns with their own recorded values.

**Induction.** Suppose Sim 1 and Sim 2 hold an identical full field at time $n$. Running
the identical compiled step:

- P1 produces identical `h` everywhere (deterministic: the `prange` loops only write
  distinct `[k,j,i]`, no cross-iteration reductions, so the result is independent of thread
  scheduling). Block A finds `h^{free} = h^{stored}` at the boundary → **misfit 0**; the
  overwrite changes nothing.
- P2–P5 then read identical inputs → identical `u,v` everywhere; block B → **misfit 0**.
- P6 restoring is deterministic given the field → identical.

Hence the fields match at time $n+1$, and by induction at all times. The misfits are
**exactly `0.0`** (not just small), which is what the smoke test confirmed.

**Periodicity caveat (intentional, for now).** The channel is periodic in $x$, so the
interior's *eastern* edge wraps to $i=0$, which is inside the sponge and is **not** a
prescribed column. Information therefore still reaches the interior from the sponge through
that wrap. That is fine here because Sim 2 simulates the *whole* domain with the sponge on,
so everything (including the sponge west of the prescribed columns) evolves identically.
The aspiration in the plan — "information enters the interior *only* through the prescribed
data" — is **not yet enforced**; it becomes meaningful when a real OBC replaces the sponge
and the region west of the boundary is no longer integrated.

---

## 6. What changes for a real OBC later

The machinery is reusable as-is. To turn this into an OBC, keep blocks A/B and
`run_replace_bc`, but **feed the boundary columns from a boundary scheme** (e.g. radiation
/ Flather / characteristic-based values, or externally supplied data) instead of from
`*_bc_all` recorded off the sponge — and drop the sponge over the interior so the prescribed
columns become the sole pathway for incoming information. The comparison diagnostics then
report a *physical* misfit (how much the interior wants to differ from the imposed
boundary) rather than the current machine-zero consistency check.
