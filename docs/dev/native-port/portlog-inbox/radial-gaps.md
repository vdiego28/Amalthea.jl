# Inbox: radial-geometry native-support-matrix gaps

Worktree `agent-ab13125817ecedc8e`. Task: close the radial ❌ cells in
`docs/dev/native-port/NATIVE_SUPPORT_MATRIX.md`'s Radial table. Do not edit
`PORT_LOG.md`/`BACKLOG.md`/`NATIVE_SUPPORT_MATRIX.md` directly (multi-agent
conflict avoidance) — this file carries what the lead should paste into
each.

---

## 1. PORT_LOG.md entry (paste verbatim, in the template)

```
## 2026-07-22 — Radial-geometry gap closure (item 1: EnvGrid Raman) — Claude (sonnet-5)
**Status:** complete (item 1). Item 2 (z-dependent radial linop/normfun)
assessed and deliberately NOT implemented — see design record below and
§2 of this inbox note. Item 3 (radial EnvGrid mixtures) not started (gated
on item 2 landing, per task instructions).
**Did:** Wired `RamanPolarEnv` (envelope Raman, SDO/rotational) into the
native radial path (`TransRadial`, EnvGrid) — closes the "Radial EnvGrid
Raman" ❌ cell in NATIVE_SUPPORT_MATRIX.md's Radial table. Previously only
`RamanPolarField` (RealGrid, carrier-field) was wired; `rhs_radial_env`
silently had no Raman step at all for EnvGrid.
**How:**
- `amalthea/src/native.rs`: new `CpuNativeSim::apply_raman_radial_env`
  method (~line 1318, right after `apply_raman_radial`) — composes two
  already-ported pieces, no new math: `rhs_mode_avg_env`'s envelope Raman
  formula (`sqr!(R::RamanPolarEnv,E) = 1/2·|E|²`, no `thg` branch, real
  `raman_solver.solve()` in/out) applied independently per r-column, the
  same way `apply_raman_radial` already applies the RealGrid ADE solve per
  r-column. Includes the same rayon contiguous-column-group parallelization
  as `apply_raman_radial` (each worker gets its own cloned
  `TimeDomainRamanSolver` — state resets at every `solve()` call, so a
  clone's output is bit-identical to the shared instance — and disjoint
  column slices of every buffer).
- Call site: `rhs_radial_env` (~line 1897, new "Step 3b") — inserted between
  the existing Kerr step and the time-window step, gated on `self.has_raman`
  (already set by the existing `native_set_raman_params` FFI — no FFI
  surface change was needed; that setter already sizes `raman_intensity`/
  `raman_p` as `n_time_over*n_r` whenever `s.is_radial`, regardless of
  RealGrid/EnvGrid, so both radial Raman paths share the same scratch
  buffers).
- `src/RK45.jl`, radial guard block (~line 1465-1660): extended the
  Kerr/plasma/Raman response allowlist and the Raman-wiring loop to also
  accept `r isa Amalthea.Nonlinear.RamanPolarEnv` alongside
  `RamanPolarField`, mirroring the mode-averaged wiring's existing
  `RamanPolarField || RamanPolarEnv` pattern exactly (same `thg_flag =
  RamanPolarField ? Cint(r.thg) : Cint(1)` convention, same
  `flatten_sdo_oscillators` eligibility check, same
  `native_set_raman_params` FFI call — no new FFI symbol). Explicitly
  rejects `RamanRespIntermediateBroadening` (`:SiO2`) for radial with its
  own `NativeIneligible` message (that response needs the FFT-convolution
  kernel, `native_set_raman_fft_params`, which is only wired for
  mode-averaged EnvGrid — not extended here, out of scope for this pass).
- New FFI/exported symbols: **none**. Everything reuses
  `native_set_raman_params` (already radial-aware) and
  `apply_raman_radial_env` is a private `native.rs` method, not exported.
**Decisions:**
- Followed the existing mode-averaged `RamanPolarField || RamanPolarEnv`
  pattern verbatim rather than inventing a new eligibility shape — the task
  brief called this out explicitly ("composition of patterns that already
  exist, not new math") and it held.
- Test energy for the new equivalence test is half the RealGrid sibling
  test's (3e-5 J vs 6e-5 J) — see Gotchas.
**Gotchas:**
- At the RealGrid sibling test's energy (6e-5 J, same gas/pressure/length),
  the EnvGrid full-solve equivalence measured 1.14e-6 — just over the
  established ~1e-6 floor tier (TESTING.md §3), even though single-step
  agreement was clean (2.6e-8) and n_threads=1-vs-4 was bit-identical. This
  is the same "ADE-vs-FFT-convolution method difference accumulating over
  fixed-dt steps" floor `test_native_radial_raman.jl`'s own doc comment
  describes for the RealGrid case (single-step ~1e-8-1e-9, not a
  summation-order floor) — it is just numerically larger here because the
  envelope Kerr_env/RamanPolarEnv combination happens to accumulate it
  faster at this specific energy/length, not a code bug (ruled out by the
  clean single-step and bit-identical-threading results). Halving the test
  energy to 3e-5 J brings full-solve to 5.7e-7 (comfortable margin) while
  keeping the non-vacuousness check (Raman-on vs Raman-off) at 6.1e-4, five
  orders above the 1e-6 non-vacuousness bar. See probe sweep in the test
  file's comment.
- `native_set_raman_params` needed **zero** changes — it already keys its
  scratch-buffer sizing off `s.is_radial` alone (`n_time_over*n_r`), not
  off RealGrid-vs-EnvGrid, because the RealGrid radial path was wired
  first (Phase D.4) and already established that sizing convention. This
  is why the whole item landed with no FFI surface change at all.
**Tests:**
- `cd amalthea && RUSTFLAGS="-D warnings" cargo build --release` → clean,
  no warnings. `cargo test` → 69/69 pass (no regressions).
- New `test/test_native_radial_env_raman.jl` (5 testsets, all pass):
  - Single-step equivalence: **1.3e-8** (tier: <1e-7, same tier as the
    RealGrid sibling, for the same ADE-vs-FFT-conv method-difference
    reason).
  - Full-solve equivalence (fixed `max_dt=min_dt=dt`, 40 steps): **5.7e-7**
    (tier: <1e-6).
  - Backend-actually-native check: `RustNativeStepper` type assertion
    passes (not a silent fallback).
  - Non-vacuousness (Julia oracle, Raman-on vs Raman-off): **6.1e-4**, five
    orders above the 1e-6 floor — proves the equivalence test isn't passing
    because both paths silently dropped Raman.
  - `n_threads=1` vs `n_threads=4`: **bit-identical** (`s1.yn == s4.yn`),
    per the repo's parallel-branch bar.
- Full radial regression: `julia --project -e 'using TestItemRunner;
  @run_package_tests filter=ti->occursin("radial", ti.filename)'` → **189
  passed, 0 failed** (includes all pre-existing
  `test_native_radial{,_env,_mixture,_plasma,_raman,_shotnoise}.jl` files
  plus the new one) — confirms the shared radial guard-block edit didn't
  regress the RealGrid Raman/plasma/mixture/shot-noise paths.
- `test_rust_ffi.jl` (this worktree only): 2/2 pass.
**Next:** item 2 (radial z-dependent linop/normfun) — design record below,
not implemented (blocked on files outside this worktree's exclusive zone,
not on missing math). Item 3 (radial EnvGrid mixtures) not started.
```

---

## 2. BACKLOG.md status line(s) for the lead to set

Add under "Open remainders lifted out of the archived phases" (or wherever
the lead tracks live radial items):

```
- ⚪ Radial z-dependent linop/normfun (two-point pressure-gradient gas
  cell) — design assessed 2026-07-22, confirmed a clean port by direct
  analogy to Phase D.5's free-space `ZDepLinopFree`/`norm_free_gradient`/
  `zdep_free_*` (identical β1(z) closed form — no waveguide `nwg` term in
  either geometry — and identical `k2[iω]-k⊥²` linop/norm structure, just
  `kr2` (1-D, N_r) in place of `kperp2` (2-D flattened y,x)). NOT
  implemented: needs `LinearOps.jl` (`ZDepLinopRadial` +
  `make_linop_radial_gradient`) and `NonlinearRHS.jl`
  (`norm_radial_gradient`), both outside the assessing worktree's
  exclusive zone (native.rs radial region + RK45.jl radial guard block +
  test files only). See docs/dev/native-port/portlog-inbox/radial-gaps.md
  §3 for the full design record. Shovel-ready for whoever owns
  LinearOps.jl/NonlinearRHS.jl next.
```

Item 1 (`RamanPolarEnv` for radial) should move from BACKLOG's live
remainders (if it was tracked there) into the "Completed work — status
index" / ARCHIVE.md-style done list; it's no longer open.

---

## 3. NATIVE_SUPPORT_MATRIX.md — exact Radial-table replacement

Current (before this work):

```
## Radial (`TransRadial`)

| Response | RealGrid | EnvGrid |
|---|---|---|
| Kerr | ✅ | ✅ |
| Plasma (PPT/ADK) | ✅ | — (Julia has no EnvGrid `PlasmaCumtrapz`) |
| Raman (SDO, `CombinedRamanResponse`) | ✅ | ❌ (`RamanPolarField` only; no radial `RamanPolarEnv` wiring) |
| Shot noise | ✅ | ✅ |
| Gas mixtures | ⚠️ Kerr-only | ❌ (mixture densityfun rejected for radial EnvGrid — see RK45.jl mixture guard) |
| z-dependent linop/normfun | ❌ (not ported for radial geometry) | ❌ |
```

Replacement (after this work — only the Raman row and its own note change;
z-dependent row gets an updated note, not a status-symbol change):

```
## Radial (`TransRadial`)

| Response | RealGrid | EnvGrid |
|---|---|---|
| Kerr | ✅ | ✅ |
| Plasma (PPT/ADK) | ✅ | — (Julia has no EnvGrid `PlasmaCumtrapz`) |
| Raman (SDO, `CombinedRamanResponse`) | ✅ | ✅ (`RamanPolarEnv`, `apply_raman_radial_env` in native.rs, 2026-07-22; `:SiO2` intermediate-broadening still ❌ — not wired for radial, only for mode-averaged EnvGrid's `native_set_raman_fft_params` path) |
| Shot noise | ✅ | ✅ |
| Gas mixtures | ⚠️ Kerr-only | ❌ (mixture densityfun rejected for radial EnvGrid — see RK45.jl mixture guard) |
| z-dependent linop/normfun | ❌ (not ported for radial geometry — see below) | ❌ |
```

And append a note under the Radial section (mirrors the "This is the most
complete geometry..." note already under Mode-averaged):

```
The radial z-dependent linop/normfun (two-point pressure-gradient gas
cell, the same case Phase D.5 ported for free-space) is a confirmed clean
port by analogy — not blocked on missing math, only on touching
`LinearOps.jl`/`NonlinearRHS.jl` from outside the assessing worktree's
zone. See `docs/dev/native-port/portlog-inbox/radial-gaps.md` §3 for the
full design record before starting it.
```

---

## 4. Dependencies outside this worktree's zone

### Item 2 — radial z-dependent linop/normfun — design record (not implemented)

**Verdict: clean port by direct analogy to the already-shipped Phase D.5
free-space case. Not new analytic derivation work.** Confirmed, not
assumed, by reading both sides side by side:

- **Linop structure is identical in shape.** Free-space's constant/z-dep
  linop (`LinearOps._fill_linop_xy!`) computes
  `βsq = k2[iω] - kperp2[ii]` where `kperp2` is a fixed 2-D array from the
  Cartesian `(ky,kx)` grid. Radial's linop (`LinearOps._fill_linop_r!`,
  `src/LinearOps.jl:277-289`) computes `βsq = k2[iω] - kr2[ir]` where `kr2
  = q.k.^2` is a fixed 1-D array from the QDHT grid. Same formula, only the
  transverse-wavenumber array's shape differs (1-D vs flattened 2-D) — and
  `native.rs`'s existing `zdep_free_kperp2: Vec<f64>` field already stores
  the free-space case as a flat `Vec`, so a `zdep_radial_kr2: Vec<f64>`
  would need no new data-layout idea, just a shorter vector.
- **β1(z) closed form is identical, not just similar.** Neither free-space
  nor radial-free-space propagation has a waveguide cladding term (`nwg`)
  — both are literally "field propagating in gas-filled open space," so
  `LinearOps.make_linop_free_gradient`'s β1(z) derivation (`γ0`, `dγ0` from
  a BigFloat-differentiated Sellmeier closure, no `nwg0`/`dnwg0` unlike the
  mode-averaged `MarcatiliMode` case which does have a waveguide term) is
  exactly the formula radial needs too, with zero new derivation.
  `BETA1_ANALYTIC.md`'s free-space-specific reasoning (no `nwg` term)
  already covers this case.
- **The normfun z-dependence separates the same way.** `norm_radial`
  (`src/NonlinearRHS.jl:753-777`) computes
  `out[iω,ir] = sqrt(k2[iω]-kr2[ir])/(μ0·ω[iω])` — z-dependence enters
  *only* through `k2[iω]=(n(ω;z)ω/c)²`, exactly like free-space's
  `norm_free`/`norm_free_gradient` (`src/NonlinearRHS.jl:966-976`), which
  is the *already-shipped* Phase D.5 precedent that a full z-dependent
  norm array (not just the linop) is portable this way. No Bessel/QDHT
  coupling introduces z-dependence beyond the `k2[iω]` term — verified by
  reading `norm_radial`'s full body, not assumed from the docstring.
- **`native.rs` already has the complete architectural pattern to copy**:
  `is_zdep_free`, `zdep_free_{flength,p0,p1,dens_lut,gamma,omega,omegawin,
  kperp2,sidx,kerr_fac_per_dens,omega0,gamma0,dgamma0,last_z}` fields
  (~native.rs:490-521), `set_free_zdep_params`/`native_set_free_zdep_params`
  FFI (~native.rs:4411, :5841), and `ensure_linop_at`/`ensure_free_norm_at`
  (~native.rs:2651, :2751) recompute methods. A radial version would add
  the same field set under a `zdep_radial_*` prefix (swapping `kperp2`
  for `kr2`, `n_y*n_x` for `n_r`), a `set_radial_zdep_params`/
  `native_set_radial_zdep_params` FFI, and `ensure_linop_at`/
  `ensure_radial_norm_at` branches keyed on `is_radial`.

**What's actually missing, and why it's out of this worktree's zone:**

1. `src/LinearOps.jl` — a new `ZDepLinopRadial` struct + a
   `make_linop_radial_gradient(grid, q, gas, L, p0, p1)` convenience
   constructor, mirroring `ZDepLinopFree`/`make_linop_free_gradient`
   (`src/LinearOps.jl:104-166`) almost line-for-line.
2. `src/NonlinearRHS.jl` — a new `ZDepNormRadial` struct +
   `norm_radial_gradient(grid, q, gas, densf)`, mirroring `ZDepNormFree`/
   `norm_free_gradient` (`src/NonlinearRHS.jl:943-976`) almost
   line-for-line.
3. `amalthea/src/native.rs` — the `zdep_radial_*` struct-field block lives
   in the shared `CpuNativeSim` field region (~lines 200-850), which is
   wider than this task's declared native.rs zone
   (`rhs_radial_env` ~1716-1870 and the radial FFI setters ~5392-5495);
   the new `ensure_linop_at`/`ensure_radial_norm_at` branches and the new
   `native_set_radial_zdep_params` FFI export would also fall partly
   outside those line ranges.
4. `src/RK45.jl` — a new "z-dependent radial" guard section, analogous to
   the existing "z-dependent mode-averaged" block
   (`src/RK45.jl:1415-1454`, just before the radial block this task owns)
   — which is a new section, not an edit inside the declared
   `~1489-1620` radial guard block.

Items 1-2 are hard blockers (this worktree's exclusive zone explicitly
excludes them). Items 3-4 are soft — a future radial-z-dep task's zone
would need to be drawn wider than this one's to cover them cleanly.
**Recommendation for the lead:** the next radial-focused task should claim
`LinearOps.jl` + `NonlinearRHS.jl` + the wider `native.rs` field region +
a new `RK45.jl` z-dep-radial section as one unit (mirroring how Phase D.5
was presumably done as one pass across the free-space equivalent of all
four files), rather than trying to split it the way this task's zone was
split.

### No other out-of-zone dependency was needed for item 1

`native_set_raman_params`'s existing radial-aware sizing (keyed on
`is_radial`, set by `native_set_radial_params`) meant item 1 needed no
change to any file outside this task's declared zone, and no new FFI
symbol at all.
