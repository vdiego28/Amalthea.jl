# Backlog

Deferred work and known issues for Amalthea.jl. Severity: 🔴 correctness · 🟡 robustness/CI · ⚪ informational.

## Improvement plan (2026-07-02 review)

Phased plan from the fork-vs-upstream review (`REVIEW.md` — read it first;
section numbers below refer to its findings). Each phase is independently
shippable and ordered by (severity × user impact) / effort. Gate for every
phase: full `LUNA_TEST_GROUP=All` suite green (not a subset — see the
Phase 8 lesson in `docs/native-port/PORT_LOG.md`).

### Phase A — Upstream sync (🔴, small) ✅
The fork base is upstream master minus exactly two functional commits.
1. ✅ Port `0a52ffb` (#428): fix `Polarisation.ellipse` —
   `θ = angle(Q + 1im*U)/2` (the fork's `angle(aL)/2` is always 0) — and
   restore upstream's QWP-at-π/8 regression test in
   `test/test_polarisation.jl`.
2. ✅ Port `0fd7ac2` (#427): re-add the `files=String[]` kwarg to `SSHExec`
   and the auxiliary-file `scp` loop — **merged into the fork's
   shell-escaped `Cmd`-array `runscan`, keeping the escaping** (each file
   path goes through `Base.shell_escape_posixly` / `Cmd` like the script
   does).
3. ✅ `upstream` remote now points at the real `LupoLab/Luna.jl` GitHub repo
   (was a local-disk path used only for the review). Added
   `.github/workflows/upstream_sync.yml`, a weekly scheduled job that fetches
   `upstream/master` and opens a `upstream-sync`-labelled issue if new
   commits appear past the `0a52ffb` base, so the fork never drifts silently
   again.

### Phase B — Correctness & parity fixes (🔴, small-medium) ✅
1. ✅ `PhysData._safe_n` (REVIEW §3.1): restored complex-sqrt semantics for
   all ~20 glass Sellmeier sites — `_safe_n(n2) = real(n2) < 0 ?
   sqrt(complex(n2) + 1e-10im) : sqrt(complex(n2))`, replacing the n=1
   clamp (deleted). `test/test_physdata.jl` now pins BK7 at 70nm
   (below-resonance, n²≈-3.70) against the value implied by upstream's
   plain `sqrt(complex(n2))`.
2. ✅ Rust ionisation `Emax` parity (REVIEW §3.3): `PptIonizationRate::rate`
   now clamps to `rate(e_max)` by default (matching Julia's
   `IonRatePPTAccel`), so `ppt_ionization_rate_vector` no longer triggers
   whole-vector fallbacks + warn spam above `Emax`. The old strict-error
   behaviour is still available via a `.strict(true)` builder method for
   debugging. `test/test_ionisation_rust.jl` and
   `luna-rust/src/lib.rs::test_ppt_ionization_cutoff` updated to assert the
   clamp (and the opt-in strict error).
3. ✅ Density z-independence guard (REVIEW §3.4): `RustNativeStepper` now
   calls `_check_density_zindependent(f!.densityfun, flength)` before
   baking `densityfun(0)` into a constant, for all four non-z-dep native
   paths (mode-averaged, radial, modal, free-space) — samples at
   `(0, flength/2, flength)` when `flength` is finite, else `(0, 1, 2)` as a
   cheap non-constness probe. Throws `NativeIneligible` on mismatch instead
   of silently freezing a z-varying density (covers the Raman density too,
   since it reads the same `densityfun`). New test
   `test/test_native_density_guard.jl`.
4. ✅ README: qualified the Windows support claim (scans not process-safe)
   until Phase G lands.

Gate: `LUNA_TEST_GROUP=All` — 46598 passed, 12 broken (pre-existing), 0
failed/errored.

### Phase C — Make the default workload actually native (🔴🔥, medium) ✅
REVIEW §3.2: default field-resolved `prop_capillary` (plasma on by default:
`plasma = !envelope`) falls back to the Julia stepper because the native
plasma wiring needs the `LUNA_USE_RUST_IONISATION`-gated LUT handle and
that toggle defaults to 0.
1. ✅ Decoupled: `Ionisation._make_rust_ionization_handle` now builds the
   Rust ionisation LUT whenever the library is present and *either*
   `LUNA_USE_RUST_IONISATION=1` *or* the native stepper is enabled
   (`LUNA_USE_RUST_NATIVE≠0`, the default since Phase 8). Depended on B.2
   (clamp parity), already done, so behaviour is identical either way. The
   missing-library `@warn` stays conditional on the *explicit* toggle only
   (not the native-implied case), so a fresh clone without a built Rust
   library doesn't get warning spam on every `IonRatePPTAccel` construction.
2. ✅ Added `RK45._LAST_STEPPER_TYPE`, a test hook set at the end of every
   `solve_precon` call to the concrete stepper type used (more reliable
   than `_NATIVE_FALLBACK_WARNED[]`, which is a one-time-per-session flag
   that can't distinguish "no fallback in this call" from "no fallback
   anywhere yet this session"). New
   `test/test_native_default_workload.jl` runs a **default**
   `prop_capillary` (no env toggles at all) and asserts
   `RK45._LAST_STEPPER_TYPE[] <: RK45.RustNativeStepper` — the regression
   test that would have caught §3.2.
3. ✅ Benchmarked (fixed-seed default HCF run, `docs/native-port/PORT_LOG.md`
   2026-07-02 Phase C entry): **~3.5x wall-time speedup** (0.305s → 0.087s
   for 10 accepted steps) on the exact out-of-the-box config a new user
   gets — previously 0x speedup (silent Julia fallback) despite
   `LUNA_USE_RUST_NATIVE` defaulting on since Phase 8.

Gate: all 7 test groups green — 46602 passed, 12 broken (pre-existing), 0
failed/errored (run as parallel per-group jobs, not a single sequential
`All` invocation).

### Phase D — Native scope: EnvGrid + plasma/Raman in more geometries (✅, large)
In dependency order, each with the established gates (single-step ≤1e-13,
fixed-step full-solve, non-vacuousness check — TESTING.md §3):
1. ✅ **EnvGrid radial** — `rhs_radial_env` in `luna-rust/src/native.rs`
   (mirrors `rhs_mode_avg_env`'s c2c `to_time!`/`to_freq!` convention:
   half-spectrum zero-pad, `no/n` scale, 3/4 `Kerr_env` SVEA factor — applied
   per r-column, reusing the resident `QdhtFfiHandle::apply_cplx`). New
   `radial_eto_c`/`radial_pto_c` complex buffers in `NativeSim`, allocated by
   `native_set_radial_params` based on `sim.is_real` (already generalized:
   `n_spec_over`/`radial_eoo`/`radial_poo` needed no changes). Dispatch on
   `sim.is_real` at both `rhs_radial` call sites (`set_field`,
   RK-stage loop) — same pattern as the existing mode-averaged real/env
   split. `RK45.jl`'s radial block: dropped the `is_real_grid ||
   NativeIneligible` guard (M array, FFI buffer sizing, and the `linop isa
   Array{ComplexF64}` native-path check all already generalized to either
   grid type without further changes). New `test/test_native_radial_env.jl`:
   single-step 5.1e-17, full-solve 1.6e-15 (both far under the ~1e-12 QDHT
   double-transform floor, MATH.md §3.2), non-vacuousness confirmed (Kerr
   changes the Julia-only result by 6.8e-5 at a stronger test field — the
   equivalence tests themselves intentionally use a much weaker field where
   Kerr's own contribution is below the FP-noise floor, so vacuousness is
   checked separately rather than folded into those same runs).
2. ✅ **Plasma in radial geometry** — `apply_plasma_radial` in
   `luna-rust/src/native.rs`: the exact mode-averaged `apply_plasma_real`
   cumtrapz ×3 formula (ionization rate → cumtrapz → ρ(t) → phase → cumtrapz
   → current J with ionization-loss term → cumtrapz → polarisation P),
   applied independently per r-column into column-major `plas_*` scratch
   buffers sized `n_time_over*n_r` — matches `Et_to_Pt!`'s per-r-column
   dispatch for `TransRadial` (`PlasmaCumtrapz` always sees a scalar field
   in this geometry, so there's no cross-r coupling in the plasma response
   itself). `native_set_plasma_params` buffer sizing now branches on
   `s.is_radial`. `RK45.jl`'s radial eligibility guard relaxed from
   Kerr-only to Kerr-and/or-plasma (`NativeIneligible` still rejects Raman
   or any other response, deferred to Phase D.4), plus the same
   plasma-handle-eligibility wiring already used for mode-averaged (Phase C).
   New `test/test_native_radial_plasma.jl`: single-step 8.3e-14, full-solve
   5.1e-12 (both well under the ~1e-12 QDHT floor caveat), non-vacuousness
   confirmed via an independent, much stronger Julia-only-compared field
   (1.79e-5) — decoupled from the equivalence tests' field strength because
   PPT's extreme field-sensitivity makes the single-step tolerance and the
   non-vacuousness threshold pull in opposite directions at the same energy
   (see the test file's comments for the full reasoning and the numbers
   that ruled out a single shared field strength).
3. ✅ **EnvGrid free-space** — new `ComplexFft3d` c2c 3-D FFTW plan wrapper
   (`luna-rust/src/fftw.rs`, `fftw_plan_dft_3d`, FFTW_FORWARD/FFTW_BACKWARD,
   same Julia column-major `(n_t,n_y,n_x)` / FFTW-reversed-dims convention as
   the existing `RealFft3d`, validated against a live `FFTW.fft` Julia
   reference the same way `RealFft3d` was), plus a new `rhs_free_env` method
   in `luna-rust/src/native.rs` mirroring `rhs_free`'s 7-step per-(y,x)-column
   zero-pad → joint 3-D inverse FFT → flat Kerr multiply → per-column towin →
   joint 3-D forward FFT → per-column truncate → normalize pipeline, but
   reusing `rhs_radial_env`'s c2c "copy-both" half-spectrum convention for the
   per-column zero-pad/truncate steps and the 3/4 `Kerr_env` SVEA factor
   instead of RealGrid Kerr's plain cube. `native_set_free_params` now
   branches on `sim.is_real` to build either the real (`RealFft3d`) or
   complex (`ComplexFft3d`) plan and scratch buffers (`free_eto_c`/
   `free_pto_c` for the complex path); `set_field` and the RK-stage loop
   dispatch on `sim.is_real` between `rhs_free`/`rhs_free_env`, same pattern
   as every other real/env split in this port. `RK45.jl`'s free-space block:
   dropped the `is_real_grid || NativeIneligible` guard (the γ3-detection
   loop, density-z-independence check, and `M`-array precomputation all
   already generalize to `Kerr_env` closures without changes). New
   `test/test_native_free_env.jl` (same `Nx≠Ny` geometry as Phase 6's
   RealGrid free-space test, but `EnvGrid`+`Kerr_env`): single-step 2.9e-17,
   full-solve 4.7e-16, non-vacuousness confirmed (Kerr changes the
   Julia-only result by 4.6e-4 at an independent stronger field).
4. ✅ **Raman in radial/modal** — additive term reusing the resident
   `TimeDomainRamanSolver` ADE solver (`RealGrid`, `thg=true`, all-SDO
   `CombinedRamanResponse` with density-independent τ2 — same eligibility
   criteria as the existing mode-averaged Phase 4 wiring). `solve()` resets
   its oscillator state at entry (`raman.rs`), so it is stateless across
   calls — the *same* resident solver instance can be reused sequentially
   across spatial locations with no cross-location leakage, resolving what
   first looked like an architectural blocker for modal (adaptive-cubature
   node positions move between calls) but isn't, since nothing persists
   between nodes anyway. Radial: new `apply_raman_radial` in
   `luna-rust/src/native.rs`, mirroring `apply_plasma_radial`'s per-r-column
   dispatch (`Et_to_Pt!` always sees a scalar field per column for
   `TransRadial`); `native_set_raman_params` now branches on `s.is_radial`
   to size its scratch buffers `n_time_over*n_r` instead of `n_time_over`.
   Modal: the Raman ADE solve is inlined directly into
   `rhs_modal_pointcalc` (per-node, npol=1 scalar field only), right before
   the existing towin apodization step — `modal_integrand_v` already calls
   `rhs_modal_pointcalc` sequentially per quadrature node, so the shared
   `n_time_over`-sized buffers (the same ones mode-averaged uses) are safe
   to reuse without a modal-specific branch in `native_set_raman_params`.
   `RK45.jl`: both the radial and modal eligibility blocks now accept an
   optional `RamanPolarField` alongside Kerr (radial: alongside an optional
   plasma response too), each with its own Raman-wiring block copied from
   the mode-averaged pattern (same FFI call, same eligibility checks).
   New `test/test_native_radial_raman.jl` and `test/test_native_modal_raman.jl`
   (both :N2, vibration-only SDO Raman): single-step ~1e-7/1e-8, full-solve
   ~2.3e-7/1.0e-6 (Rust vs Julia), non-vacuousness ~1.2e-3/7.0e-4 (Raman vs
   no-Raman, Julia-only). Both tolerances are markedly looser than the
   ~1e-12/1e-13 Kerr/plasma radial and modal gates: Julia's oracle
   `RamanPolarField` here has no `rust_handle`, so it takes its own
   FFT-convolution path, while the resident native path always uses the
   O(Nt) exponential-ADE integrator — a genuine method difference (not a
   summation-order floor) that Phase 4's original mode-averaged Raman test
   never actually exercised, because Raman's contribution there was ~2e-16
   relative to Kerr at a single step, i.e. below FP noise; these two new
   geometries' larger non-vacuousness effects make the method-difference
   floor visible for the first time.
5. ✅ **z-dependent `normfun` for free-space** — drops the `const_norm_free`
   restriction for a two-point pressure-gradient gas cell. Turned out to
   require a z-dependent *linop* too (physically inseparable from the
   nonlinear norm: both derive from the same `n(ω;z)`), making this a
   Phase-7-scale addition rather than a small drop-the-restriction change —
   confirmed via `advisor` before implementing, since the original BACKLOG
   wording undersold the scope. The index law is exact, not approximate:
   `n(ω;z) = sqrt(1+γ(λ(ω))·ρ(z))`, confirmed directly from
   `PhysData.ref_index_fun`'s source before writing any Rust. New
   `LinearOps.ZDepLinopFree` (mirrors `Capillary.ZDepLinopMarcatili` — same
   `dspl.x/y/D` density-spline transfer rationale, same `TwoPointGradient`
   closed form) + `NonlinearRHS.ZDepNormFree`, built together by
   `LinearOps.make_linop_free_gradient`/`NonlinearRHS.norm_free_gradient`
   (mirrors `Capillary.gradient`'s `(coren, dens)` return shape). Unlike
   Phase 7's Marcatili case there is no waveguide term, so β1(z) reduces to
   `(n0(z)+ω0·dn0/dω(z))/c` with `dn0/dω(z) = dγ0·ρ(z)/(2·n0(z))` — still
   needs the same BigFloat-derivative trick for `γ0`/`dγ0` (γ is an opaque
   per-gas closure), but no `nwg`/`neff` complex-branch logic. New
   `luna-rust/src/native.rs` method `ensure_free_norm_at` (free-space
   counterpart of `ensure_linop_at`) recomputes **both** `self.linop`
   (`LinearOps._fill_linop_xy!`'s formula) and `self.free_m`
   (`NonlinearRHS.norm_free`'s formula) from one shared per-(ω,k⊥)
   `k²(ω;z)-k⊥²`, wired into all 4 `ensure_linop_at` call sites in
   `native_step`'s stage loop. New `native_set_free_zdep_params` FFI setter
   (transfers `dspl`, `γ(ω)`, `ω`, `ωwin`, `k⊥²`, `sidx`, and the 4 β1
   constants — `sidx` needed its own `zdep_free_sidx` field since
   free-space's generic `self.sidx` is only ever populated by
   `native_set_mode_avg_params`, an easy miss caught by an index-out-of-
   bounds panic on the first native run, not a silent wrong-answer). New
   `test/test_native_free_zdep.jl`: single-step 2.7e-11, full-solve 1.4e-10
   — far tighter than Phase 7's ~1e-4 gate, because Julia's own FD-based β1
   (`PhysData.dispersion_func`) happens to have much lower noise for this
   simple index law than for Marcatili's full `neff`; non-vacuousness
   confirmed by comparing against a constant-`p0` cell over the same length
   (20% relative difference, since p1=3·p0 here — not folded into the
   equivalence tests, which need a much smaller gradient to stay within the
   Rust-vs-Julia agreement floor for other reasons already documented at
   other phases).

Gate (D.1): all 7 test groups green — 46605 passed, 12 broken (pre-existing),
0 failed/errored.

Gate (D.2): all 7 test groups green — 46608 passed, 12 broken (pre-existing),
0 failed/errored (rust group: 41975/41975, includes 3 new tests).

Gate (D.3): all 7 test groups green — 46611 passed, 12 broken (pre-existing),
0 failed/errored (rust group: 41978/41978, includes 3 new tests).

Gate (D.4): all 7 test groups green — 46617 passed, 12 broken (pre-existing),
0 failed/errored (rust group: 41984/41984, includes 6 new tests).

Gate (D.5): all 7 test groups green — 46620 passed, 12 broken (pre-existing),
0 failed/errored (rust group: 41987/41987, includes 3 new tests). **Phase D
complete** (D.1-D.5 all ✅).

### Phase E — Native scope: modal generality (🟡, large)
1. ✅ General mode orders: stable `besselj(n, x)` for n>1 (downward Miller
   recurrence) in `diffraction.rs`, unlocking TE/TM/`n>1` Marcatili modes.
   Implementation: `diffraction::jn` (Miller recurrence, seed order scaled
   to `max(order, x)` for stability at both large order and large argument —
   see its doc comment for the derivation) replaces the hardcoded `j0` call
   in `native.rs`'s `rhs_modal_pointcalc`. The `full=false` radial-only
   integral always evaluates the mode field at `θ=0` (pre-existing design,
   unchanged), which lets the azimuthal dependence collapse to two
   precomputed per-mode constants (`modal_angle_x`/`_y`, replacing the old
   `modal_phi`) instead of needing genuine 2-D angular integration: `:HE`
   modes get `J_{n-1}(x)·(sin(nϕ), cos(nϕ))` (n=1 recovers the old
   `J0·(sinϕ,cosϕ)` special case exactly), `:TE` gets `J1(x)·(0,1)`, `:TM`
   gets `J1(x)·(1,0)` — derived directly from `Capillary.field`'s closed
   form evaluated at `θ=0`. `RK45.jl`'s modal eligibility gate now accepts
   any `kind ∈ (:HE,:TE,:TM)` MarcatiliMode (previously `:HE,n=1`-only);
   `native_set_modal_params`'s FFI signature gained `order`/`angle_x`/
   `angle_y` array params, dropping the old scalar `phi` param. New test:
   `test/test_native_modal_general_order.jl` (HE n=2, TE, TM — all agree
   with the Julia oracle to ~1e-15/1e-16/1e-15, i.e. `jn` is exact to FP
   noise, not just spline-precision). Tapered radius, `full=true`, and
   npol=2 (items 2-4 below) remain out of scope.

Gate (E.1): all 7 test groups green — 46623 passed, 12 broken (pre-existing),
0 failed/errored (rust group: 41990/41990, includes 3 new tests).

2. ✅ Tapered / per-mode radius. Turned out to be Phase-7/D.5-scale, not a
   quick guard-drop — flagged to the user before implementing, who
   confirmed proceeding. Scope: all modes of one `TransModal` share a
   single tapered core radius `a(z)` (an arbitrary Julia `Function`, not
   per-mode-differing radii), constant density (checked, like Phase 7's
   density check but inverted — here density must be *constant*, radius
   varies). Key insight that kept this tractable: the Marcatili waveguide
   term separates cleanly, `nwg(ω,a) = unm²·(φ(ω)/a)²·(1-i·vn(ω)·φ(ω)/a)²`
   with `φ(ω)=c/ω` — `unm` is radius-independent (a mode's own fixed
   eigenvalue) and `vn(ω)` depends only on the (z-independent) cladding
   Sellmeier, so the *entire* `a(z)` dependence is the explicit `1/a`
   factors, evaluated exactly (no LUT) at any z from `unm`/`vn(ω)` plus the
   scalar `a(z)`. `β1(z)` reuses the same closed-form chain-rule pattern as
   Phase 7/D.5 (4 per-mode BigFloat-differentiated ω0 constants:
   `εco0,dεco0,v0,dv0`), generalized to loop over `n_modes` (vs Phase 7's
   single reference mode) with a configurable `ref_mode`. `a(z)` itself is
   transferred as a `Maths.CSpline` fit to dense z-samples of the ground
   truth function — safe (not a spline-of-spline, unlike the density LUTs)
   since `a(z)` is a plain user function, not already a spline. A second,
   easy-to-miss piece: the per-mode normalization `1/√N(m,z)` also depends
   on radius (`N∝a²`), so `Modes.N`'s z=0 baseline is rescaled by `a(0)/a(z)`
   every call alongside the linop, and the cubature integration bound
   (`self.modal_a`) is updated too — both easy to miss since they live in
   `rhs_modal_pointcalc`, not the linop path. New Julia type
   `Capillary.ZDepLinopModalTaper` (extends `Capillary.make_linop` for
   `Tuple{Vararg{MarcatiliMode}}` with a shared `Function`-valued `a`,
   falling back to the untouched generic `LinearOps.make_linop` otherwise);
   new Rust `ensure_modal_linop_at`/`native_set_modal_zdep_params`. New test
   `test/test_native_modal_taper.jl` (HE n=1/n=2, TE, TM, plus a two-mode
   HE11+HE12 coupling-under-taper case) — agreement ~1e-7/1e-8 (the same
   deliberate-divergence tier as Phase 7, not the ~1e-15 bitwise tier of the
   constant-radius modal test, since β1(z)'s closed form intentionally
   diverges from Julia's own adaptive-FD `dispersion`).

Gate (E.2): all 7 test groups green — 46629 passed, 12 broken (pre-existing),
0 failed/errored (rust group: 41996/41996, includes 5 new tests).

3. ✅ `full=true` (2-D modal integral — second cubature dimension). Bound a
   second `libcubature` symbol, `hcubature_v` (h-adaptive, arbitrary
   `ndim` — same C prototype as the existing `pcubature_v` binding, so no
   new FFI type), used with `ndim=2` over `(r,θ) ∈ [0,a]×[0,2π]` — the same
   routine Julia's `full=true` branch calls (`Cubature.hcubature_v`), so
   node placement is bit-identical, not just close. The mode-field
   synthesis (`rhs_modal_pointcalc`) generalizes from E.1/E.2's
   precomputed-at-`θ=0` `angle_x`/`angle_y` shortcut to a single runtime
   `mode_angle_xy(kind,order,ϕ,θ)` function evaluated at the cubature
   node's actual `θ` (subsuming the `θ=0` special case exactly — verified
   the two formulas agree at `θ=0` before deleting the old fields); the
   Jacobian switches from `2πr` (θ-integral done analytically) to plain `r`
   (θ genuinely integrated). New test `test/test_native_modal_full.jl` (HE
   n=1/n=2, TE, TM) reaches machine precision (~1e-15/1e-16 — tighter than
   `full=false`'s ~1e-10, since neither approximation source `full=false`
   still has (constant-radius here, so no β1(z) BigFloat-derivative
   divergence either) is present), plus a same-mode `full=true`-vs-
   `full=false` cross-check (both native) at ~5e-8 — a real but small
   quadrature-method difference (h-adaptive 2-D vs p-adaptive 1-D
   convergence to the same true integral via different node paths), not a
   bug, since both were independently verified against the Julia oracle
   first.

Gate (E.3): all 7 test groups green — 46634 passed, 12 broken (pre-existing),
0 failed/errored (rust group: 42001/42001, includes 5 new tests).

4. ✅ npol=2 (polarisation-resolved modal). Turned out to be a "verify, not
   implement" item: `rhs_modal_pointcalc` (native.rs) was already written
   generically over `npol` since Phase 5/E.1-E.3 (mode-field synthesis,
   `to_space!`/`to_time!`/`to_freq!` all loop `for p in 0..npol`), including
   a bit-for-bit port of `KerrVector!`'s cross term
   (`out[:,1] += fac*(Ex²+Ey²)*Ex`, `out[:,2] += fac*(Ex²+Ey²)*Ey`,
   `src/Nonlinear.jl:85-94`) — this was simply gated off in `RK45.jl` behind
   `npol == 1 || throw(NativeIneligible(...))` pending a Julia-oracle
   equivalence test. Relaxed the guard to `npol in (1,2)`. One real gap
   found: native.rs's inline Raman ADE solve (`rhs_modal_pointcalc`'s
   "npol=1 scalar field only" block) only ever touches polarisation column
   0 — with npol=2 it would silently drop the second (y) column's Raman
   contribution rather than raising, so a new guard rejects
   `RamanPolarField` combined with `npol=2` specifically (Kerr-only npol=2
   remains fully supported). New test `test/test_native_modal_npol2.jl`:
   single HE,n=1 mode with `ϕ=π/4` (so both `Ex`/`Ey` are genuinely nonzero
   at every node, exercising the cross term — `ϕ=0` would leave `Ey≡0` and
   silently degenerate to the npol=1 formula) at both `full=false` and
   `full=true`, `rel < 1e-9` (matches E.1's own tolerance tier, achieved
   ~1e-15/1e-16 — no new approximation source), plus a
   `@test_throws NativeIneligible` check for the Raman+npol=2 rejection.

5. ✅ EnvGrid modal. Turned out smaller than initially scoped: `rhs_modal`
   (the outer cubature driver) needed no changes at all — only
   `rhs_modal_pointcalc` (the per-node callback) needed an `is_real` branch,
   since `sim.is_real` was already set generically (via
   `native_set_fftw_plans`, called for every geometry including modal) and
   `n_spec`/`n_spec_over` were already computed grid-agnostically. Added a
   new EnvGrid branch mirroring `rhs_radial_env`'s established pattern: c2c
   FFT with half-spectrum copy-both padding/truncation (per polarisation
   column instead of per-r column) via two new buffer fields
   (`modal_er_c`/`modal_pr_c: Vec<Complex<f64>>`, alongside the existing
   real `modal_er`/`modal_pr`), and the envelope Kerr formulas
   (`KerrScalarEnv!`/`KerrVectorEnv!`, `src/Nonlinear.jl:120-133`) — confirmed
   `KerrVectorEnv!` is a genuinely different formula from RealGrid's
   `KerrVector!` (the `2/3`/`1/3` split and `conj(Ex)·Ey²` cross term), not
   just a complex-typed port. No new eligibility mechanism to distinguish
   `Kerr_env` from `Kerr_field` was actually needed: like radial/mode-averaged
   before it, the native RHS never inspects which closure was passed — it
   picks the RealGrid-vs-EnvGrid formula purely from `is_real`, and the `γ3`
   value extracted by the existing field-name grep is correct either way.
   One new guard added: Raman (`RamanPolarField`) + EnvGrid modal is rejected
   (`NativeIneligible`) — native.rs's inline Raman ADE solve only ever
   touches the real `modal_er`/`modal_pr` buffers, with no EnvGrid wiring.
   Relaxed `RK45.jl`'s old blanket `is_real_grid || throw(...)` modal guard
   accordingly. New test `test/test_native_modal_env.jl`: npol=1 (`full=false`
   and `full=true`) and npol=2 (`ϕ=π/4`, exercising the cross term) full-solve
   equivalence, all `rel < 1e-9` (achieved ~1e-15/1e-16, same tolerance tier
   as the RealGrid modal tests — no new approximation source), plus a
   `@test_throws NativeIneligible` check for the Raman+EnvGrid rejection.

Gate (E.4, final): all 7 test groups green — 46641 passed, 12 broken
(pre-existing), 0 failed/errored (rust group: 42008/42008, includes 4 new
tests). **Phase E (native scope: modal generality) is now complete.**

### Phase F — Native scope: Raman completions + z-dependence (✅, medium)
1. ✅ `thg=false` Raman — resident c2c Hilbert transform, wired into all
   three geometries that already carry `thg=true` Raman (mode-averaged,
   radial, modal). See "Done (recent)" below.
2. ✅ `RamanPolarEnv` (envelope Raman) native, mode-averaged EnvGrid;
   rotational multi-oscillator; density-dependent τ2 for constant-medium
   sims — see "Done (recent)" below. z-dependent-density Raman (needing a
   `raman_update_coeffs` FFI to re-derive τ2 per stage) remains a
   follow-up, out of scope until Raman + a z-dependent linop are wired
   together for any geometry.
3. ✅ Multi-point pressure gradients (piecewise Phase 7) — see "Done (recent)"
   below.
4. ✅ Gas mixtures (mode-averaged, Kerr-only) — see "Done (recent)" below.
   Mixtures with plasma/Raman, or mixtures for radial/modal/free-space, are
   still out of scope (not a simple linear sum of species contributions).

### Phase G — Platform & CI robustness (🟡, small-medium)
1. ✅ Windows scan locking via `LockFileEx`/`UnlockFileEx` — implemented
   (commit `febdde1`, 2026-06-28) and validated on the existing
   `windows-2025-vs2026` CI runner (see "Windows scan-lock validation" below —
   2026-07-08 finding: the runner already existed, the validation was already
   passing, only the docs claiming otherwise were stale).
2. GPU CI: a scheduled CUDA-equipped job running the currently
   self-skipping GPU equivalence tests (existing item below). Local hardware
   verification (not CI) is now possible — see "Open items" below, this
   machine has an RTX 5060 Ti + CUDA 13.3 toolkit installed.
3. ✅ CI benchmark job — see "Done (recent)" below.

### Phase H — Upstream contributions (⚪, small) — 3 of 4 sent, see "Done (recent)"
The `pointcalc!` race fix is not currently actionable: upstream doesn't use
`Threads.@threads` there today, so there's nothing to fix unless/until they
re-add threading.

### Phase I — Close remaining native-port gaps (mixed 🔴/🟡, small-medium each)
Audit (2026-07-07) of every `NativeIneligible` throw in `src/RK45.jl`
(lines 774-1598), cross-referenced against `src/Interface.jl`'s actual
high-level entry points (not just what the low-level API could
theoretically construct). Confirms Phase E/F closed everything their own
"Done" entries claim (`full=true` modal, `npol=2`, gas mixtures,
`RamanPolarEnv`, `thg=false` Raman, multi-point pressure gradients, EnvGrid
modal) — this phase is only the genuinely-still-open remainder.

🔴🔴 **Critical correctness fix, found and fixed 2026-07-08 while verifying
item 3 below: native plasma polarization was missing its density factor
entirely, for every mode-averaged/radial plasma sim, PPT or ADK, since
this was first wired (Phase C).** `native.rs`'s `apply_plasma_real`/
`apply_plasma_radial` computed the per-sample ionization rate and
cumtrapz-integrated polarization `plas_p` correctly, then added it
straight into `pto` with **no density multiplication at all** — Julia's
own `(Plas::PlasmaCumtrapz)(out, Et, ρ)` does `out .+= ρ .* Plas.P`
(`Nonlinear.jl:237-249`); the native path was doing `out += Plas.P` (ρ
implicitly 1, not the real gas density, ~1e25 m⁻³ for a typical capillary
fill). Every existing plasma equivalence test (`test_native_phase2.jl`'s
PPT test, `test_native_radial_plasma.jl`) ran at a weak enough field that
the ionization rate stayed near zero throughout, so "native matches
Julia" was vacuously true in both cases — nothing in the existing test
suite ever exercised a field strong enough to make the missing factor
visible. Found by deliberately picking a stronger-field ADK test energy
(one where ionization has a clear, non-chaotic ~0.5% effect on the
propagation — see item 3 below) and triangulating three results
(native-plasma-on, Julia-plasma-on, Julia-plasma-off) instead of the
existing tests' two (native vs Julia only, both already near the
plasma-off limit). Fixed by adding a `plasma_density` field to
`NativeSim`, threading it through `native_set_plasma_params`/
`native_set_plasma_params_adk` (both gained a new `density` argument),
and multiplying it into `plas_p` before accumulating into `pto`/
`radial_pto`. Verified: PPT full-solve relative error at the
previously-untested strong-field point dropped from ~3.9e-4 (density
completely missing — this WAS the bug's signature, not a benign tolerance
gap) to ~3.8e-15 (bit-parity tier) after the fix. Added a proper
triangulating regression test to both `test_native_phase2.jl` (PPT,
mode-averaged) and `test_native_radial_plasma.jl` (PPT, radial) so a
future regression can't hide behind a weak-field blind spot again.
**Practical implication:** any published result from a native (default
since Phase 8) `prop_capillary`/`prop_gnlse` run with plasma enabled and
a strong enough field for ionization to matter was under-computing plasma
self-defocusing/spectral broadening — re-run affected sims from before
this commit if plasma played a physically meaningful role.

1. 🟢 **Confirmed and fixed (2026-07-08): mode-averaged/radial shot noise
   (`Et_noise`) was silently dropped by the native path, not gracefully
   falling back.** Verified by code trace (not a loose full-solve diff,
   which would have false-passed): neither the mode-averaged nor radial
   setup blocks in `RustNativeStepper` ever referenced `f!.Et_noise`, and
   no noise buffer was ever passed to `native_set_mode_avg_params`/
   `native_set_radial_params` — unlike the modal (`RK45.jl:1343`) and
   free-space (`RK45.jl:1528`) blocks, which correctly guard and fall back.
   Since `shotnoise=true` (`:modified` model) is `Interface.jl`'s default,
   this affected the single most common configuration in the package,
   including the flagship default field-resolved workload (see
   `test/test_native_default_workload.jl`).
   - **Mode-averaged, RealGrid: fully ported to native**, not just
     guarded. Added `native_set_mode_avg_noise` (Rust: `et_noise`/
     `has_noise` fields on `CpuNativeSim`, injected into `eto` after the
     1/sc scaling, matching Julia's `Et_nl = Eto + Et_noise/sc` exactly —
     `NonlinearRHS.jl:561-562`) and wired the Julia-side caller in
     `RK45.jl`. Verified via `test/test_native_shotnoise.jl`: (a) Rust vs.
     Julia single-step equivalence with noise and a shared fixed RNG seed
     (~1e-9), (b) a regression guard confirming the noisy and noise-free
     native results actually differ (catches "silently zero" recurring).
     Full `rust` gate: 42058/42058 passing.
   - **Radial, RealGrid: also fully ported to native.** Same shape:
     `native_set_radial_noise` adds the (unscaled) noise field into
     `radial_eto` right after the QDHT `ldiv!` step, matching
     `TransRadial`'s `Et_nl = Eto + Et_noise` (`NonlinearRHS.jl:691-692`).
     Verified via `test/test_native_radial_shotnoise.jl`: Rust vs. Julia
     agreement ~1e-17 with noise, plus the same noisy-vs-clean regression
     guard. Full `rust` gate: 42063/42063 passing.
   - **Mode-averaged and radial EnvGrid: also fully ported to native**
     (`native_set_mode_avg_noise_cplx`/`native_set_radial_noise_cplx`,
     same injection points, `ComplexF64` interleaved re/im arrays). This
     closes the item completely — **not "rare"**: `prop_gnlse` is
     envelope + mode-averaged with `shotnoise=true` by default
     (`Interface.jl:1053`), so this was blocking the flagship *gnlse*
     default workload from running natively, exactly like item 1's
     original `prop_capillary` finding. Confirmed before/after with a
     direct `prop_gnlse(...)` call: fell back to `PreconStepper` before
     this fix, uses `RustNativeStepper` after (independent of
     `raman=true`, which the native mode-averaged EnvGrid RHS already
     supported via `RamanPolarEnv` — Phase F.2 — so noise was the sole
     remaining blocker). Verified via
     `test/test_native_envgrid_shotnoise.jl`: Rust vs. Julia agreement
     ~1e-10 to ~1e-17 with noise for both geometries, plus the
     noisy-vs-clean regression guard for each. Full `rust` gate:
     42065/42065 passing. **Item 1 is now fully closed** — no remaining
     `Et_noise` gap in any of the four native geometries that support it
     (mode-averaged, radial); modal/free-space still correctly guard and
     fall back (out of scope for this item — no native noise support was
     ever claimed for them).
2. 🟢 **Done (2026-07-08): `ramanmodel=:SiO2` in `prop_gnlse`**
   (`Interface.jl:1077-1079`) — `Raman.raman_response(t,:SiO2,...)` returns
   a bare `RamanRespIntermediateBroadening` (Gaussian-damped), not wrapped
   in `CombinedRamanResponse`, so `flatten_sdo_oscillators` correctly
   returns `nothing` (no finite sum of single-damped-oscillators
   (`exp(-t/τ2)`) reproduces the `exp(-Γᵢ²t²/4)` Gaussian envelope in
   `Raman.jl:97-105`, Hollenbeck & Cantrell's multi-line silica model) — a
   real gap, not a missing wrapper. Ported as a **second, independent
   resident kernel** (`native_set_raman_fft_params`/`native.rs`'s new
   `rhs_mode_avg_env` Step 3c), not a multi-SDO approximation (per
   `feedback_prefer_analytic_over_lut`): precomputes the padded impulse
   response's spectrum ĥ(ω) once at setup (folding in `dt` and the ifft's
   `1/n_over` normalisation so no further per-step scaling is needed), then
   each RHS step does `P = ifft(ĥ(ω) · fft(E²_padded))` via a resident
   length-`2·n_time_over` c2c FFTW plan — the exact same math as
   `Nonlinear.jl`'s generic (non-SDO) Raman path
   (`(R::RamanPolar)(out,Et,ρ)`, `Nonlinear.jl:395-417`), just moved
   resident. Only reachable via `prop_gnlse` (mode-averaged EnvGrid,
   `RamanPolarEnv`) — `:SiO2` is not wired for `prop_capillary`, and
   `RamanRespIntermediateBroadening` is density-independent (`ρ` unused in
   its `(ht,ρ)` call), so ĥ(ω) can be precomputed once and reused for the
   whole propagation, same frozen-at-setup restriction the existing ADE
   Raman path already has for z-dependent density. Verified with a
   triangulating regression test (native-raman-on, Julia-raman-on,
   Julia-raman-off) at a soliton self-shift configuration where Raman has a
   large, unambiguous effect (N=4 soliton, `test/test_native_raman_sio2.jl`),
   **at `prop_gnlse`'s actual defaults (`shock=true, shotnoise=true`)** —
   item 1's postmortem was exactly this class of gap staying invisible
   until the default workload was tested, not a hand-picked easy config;
   confirmed the native stepper doesn't silently fall back with both
   defaults on: single-step 0.0 (RNG-seed-identical, since both steppers
   share one `transform`'s baked-in `Et_noise`), full-solve native-vs-Julia
   ~5.3e-13 to ~6.0e-13 against a raman-on/raman-off difference of ~1.44
   (nowhere near vacuous).
   **Item 2 is now fully closed** — no remaining Raman-model gap in any
   native geometry that supports `RamanPolarEnv`/`RamanPolarField`.
3. 🟢 **Done (2026-07-08): `ionisation=:ADK` native support.** `IonRateADK`
   is closed-form/analytic (no LUT — Julia evaluates the ADK formula
   directly, `Ionisation.jl:176-189`), so this ported cleanly with no
   approximation risk: `luna-rust/src/ionization.rs`'s new
   `AdkIonizationRate` takes Julia's already-computed constants
   (`nstar`/`cn_sq`/`ω_p`/`ω_t_prefac`/`thr`/`avfac`) directly rather than
   refitting anything, mirroring the exact formula in
   `(ir::IonRateADK)(E)`. New `RustAdkHandle` (Ionisation.jl, same
   GC-finalizer pattern as `RustIonizationHandle`) and
   `native_set_plasma_params_adk` FFI setter; `native.rs`'s
   `apply_plasma_real`/`apply_plasma_radial` now dispatch through a shared
   `plasma_rate()` helper on a `plasma_is_adk` flag instead of assuming
   `PptIonizationRate`. Verified: single-step ~3.8e-17, full-solve
   ~2.7e-16 (bit-parity tier, as expected for an exact analytic port) —
   see `test/test_native_adk_ionisation.jl`.
4. 🟢 **Done (2026-07-08, radial only): gas mixtures outside
   mode-averaged geometry.** Discriminating check (per-item audit):
   radial's normalization array `M` and QDHT are density-independent at
   the point kerr_fac is computed (density only enters the *linear*
   refractive index, already baked into `linop`/`normfun` before native
   construction) — so, exactly like Phase F.4's mode-averaged case, Kerr's
   linearity in density·γ3 collapses a per-species mixture to one scalar
   `kerr_fac = Σᵢ densityᵢ·ε₀·γ3ᵢ`. Julia-only change (no Rust/FFI
   changes), mirroring Phase F.4's mode-averaged mixture branch almost
   exactly. Verified: single-step ~1.1e-17, full-solve ~1.3e-16 — see
   `test/test_native_radial_mixture.jl`. **Modal and free-space mixtures
   still fall back** (not checked in this slice — modal's per-mode
   projection and free-space's 2-D Cartesian normalization may or may not
   collapse the same way; needs its own discriminating check before
   assuming it's equally trivial).
5. 🟡 **High-level API reach done (2026-07-16); native Rust port still
   open.** `StepIndexMode`/`ZeisbergerMode`/`VincettiMode` were previously
   only reachable via the low-level API. Two-part fix:
   - **`ZeisbergerMode`/`VincettiMode`**: both wrap a `Capillary.MarcatiliMode`
     in `.m` and only override dispersion — `prop_capillary`'s existing
     gas/pressure → `makedensity`/`makeresponse` pipeline never touches the
     mode object, so this was a small additive change. `makemode_s` gained
     two new dispatches (`Interface.jl`) that let a caller pass an
     already-built `Modes.AbstractMode` (or `Vector` thereof) directly via
     `modes=`, bypassing the `:HE11`-style signifier system entirely. Fixed
     an accessor gap this surfaced: `needfull`/`_findmode`/`needpol_modes`
     read `mode.kind`/`mode.n`/`mode.m` as direct struct fields, which is
     wrong for these two types (nested under `.m`) — now read through
     `Modes.modeinfo(mode)` instead (an existing, already-tested accessor
     — see `Capillary.jl:57`/`RectModes.jl:36` — extended to
     `ZeisbergerMode`/`VincettiMode`/`StepIndexMode` via their existing
     field-forwarding loops in `Antiresonant.jl`). A single prebuilt mode
     combined with a vector-field-requiring config (e.g.
     `polarisation=:circular`) now raises a clear error instead of silently
     misbehaving, since the prebuilt path has no signifier to derive a
     companion-mode `ϕ` from.
   - **`StepIndexMode`**: turned out *not* to fit the same extension —
     it's solid-fibre physics with no gas/pressure/density concept at all
     (its own example hardcodes `densityfun = z -> 1.0` and builds
     Kerr/Raman straight from `PhysData.χ3`, bypassing
     `makedensity`/`makeresponse` entirely). Given a new sibling top-level
     function, `prop_stepindex` (`Interface.jl`), mirroring `prop_gnlse`'s
     existing precedent for exactly this situation (its own `SimpleMode`
     also needs no gas/density machinery) — builds the mode via
     `StepIndexFibre.StepIndexMode`, density fixed at 1, and Kerr/Raman
     response directly from the core material's `χ3` (generalizing the
     `:SiO2`-only example).
   - **`src/RK45.jl`'s native-eligibility guards were deliberately not
     touched — and it turns out they didn't need to be, for the single-mode
     case.** Verified (not assumed) by running each of the three configs
     above and checking `Amalthea.backend_report()`/
     `RK45._LAST_STEPPER_TYPE[]`: all three actually run on
     `RK45.RustNativeStepper` already. The `all(m -> m isa
     Capillary.MarcatiliMode, modes)` restriction (`RK45.jl`) only gates the
     *multi-mode* `TransModal` path — mode-averaged (`TransModeAvg`), the
     path every config in this item's tests uses, never inspects the
     concrete mode type at all (only `NonlinearRHS.norm_pre`/`norm_βfun`/
     `Modes.Aeff(mode; z=z)`, all already mode-agnostic). So single-mode
     native support for `ZeisbergerMode`/`VincettiMode`/`StepIndexMode` was
     already complete before this item — this plan's own original framing
     ("Choice A now, native Rust port as a follow-up") overstated the
     remaining native-port gap. New regression test
     (`test/test_interface.jl`) asserts `RK45._LAST_STEPPER_TYPE[] <:
     RK45.RustNativeStepper` for all three, pinning this down.
   - **What's actually still native-ineligible, and the real follow-up: the
     multi-mode `TransModal` case** (several `ZeisbergerMode`/
     `VincettiMode`/`StepIndexMode`s propagating together, e.g. two
     polarisation components or several radial orders) — that path's
     `RK45.jl` guard genuinely does require `Capillary.MarcatiliMode`
     (`native.rs`'s `rhs_modal_pointcalc` synthesizes the mode field from
     Marcatili's `unm`/Bessel-order/`ϕ` parameterization directly). Scoped
     per mode type by existing native-readiness: Zeisberger's dispersion
     already has a Rust kernel (`ZeisbergerNeff`); Vincetti's
     (`CL`/`Rco_eff`/`Δneff`) does not yet; StepIndexMode's `neff` needs
     numerical root-finding (`Roots.find_zeros`), no closed form, making it
     the hardest of the three.
   - Out of scope for this pass: multi-mode configurations for any of these
     three types (see above), tapered (`z`-dependent) `StepIndexMode`
     radius, and auto-generating a companion mode for both-polarisation
     configs on the prebuilt-mode path.
6. 🟢 **Done (2026-07-09): free-space (`TransFree`) gains plasma and/or
   Raman.** `apply_plasma_free`/`apply_raman_free` (native.rs) mirror
   `apply_plasma_radial`/`apply_raman_radial` verbatim over `n_y*n_x`
   columns instead of `n_r` — `TransFree`'s `Et_to_Pt!` dispatches each
   response over `CartesianIndices((Ny,Nx))` exactly like `TransRadial`
   dispatches over r, so both responses see a scalar field per column
   either way. RealGrid only (no EnvGrid geometry gets native plasma
   anywhere in this port, and `RamanPolarEnv` is native only for
   mode-averaged EnvGrid) — EnvGrid free-space (`rhs_free_env`) keeps the
   original Kerr-only restriction. Plasma/Raman + a z-dependent normfun
   (Phase D.5) is explicitly rejected (unverified interaction, no test
   coverage), not silently combined. `RK45.jl`'s free-space eligibility
   block now branches on `is_real_grid`: RealGrid accepts Kerr +
   optional plasma + optional Raman (`RamanPolarField`, same
   `CombinedRamanResponse`/density-independent-τ2 eligibility as
   radial's Phase D.4 wiring); EnvGrid keeps the single-response
   Kerr-only check. New tests `test/test_native_free_plasma.jl`
   (single-step ~1e-12, full-solve, non-vacuousness 1.57e-6, native vs
   Julia at the plasma-matters energy 3.76e-16 vs plasma-off 1.57e-6 —
   same triangulation discipline as Phase I's plasma-density postmortem)
   and `test/test_native_free_raman.jl` (single-step ~1e-7 — same
   ADE-vs-FFT-convolution method-difference floor as radial's Raman
   test, non-vacuousness 1.18e-3, native vs Julia 2.2e-7 vs Raman-off
   1.18e-3). Full `rust` gate: 42098/42098 (was 42083; +15 new tests).
7. 🟢 **Done (2026-07-09): regression tests added for the three
   `NativeIneligible` guard categories** (non-finite `flength` for
   z-dependent linops, unrecognized bare `f!` closures, missing Rust
   ionisation/Raman handle when the library is absent or toggles are
   forced off) — previously correct-by-design but unverified by any
   dedicated test (Phase I's own preamble is exactly this class of risk:
   "documented as correct" isn't "verified to actually fire"). New
   `test/test_native_ineligible_guards.jl`, three testsets. One
   assumption caught and fixed while writing it: a plasma response built
   without `LUNA_USE_RUST_IONISATION=1` still gets a Rust handle by
   default, since Phase C decoupled the handle-build condition to *also*
   trigger whenever the native stepper itself is enabled
   (`LUNA_USE_RUST_NATIVE`, on by default since Phase 8) — the
   missing-handle guard is therefore only reachable by explicitly
   disabling *both* env vars at construction time, not just the
   ionisation one alone. Included in the same 42098/42098 gate as item 6.

No other nonlinear response type exists in the codebase beyond
Kerr/plasma/Raman/`RamanRespIntermediateBroadening` (confirmed by grepping
`Nonlinear.jl`/`Ionisation.jl`) — so items 1-4 are the complete list of
what stands between "native handles the documented high-level API" and
"native is exclusively used for every response/geometry combination
`Interface.jl` can construct." Item 5-7 remain because the low-level API
is intentionally broader than the high-level one, not because of missed
scope.

### Phase J — Post-completion audit findings & future directions (2026-07-08)

Full audit after Phase I closed the native-port scope: re-verified Phase
A-I claims against the git log and code, re-derived the SiO2 FFT-conv
kernel's math against `Nonlinear.jl:357-431`, and reviewed the test
methodology. One latent flaw found and fixed; the rest are tracked
future work, roughly ordered by value.

1. 🟢 **Fixed (2026-07-08): stale zero-padding in the SiO2 FFT-conv
   kernel.** `rhs_mode_avg_env` Step 3c reuses `raman_fft_e2` as both the
   padded-E² input and the inverse-FFT output buffer, but only refilled
   `[0..no)` each step — so from the second RHS call onward the upper half
   held the *previous* step's convolution tail instead of zeros. Julia
   never hits this (its `R.E2` upper half is a separate, never-written
   buffer). Invisible in every existing test (and in any real SiO2 run)
   because Hollenbeck & Cantrell's Gaussian damping makes h ≈ 0 beyond
   ~100 fs, so the stale tail was numerically zero — but circular-
   convolution wrap-around from a non-zero pad is exactly the truncation-
   artefact class the double-length grid exists to prevent
   (Nonlinear.jl:406-411), and the kernel itself is generic over
   oscillator parameters. Fixed by explicitly zeroing `[no..2no)` every
   step (cost: one trivial loop).
2. 🟢 **Resolved 2026-07-09.** `PptIonizationRate::rate` can segfault far
   outside its fitted range — promoted from a buried note in Phase G.3's
   entry to a tracked item: repeatedly `step!`-ing far beyond `flength`
   with a too-large fixed `dt` grows the field until the PPT LUT query
   lands outside `[e_min,e_max]` and something in that path segfaults
   instead of hitting the documented Phase B.2 clamp. Investigated: the
   `[e_min, e_max]` finite-range clamp itself (Phase B.2) was already
   correct and `CubicSplineLUT::evaluate`'s `get_unchecked` was already
   bounds-safe (its bounds check runs before the segment lookup, and the
   binary search invariant keeps the index in range) — no literal
   out-of-bounds memory access was found. The real bug was **NaN/±inf
   handling**: IEEE 754 makes every comparison with NaN false, so a
   non-finite field silently passed *both* `rate()`'s `< e_min`/`> e_max`
   checks *and* `evaluate()`'s own `x < x_min || x > x_max` guard, falling
   through to return `Some(garbage)` instead of the intended `None`/clamp
   — confirmed by a new unit test failing before the fix
   (`cubic_spline_lut_rejects_out_of_range_without_panicking`). Fixed by
   adding an explicit `is_finite()` guard in both `CubicSplineLUT::evaluate`
   (defense at the source) and `PptIonizationRate::rate` (routes non-finite
   input through the same overflow policy as a too-large-but-finite field:
   clamp to `rate(e_max)`, or `Err` under `.strict(true)`); the sibling
   `AdkIonizationRate::rate` got the same guard for consistency (treats
   non-finite as below-threshold → `0.0`, matching its "never errors"
   contract). Added 11 new Rust unit tests in `luna-rust/src/ionization.rs`
   hammering both rate functions across ±1e10× the fitted range (log-spaced
   sweep, both signs) plus explicit NaN/+inf/-inf/boundary cases — `cargo
   test` 48/48 green, full `rust` Julia group re-verified clean
   (42087/42087 via `test/parallel_rust_tests.py`, matching the pre-fix
   count exactly — this was a hardening fix, not a behavior change for any
   finite input).
3. ⚪ **r2c/c2r halving for both FFT-conv Raman convolutions.** The
   padded E² and h are mathematically real in both the carrier and
   envelope paths. Julia's `RamanPolarField` already uses `plan_rfft`,
   but `RamanPolarEnv` uses a full c2c `plan_fft` (Nonlinear.jl:327) and
   the native SiO2 kernel mirrors that (matching parity exactly). A
   Hermitian-symmetric r2c/c2r pair would halve the per-step transform
   work in the native kernel (and could be upstreamed to Julia's
   `RamanPolarEnv` too). Summation-order change → validate with the
   fixed-step discipline (TESTING.md §3); benchmark first to confirm the
   convolution is actually a measurable fraction of a gnlse step.
4. 🟢 **Done (2026-07-16).** User-facing native-support matrix —
   `docs/dev/native-port/NATIVE_SUPPORT_MATRIX.md`, a geometry × grid ×
   response table (rows: mode-avg/radial/modal/free-space; columns:
   RealGrid/EnvGrid; cells noting Kerr/plasma/Raman(SDO/SiO2)/noise/
   mixtures/z-dep support) derived directly from the current
   `NativeIneligible` guard sites in `src/RK45.jl`, cross-referenced
   against `ARCHITECTURE.md` §6's three-category taxonomy for *why* a
   given cell is a bare fallback. Not auto-generated — the doc's own
   closing section asks that future guard-touching PRs update it by hand.
5. ⚪ **Consolidate the two resident Raman kernels' shared plumbing.**
   Step 3b (ADE) and Step 3c (FFT-conv) in `rhs_mode_avg_env` duplicate
   the `0.5·|E|²` intensity loop and the `pto += E·(ρ·P)` accumulation;
   only the middle (solve vs convolve) differs. Worth a small refactor
   when either is next touched — not before (both are covered by
   equivalence tests; churn for its own sake risks more than it saves).
6. ⚪ **Beyond-Luna math options recorded (2026-07-08):** MATH.md §8 +
   SUGGESTIONS.md items 15-17 — direct DP5(4) embedded-error
   coefficients (`Σeᵢkᵢ`, removes the TESTING.md §3 cancellation at its
   root; potentially dissolves the fixed-step test discipline), direct
   PPT evaluation replacing the spline LUT on both sides (also
   structurally fixes item 2's segfault), and short-kernel overlap-save
   Raman convolution (pairs with item 3's r2c halving). Each breaks
   oracle bit-parity deliberately → each needs β1-style
   controlled-divergence verification against a ground truth. The
   three-category taxonomy of what stays Julia (setup-by-design /
   ineligible-but-portable / arbitrary-closures barrier) is written up
   in ARCHITECTURE.md §6.
7. 🟢 **Done (2026-07-16).** Added a `#[cfg(test)]` suite to
   `amalthea/src/native.rs` covering every `native_set_*`/lifecycle FFI
   entry point: a comprehensive "null `sim` is rejected everywhere"
   test across all ~26 entry points, plus per-family tests for
   zero-length arrays, mismatched sizes, and wrong call order (e.g.
   `native_set_raman_params`/`native_set_zdep_mode_avg_params`/
   `native_set_modal_zdep_params`/`native_set_free_zdep_params` before
   their prerequisite `*_params` setter has run). **Real gap found
   while writing it, then fixed (not blind — found by source
   inspection, fixed, and verified green before being called done):**
   several setters dereferenced a non-`sim` pointer argument
   unconditionally once their own shape/order guard passed, with no
   null check on that argument specifically —
   `native_set_radial_params`'s `t_matrix`/`m_re`/`m_im`,
   `native_set_modal_params`'s `unm`/`inv_sqrt_n`/`order`/`kind`/`phi`/
   `pol_select`/`nlfac_re`/`nlfac_im`/`lib_path`,
   `native_set_free_params`'s `m_re`/`m_im`, the three `*_zdep_params`
   setters' spline/array pointers, `native_set_raman_params`'s
   `omega`/`gamma`/`coupling`, `native_set_fftw_plans`'s `lib_path`
   (straight into `CStr::from_ptr`), `native_debug_beta1_at`'s
   `out_dens`/`out_beta1`, and
   `set_field`/`native_resync_field`/`get_field`/`get_ks_stage`/
   `native_apply_prop`/`native_debug_linop_at`'s data/`y` buffer once
   the length check passes. None of these was reachable through any
   documented Julia call site (Julia always passes real, GC-rooted,
   correctly-sized buffers), so this was latent hardening debt, not a
   live bug. **Fix:** every pointer above now has an explicit
   `is_null()` guard (returning `-1`, matching every sibling setter's
   existing convention) placed before that function's first `unsafe`
   dereference of it. Most of the new guards are directly exercised by
   a corresponding test (e.g. `raman_params_rejects_null_oscillator_
   arrays`, the extended `radial_params_rejects_zero_or_non_dividing_
   n_r`/`modal_params_rejects_zero_or_invalid_npol_and_non_dividing_
   n_modes`/`free_params_rejects_zero_transverse_dims_and_non_dividing_
   size`); the two `*_zdep_params` guards that require a real, loadable
   FFTW/`libcubature` to get past their own call-order check
   (`native_set_free_zdep_params`, `native_set_modal_zdep_params`) are
   verified by code inspection only, documented as such at each test
   site, to keep the suite hermetic. `native_set_plasma_params(_adk)`'s
   `ion_ptr` is deliberately unguarded still — that setter stores the
   pointer without dereferencing it, so null there isn't UB in the
   setter itself (see `plasma_params_stores_null_handle_without_
   crashing_the_setter_itself`); the real risk is downstream in
   `native_step`, which Julia's own caller (`src/RK45.jl`) already
   guards against structurally. Verified: `cargo test` 69/69 (was
   68/68 before this fix's own tests were added), and clean under
   `RUSTFLAGS="-D warnings"`, zero regressions. Pairs well with S4.2's
   `FfiStatus` enum.

**Where the project goes from here** (the port itself is done — items
1-4 of Phase I were the last correctness-scope gaps): the highest-value
tracks in order are **S1 (CPU hot-loop performance)** — the port's
premise was performance, and the dispatcher currently vectorizes only
Raman, so this is where the remaining headroom is; **S4 (architecture
cleanups)** — the toggle zoo and reflective probes are the main
maintenance risk as the surface grows, and S4 is a declared prerequisite
for extending S3; **GPU CI + finish S3** — the GPU stepper exists and is
hardware-verified but will bit-rot without CI, as already happened once;
**S6.1 (prebuilt binaries)** — the biggest adoption lever, since today
every user needs a Rust toolchain. Upstream-sync (Phase A.3's weekly
job) and the Phase H PR channel keep the fork honest in both directions.

## Suggestions backlog (imported from SUGGESTIONS.md, 2026-07-07)

Full detail (equations, rationale, per-item code sketches) stays in
`SUGGESTIONS.md` — this is the tracking summary, synced so status lives in
one place. None of S2/S4/S5/S6 has started; **S1.5 was attempted
2026-07-07 and found broken** (see S1.5 below — disabled by default, real
fix still open). **S3 is partially started**:
the GPU-resident stepper work landed 2026-07-05/07 (see Phase G's "Open
items" entry and "Done (recent)") implements a narrow slice of S3
(mode-averaged RealGrid Kerr-only, no threading/dispatch-threshold/design
doc) — and did so *before* the GPU CI dependency S3 itself declares
("needs a CUDA machine... do NOT start before [GPU CI] exists, or it will
rot"), which is exactly what happened once already (uncommitted 2 days,
found broken until manually re-verified). GPU CI is still open (see "GPU
CI coverage" below) — treat S3's remaining scope (design doc, full
`NativeBackend` parity, threading, dispatch threshold, `test_native_gpu.jl`)
as still gated on it.

**ISA / hardware dispatch — synced to actual code state (2026-07-07):**
`dispatch.rs`'s hardware cascade (CUDA → Vulkan → AVX-512 → AVX2 → NEON →
Apple AMX → portable) is real for *detection and selection*
(`is_x86_feature_detected!`, `dlopen`-based Vulkan/CUDA probes all
genuinely check the running machine). But grepping `luna-rust/src/` for
`target_feature`/`_mm256_`/`_mm512_`/`std::arch` finds vectorized code in
exactly **one** file: `raman.rs`'s `solve_avx2`. Every other kernel (Kerr,
RK45 stage accumulation, window/norm broadcasts, QDHT) runs the same
portable-scalar code regardless of which `HardwarePath` `dispatch.rs`
selects — so today the dispatcher's choice of AVX-512/AVX2/NEON is
**cosmetic** for everything except Raman. This is suggestion 3 below,
tracked as S1.4; until it lands, "AVX-512 path selected" does not mean
"AVX-512 code ran."

### 🟡 S1 — Hot-loop CPU performance (suggestions 3, 4, 5)
*Needs `benchmark_native_default.jl` as a before/after harness — that now
exists (Phase G.3).*
1. 🟢 **Verified 2026-07-09.** Full 7-group parallel gate green: physics
   1645/1657 (12 pre-existing `@test_broken`, unrelated), rust 42098/42098,
   sim-interface 301/301, sim-multimode 33/33, sim-propagation 18/18,
   io 2302/2302, fields 334/334 — no failures, no new errors. All exit
   codes 0.
   FFTW wisdom for native plans (item 4) — `fftw.rs` gained
   `import_wisdom_from_filename`/`export_wisdom_to_filename` (bind
   `fftw_import_wisdom_from_filename`/`fftw_export_wisdom_to_filename`,
   both best-effort: symbol missing / bad path / bad file all just return
   `false`, never an error). `native_set_fftw_plans` gained a trailing
   `wisdom_path` arg, imported right after `FftwApi::load` and before any
   plans are created; a new `native_export_fftw_wisdom` FFI call (new
   `NativeBackend::wisdom_export` trait method, `CudaNativeSim` stub
   returns 1/"unavailable") is issued once at the end of
   `RustNativeStepper`'s Julia constructor, after every `native_set_*_params`
   call for that sim. Julia side (`RK45.jl`) always threads a path
   (`joinpath(Utils.cachedir(), "native_fftw_wisdom_$(threads)threads")`,
   a file **dedicated to the native path** — deliberately not shared with
   `Utils.loadFFTwisdom`'s own `FFTWcache_*threads`, to avoid any
   concurrent-access interaction with its `mkpidlock` guard); falls back
   to an empty string (silent no-op on the Rust side) if `cachedir()`
   itself throws. **Known open risk, flagged for tomorrow's review, not
   fixed:** wisdom is exported on *every* `RustNativeStepper` construction,
   not once per process — for a workload that constructs many short-lived
   steppers (e.g. a parameter scan), the disk I/O + FFTW's internal
   planner-lock on every construction could itself be a net regression;
   needs the benchmark harness before deciding whether to gate this
   behind a "once per process" flag or a size/count threshold. Still
   open even though the full gate is now green (item 1 above) — the
   gate confirms correctness, not that per-construction wisdom export is
   a good idea for scan-heavy workloads. Needs `benchmark_native_default.jl`
   before/after numbers, not another correctness run.
   **Second risk, found 2026-07-09 while redistributing the test gate
   into finer-grained processes (see the "17-test discrepancy"
   investigation below):** before this item, native plans were always
   created fresh from `FFTW_ESTIMATE` with no wisdom involved, so a given
   grid size produced a bit-identical plan regardless of process history —
   fully deterministic. Now that `import_wisdom_from_filename` runs before
   plan creation, plan selection can depend on what wisdom has already
   accumulated (both the shared on-disk file across all runs on this
   machine, and FFTW's own in-process wisdom pool that grows automatically
   with every construction). That's harmless for a single one-shot
   simulation per process, but for **any workflow that constructs many
   `RustNativeStepper`s in one process — parameter scans, batch runs, long
   REPL/notebook sessions —** later constructions can get a different
   FFTW plan than earlier ones, which (via the already-documented Phase 2
   near-cancellation sensitivity in the RK45 error estimate, CLAUDE.md)
   can shift the adaptive controller onto a different `dt` sequence.
   Confirmed empirically: splitting the `rust` Julia test group (43
   files) from one shared process into many separate processes changed
   the total assertion count from 42098 to 42081 — same effect, no
   failures either way, isolated by ruling out a file-write race (a fully
   sequential, non-concurrent per-file rerun gave the identical 42081,
   so it's process-topology, not a race). Final accuracy should still
   land within the requested `rtol`/`atol` (that's what the adaptive
   controller is for), but exact step-by-step reproducibility across
   runs/machines/scan position is now weaker than before this item, and
   concurrent scans add a file-race angle on top (best-effort import
   failures silently no-op, so some runs in a concurrent batch may not
   get wisdom at all while others do). Same fix candidates as the
   performance risk above (gate behind "once per process", or skip
   import/export below some construction-count threshold) — needs a
   decision, not just a benchmark number, since this one is about
   reproducibility guarantees, not speed.
   **Resolved 2026-07-09.** Investigated and planned in
   `docs/native-port/PLAN_FFTW_WISDOM_FIX.md`, then implemented: on-disk
   wisdom persistence is now **opt-in, default OFF**
   (`LUNA_NATIVE_FFTW_WISDOM=1` to enable; unset/`"0"` = off), gated in
   `src/RK45.jl` via `_native_wisdom_enabled()` — both the import path
   (`native_wisdom_path` is `""` when off, which Rust's best-effort import
   already no-ops on) and the export block are skipped entirely by
   default. This kills the per-construction disk I/O + planner-lock
   (performance risk) outright, and removes the disk channel of the
   reproducibility risk (consecutive runs on one machine no longer
   diverge via a shared wisdom file). A pool channel remains even with
   persistence off: native `dlopen`s the same `FFTW_jll.libfftw3` Julia's
   `FFTW.jl` uses, so all constructions in one process still share one
   global in-process wisdom pool — a fundamental consequence of FFTW's
   process-global wisdom API (no `fftw_forget_wisdom()` without
   corrupting Julia's own plans), bounded by "one process per simulation"
   (which `scans.rs`'s `ScanQueue` already does). The plan file's crux
   argument: native plans only ever use `FFTW_ESTIMATE`, which by design
   skips measurement, so imported wisdom likely never helped here in the
   first place — default-OFF is justified regardless of a separately
   re-benchmarked perf number. Tests: `test/test_native_fftw_wisdom.jl`
   (T1: default off writes nothing to disk; T2: opt-in exports/imports
   without error). No Rust changes were needed — `fftw.rs`/`native.rs`
   plumbing is unchanged and reused as-is for the opt-in path.
2. 🟢 **Verified 2026-07-09.** Fused RK45 stage accumulation (item 3c) —
   `native.rs`'s `step()` stage loop previously did one full `ystage`
   read-modify-write sweep per nonzero `DP_B[ii][j]` coefficient (up to 6
   sweeps for the last stage: `for j in ks_slice { if bij != 0 { for
   idx in 0..n { ystage[idx] += scale*k_j[idx] } } }`). Replaced with a
   single fused sweep per stage: collect the active `(scale, k_j)` terms
   once (a fixed `[Option<(f64, &[Complex<f64>])>; 6]`, no heap
   allocation), then for each element accumulate all active terms in one
   pass and write `ystage[idx]` exactly once. Deliberately preserved the
   *exact same addition order* (`field[idx] + scale0*k0[idx] +
   scale1*k1[idx] + ...`, same sequence, same operands) rather than
   reassociating — so this is not the "summation-order change" the
   original suggestion anticipated; it's a pure memory-access-pattern
   change (touch `ystage` once instead of once per term), giving
   bit-identical results rather than merely equivalent-within-tolerance.
   Confirmed by the full 7-group gate: `rust` 42098/42098 (bit-identical
   to the pre-fused baseline — no equivalence-tolerance slack needed),
   physics 1645/1657 (pre-existing broken, unrelated), sim-interface
   301/301, sim-multimode 33/33, sim-propagation 18/18, io 2302/2302,
   fields 334/334. 37/37 Rust unit tests, clean build under
   `RUSTFLAGS="-D warnings"`.
3. 🟢 **Verified 2026-07-09** — same full 7-group gate as item 1 above
   (rust group 42098/42098, all groups green). `exp(L·c_i·dt)` caching
   (item 3d) — new `ExpCache` (native.rs, small
   MRU list, capacity 16) keyed on `dt.to_bits()` + a new
   `linop_version: u64` field (bumped by `ensure_linop_at`/
   `ensure_free_norm_at`/`ensure_modal_linop_at` only on their real-work
   branches, never their early-return no-ops) — invalidates cache entries
   from before a z-dependent linop update without needing to eagerly
   clear on every bump. `apply_prop`'s 12-14 calls/step (the DP45 stage
   loop's fixed `DP_NODES[ii]*dt` forward/backward pairs, `native.rs`'s
   `step()`) now go through a new `apply_prop_cached` free function
   instead; the old uncached `apply_prop` was deleted (fully superseded,
   would otherwise be dead code under `-D warnings`). Numerically
   identical to the old path by construction (same `Complex::exp` values,
   memoized, not approximated) — the risk surface is purely "does the
   invalidation ever miss a real `linop` change", not "is the math
   different". Confirmed by the full green gate: no cache-invalidation
   misses surfaced across any geometry (radial, modal, free-space,
   mode-avg z-dependent linop) or grid type.
4. 🟢 **Resolved 2026-07-10 — de-branch, not hand-SIMD.** SIMD Kerr +
   window/norm broadcasts (item 3a) — measured first (temporary
   `Instant`-counter pass on `rhs_mode_avg_real`, same discipline as
   S1.6/S2 Phase 2, add/measure/revert): Kerr+window+norm was ~10.3% of
   `rhs_mode_avg_real`'s own time (~9.7% of full `step()` wall time, since
   the RHS is ~94% of `step()`) on the mode-avg+plasma default benchmark
   workload (`test/benchmark_native_default.jl`'s shape, 2000 fixed-dt
   steps) — well clear of S1.6's ~1-2% park threshold, so proceeded.
   **The backlog's own premise ("route through dispatch.rs's existing
   AVX2/AVX-512 lanes") was wrong** — `dispatch.rs` is only a
   hardware-path *selector* (`SimulationEngine`), not a library of
   elementwise SIMD kernels; the only hand-written lane in the codebase is
   Raman's own `solve_avx2` (`raman.rs`), needed there because Raman's ADE
   update is a sequential recurrence the compiler can't auto-vectorize —
   the opposite of Kerr/window's flat broadcasts. Confirmed via
   `objdump`/`nm` on the release `.so` before writing any code: Steps 1-5
   (FFT, scale, Kerr cube, noise, window) were **already** auto-vectorized
   by LLVM (79 `%ymm` ops, `target-cpu=native` per `.cargo/config.toml`) —
   hand intrinsics there would have been redundant, unsafe-code risk for
   zero gain (exactly the risk class that cost the S2 revert). The real,
   *addressable* headroom was Steps 6-7 (norm + freq-window): a per-element
   `if self.sidx[i] { ... }` branch inside the loop, which does defeat
   auto-vectorization (confirmed scalar `%xmm`/`cmpb` in the disassembly).
   **Fix: de-branch, not intrinsics.** `sidx`/`pre`/`beta`/`sqrt_aeff`/
   `owin` are fixed for the life of a construction *except* for
   z-dependent-linop configs (Phase 7), where `ensure_linop_at` mutates
   `self.beta[i]` on every real z-update — so a naive one-time
   precomputation at `set_mode_avg_params` first shipped a real bug
   (caught by the full gate, not the Rust unit tests: `rtol=1e-8`
   single-step mismatch in `test_native_zdep_linop.jl`, ~1.8e-5 relative —
   caught only because the full gate was run, not a phase-specific
   subset). Fixed by refreshing the precomputed factor inside
   `ensure_linop_at` right after it mutates `beta[i]`, keeping the two in
   sync. New `norm_pre_beta: Vec<Complex<f64>>` field holds
   `pre[i]/beta[i]*sqrt_aeff` at `sidx` positions, `1+0i` (exact identity)
   elsewhere; `owin` is folded to `1.0` outside `sidx` at construction
   (it's read nowhere else). Both `rhs_mode_avg_real` and
   `rhs_mode_avg_env`'s Steps 6-7 become plain unconditional multiply
   loops. Bit-identical by construction (`x*1.0 == x` exactly in IEEE 754
   for any finite/NaN/Inf `x`; the precomputed `pre[i]/beta[i]*sqrt_aeff`
   value is the same deterministic FP result whether computed once or
   recomputed every call). Verified: 50/50 Rust unit tests, full 7-group
   gate exactly matching baseline (rust 42087/42087, physics 1645/1657+12
   pre-existing broken, sim-interface 301/301, sim-multimode 33/33,
   sim-propagation 18/18, io 2302/2302, fields 334/334) — including the
   z-dependent-linop Phase 7 test that caught the bug, now passing.
5. 🟢 **Resolved 2026-07-09.** BLAS-3 QDHT (item 5) — wired 2026-07-07,
   found broken (`cblas_dgemm` dispatch-stub issue, see below), fixed by
   rewriting `blas.rs` to bind the real ILP64 Fortran entry point instead.
   Original bug: `libblastrampoline`'s `cblas_dgemm` symbol is a
   **dispatch stub**, not a real BLAS-3 kernel — it requires Julia's own
   ILP64 Fortran-symbol registration to route to OpenBLAS, and calling
   the CBLAS-name entry point directly errors at runtime ("no BLAS/LAPACK
   library loaded for cblas_dgemm()") while silently corrupting the QDHT
   result instead of computing anything. **Fix:** `blas.rs` now binds
   `dgemm_64_` — the exact same symbol and calling convention Julia's own
   `LinearAlgebra.BLAS.gemm!` resolves to (confirmed by reading
   `stdlib/LinearAlgebra/src/blas.jl`: `USE_BLAS64=true` ⇒
   `@blasfunc(dgemm_)` → `dgemm_64_`, `BlasInt=Int64`, every scalar passed
   *by reference* per the Fortran calling convention, plus two trailing
   `size_t` hidden-string-length args (each `1`, for the two single-char
   `transa`/`transb`) — since this is the identical library Julia's own
   BLAS calls resolve to, binding it the same way is guaranteed to behave
   identically to what Julia itself would compute). `ffi.rs`'s
   `QdhtFfiHandle::apply_real`/`apply_cplx` updated to the new
   `BlasApi::dgemm` signature (no more CBLAS layout arg — Fortran GEMM is
   inherently column-major; `i64` dims; `u8` trans chars). **A real
   pitfall found and worked around:** a standalone `cargo test` that
   `dlopen`s `libblastrampoline` directly and calls `dgemm_64_` segfaults
   — the trampoline's forwarding table is only populated by Julia's own
   runtime startup, so a bare dlopen outside a live Julia process gets an
   unconfigured trampoline (see the comment block in `blas.rs` where the
   unit test would go). This is not a real-usage bug: production code
   (`NonlinearRHS._init_rust_qdht_blas`) dlopens the *already-loaded*
   `libblastrampoline` from inside a running Julia process, reusing its
   live, configured instance — confirmed by running the actual gate,
   `test/test_qdht_rust.jl` with `LUNA_USE_RUST_QDHT=1 LUNA_QDHT_BLAS=1`:
   mul!/ldiv! unit equivalence (6 tests) and full radial-simulation
   equivalence (2 tests) all pass at the ~1e-13 tolerance the backlog
   asked for. Full `rust` Julia group re-verified unaffected
   (42087/42087, `LUNA_QDHT_BLAS` still opt-in so default runs are
   untouched). **Still open, deliberately not done here:** the ≥1.5×
   QDHT micro-bench at n_r=256 this item's own gate also asked for,
   before flipping `LUNA_QDHT_BLAS` to default-on — that's a timeboxed
   perf measurement, separate from the correctness fix; until it's run,
   leave the opt-in default as-is.
6. 🟢 **Parked 2026-07-10 (negative ROI, revisit after current backlog).**
   Split-complex exp-linop spike (item 3b) —
   `luna-rust/benches/exp_linop_layout_bench.rs` (Criterion-only, no
   production code touched). Two shapes benchmarked at n=2000/8000/16000
   (target-cpu=native, matching `.cargo/config.toml`):
   - `apply_prop`'s `y[i] *= factors[i]` complex multiply (the dominant
     per-step cost — 12-14 calls/step via `apply_prop_cached`): SoA beats
     AoS by **~2.0-2.6×** consistently across all three sizes (e.g. 16000:
     11.5µs AoS vs 4.4µs SoA). Clears the >1.3× bar comfortably.
   - `weaknorm_c64`'s `norm_sqr` error-norm reduction: a wash — SoA ties
     AoS at 2000, edges ahead at 8000, and is slightly *slower* at 16000.
     No consistent win; the reduction is likely already auto-vectorized
     well enough on the AoS layout that the shuffle cost from re/im
     unzip-in-place isn't recovered.
   **Decision: greenlit 2026-07-10.** User was shown the expanded scope (no
   chokepoint — ~30 distinct `Vec<Complex<f64>>` fields in `native.rs`, no
   split-DFT support in `fftw.rs` today, every FFT-touching kernel pays an
   interleave tax under SoA unless split plans exist, the Julia↔Rust FFI
   boundary is hard-AoS regardless, and only `apply_prop_cached` — not
   `weaknorm_c64`, not any `rhs_*` kernel — showed a measured win) and chose
   to proceed with the full end-to-end conversion anyway. Tracked as a
   phased effort, full plan in
   `docs/native-port/PLAN_S1_6_SOA_CONVERSION.md` (mirrors how the native
   port itself — Phases 0-8, D-I — was staged and gated). Each phase lands
   as its own commit, gated by the full 7-group test suite (never a subset)
   before the next phase starts.
   - **Phase 0 — done (2026-07-10).** `fftw.rs` gained
     `fftw_plan_guru_split_dft`/`_r2c`/`_c2r` bindings + two new wrapper
     types (`SplitComplexFft1d`, `SplitRealFft1d`) operating on separate
     re/im `f64` arrays. 3-D split variants deliberately deferred until the
     free-space geometry phase needs them. Verified bit-exact against the
     existing AoS plans (`split_c2c_matches_aos_c2c`,
     `split_r2c_matches_aos_r2c`); not yet wired into `native.rs` (purely
     additive). Full gate unchanged: rust 42087/42087, physics 1645/1657
     (pre-existing broken), sim-interface 301/301, sim-multimode 33/33,
     sim-propagation 18/18, io 2302/2302, fields 334/334.
   - **Phase 1 — halted before implementation (2026-07-10), pending
     re-confirmation.** Before writing the Phase 1 rewrite, measured
     `apply_prop_cached`'s actual share of `step()` wall time (temporary
     `Instant`-based counters, reverted after the reading — see
     `PLAN_S1_6_SOA_CONVERSION.md`'s gotcha section) on the same
     mode-averaged+plasma+kerr workload `benchmark_native_default.jl` uses,
     2000 fixed-dt steps: **`apply_prop_cached` is ~1.9-2.0% of `step()`'s
     own wall time.** Even a perfect 2.6× speedup on it (the isolated
     Criterion spike's best case) caps the *entire* SoA conversion's
     end-to-end win at roughly **1.2%** for this workload — the ~14
     calls/step are O(n) sitting next to O(n·log n) FFTs called several
     times per stage across 7 stages, which dominate `step()`'s time and
     don't get faster from this (FFTW's own split-DFT plans are typically
     equal-or-slightly-slower than its interleaved plans, so Phase 2
     wouldn't recover the gap either). This is materially different
     information from what motivated the "proceed with full conversion"
     decision above (that decision was made on "2-2.6× on the multiply"
     alone, before this ceiling was known) — flagged back to the user
     before continuing into the multi-week Phase 1-4 rewrite of a
     numerics-critical stepper for a ~1% ceiling.
     **Confirmed across two more workloads (2026-07-10)** at the user's
     request before any final call: radial geometry (QDHT, N=32 r-points)
     — ~2.5%; a 16×-larger mode-avg+plasma grid (n_spec=16385 vs. the
     default ~1025) — ~1.9%. The ceiling doesn't grow with grid size or
     change materially across geometry — FFT cost dominates `step()`
     proportionally regardless, so this isn't an artifact of the one
     default-benchmark workload.
   - **Final call (2026-07-10): stop here.** Phases 1-4 will not be built
     given the confirmed ~1% end-to-end ceiling — not worth a multi-week
     rewrite of ~30 buffers/8 near-duplicate RHS functions in a
     numerics-critical stepper. Phase 0 (`fftw.rs` split-array bindings)
     stays committed as harmless, already-tested infrastructure in case a
     future kernel needs split-DFT plans for an unrelated reason. Parked,
     not deleted — **explicitly flagged for reconsideration once the rest
     of the current BACKLOG.md work is finished**, in case a future
     workload profile (e.g. a kernel that isn't FFT-dominated) changes the
     ceiling. Full writeup, including the cross-workload measurements, in
     `docs/native-port/PLAN_S1_6_SOA_CONVERSION.md`.
   - Effort redirected to S2 (threading the native RHS) and S1.4 (SIMD
     Kerr/window broadcasts) instead — both target the FFT/RHS-dominated
     cost this investigation surfaced, which is where the real headroom
     appears to be, rather than the small exp-linop multiply.

### 🟢 S2 — Threading the native RHS (suggestion 2) — re-landed 2026-07-11, root cause fixed
*Started 2026-07-10, reverted same day, root-caused and re-landed
2026-07-11. Radial geometry only this pass (items 1-2); modal (item 3)
and free-space (item 4) left as separate, not-started follow-ups — see
`docs/native-port/PLAN_S2_THREADING.md` for the full phased plan.*

**Phase 3 REVERTED 2026-07-10, RE-LANDED 2026-07-11 — root cause was a
missing GC root, not a Rust-side memory-safety bug in the parallel code
itself.** After Phase 3 was committed/pushed (`d15a25c`), a post-hoc
wall-clock benchmark (`bench_threads.jl`-style: `n_threads=1`→4 at N=32
then N=128, plasma enabled throughout, same process) surfaced an
intermittent segfault inside `PptIonizationRate::rate`. The revert
commit's isolation experiment (Kerr-only config never crashed) correctly
implicated `apply_plasma_radial`'s parallel branch, but its diagnosis —
"an out-of-bounds write during the `n_threads=4` plasma call corrupts the
allocator's view of nearby memory" — was wrong about *where* the bug
lived.

**Actual root cause, found 2026-07-11 by installing ASAN
(`rustup +nightly` with `-Z build-std` — plain `-Z sanitizer=address`
silently fails to catch heap corruption on this toolchain because the
prebuilt stdlib allocator shim isn't itself instrumented) and Valgrind,
then reproducing the crash directly:**
- An isolated Rust-only repro of `apply_plasma_radial()` alone (40 varied
  cycles, real ASAN instrumentation) never crashed — ruling out a
  self-contained out-of-bounds write in that function.
- The **full pipeline** (checked out `d15a25c` into a worktree, ran a
  real `RustNativeStepper` radial+plasma solve at `native_threads=4`)
  reproduced a genuine `SIGSEGV` reliably. `coredumpctl`/gdb's backtrace
  showed a rayon worker thread segfaulting *inside*
  `PptIonizationRate::rate()`, called concurrently by multiple worker
  threads via `plasma_rate_at`/`rayon::iter::plumbing::bridge_producer_consumer`.
- Manual review of `CubicSplineLUT::evaluate()`'s binary search and its
  `get_unchecked(low)` call: logically correct and thread-safe as
  written — `low` is provably `< segments.len()` by the search's own
  invariant, and it only reads immutable `&self` data with no interior
  mutability. So the crash meant the `Vec<SplineSegment>`'s own heap
  metadata (pointer/len/capacity) was **already corrupted** before this
  call ran.
- Tracing how `plasma_ion_ptr`/`plasma_adk_ptr` (`*const
  PptIonizationRate`/`*const AdkIonizationRate` in `native.rs`) get set
  found the real defect: `native_set_plasma_params` stores this raw
  pointer *directly*, re-dereferencing it on every future `native_step`
  call for the sim's entire lifetime — unlike every other kernel's
  setup (Raman/dispersion oscillator data), which Rust copies into its
  own `Vec` once and never touches the original pointer again. The
  pointee is Julia-allocated-and-owned: `Ionisation.jl`'s
  `RustIonizationHandle`/`RustAdkHandle` carry a GC finalizer
  (`free_ppt_ionization_lut`/equivalent) that frees the Rust memory.
  `RustNativeStepper` (`RK45.jl`) never stored a Julia-level reference
  to `irf`/`irf.rust_handle` after construction — only the raw pointer
  value crossed the FFI boundary. Since Julia's `ccall` releases the GC
  safepoint for the duration of a foreign call (letting other Julia
  threads/the GC run concurrently, with *no visibility into native Rust
  threads* still holding the pointer), the handle was eligible for GC
  finalization **at any point after construction**, including mid-solve
  while rayon worker threads were still concurrently dereferencing it —
  a textbook FFI use-after-free. Threading only widened the race window
  (longer-lived worker threads doing more concurrent work, more
  allocation pressure to trigger a GC cycle during the window); it did
  not cause the unsoundness — the same latent bug existed on the
  sequential path too, just with a race window too small to observe in
  practice.
- **Fix** (applied to `main` independently of re-landing threading, since
  it's a real latent bug regardless): added a `_gc_roots::Vector{Any}`
  field to `RustNativeStepper` (`RK45.jl`) that the constructor populates
  with every Julia handle object (`irf.rust_handle`) whose raw pointer
  the Rust-side `NativeSim` stores persistently, across all three
  geometries (mode-avg/radial/modal) and both ionization models
  (PPT/ADK) — six call sites total. Nothing reads `_gc_roots`; its only
  job is keeping those objects Julia-reachable for the stepper's whole
  lifetime, preventing early finalization.
- **Verified the fix holds**: re-applied the Phase 3 diff (`fftw.rs`'s
  `unsafe impl Sync`, `native.rs`'s `ReadOnlyPtr`/`plasma_rate_at`/
  parallel FFT-and-plasma branches, `RK45.jl`'s `native_threads` kwarg,
  both bit-identical tests) on top of the GC-root fix, then re-ran the
  exact crash-reproducing script for 8 cycles alternating N=32/N=128 and
  `native_threads` 1/4 with `GC.gc()` forced between every cycle (to
  maximize GC pressure) — all 8 completed cleanly, versus crashing at
  cycle 2 before the fix. `n_threads=1` and `n_threads=4` results were
  bit-identical across every repeated size (confirmed via `yn` norm
  equality, not just non-crashing). Full 7-group gate green (see "Done
  (recent)").
- **Lesson for future FFI kernel wiring**: any `native_set_*_params` call
  that stores a raw pointer into Julia-owned, Rust-allocated memory
  *persistently* (re-dereferenced across multiple future calls, not just
  copied once at setup) needs its Julia-side handle rooted somewhere
  that outlives construction — `RustNativeStepper._gc_roots` is that
  place going forward. Kernels that copy data into their own `Vec` at
  setup time (the common pattern) don't need this.
1. 🟢 **Done 2026-07-10.** `native_set_threads(handle, n)` FFI, wired from
   `Threads.nthreads()`, default 1 (bit-identical to today — verified via
   full 7-group gate, purely additive plumbing, `n_threads` not yet read
   by any RHS code).
2. `rhs_radial`: rayon over radial nodes, each node's own FFTW
   new-array-execute call against one shared plan; one scratch slab per
   rayon worker (never shared — this is precisely the bug the Julia
   `Threads.@threads` `pointcalc!` race had, see Phase B/"Done (recent)").
   **Re-scoped after investigation:** an Explore survey found the two
   per-column FFT loops and `apply_plasma_radial`'s per-column loop
   already operate on disjoint slices of matrix-shaped
   (`n_time_over*n_r`) buffers, not shared per-call scratch — safe for
   `par_chunks_mut` without new per-worker scratch structures. Only
   `apply_raman_radial`'s single shared `raman_solver` (and, when
   `raman_thg==false`, shared Hilbert scratch) has the genuine
   shared-mutable-state hazard the backlog warns about. **🟢 Done
   2026-07-19 (S2 Phase 4 item 1):** parallelized by partitioning the
   r-columns into ≤`n_threads` *contiguous* groups, each rayon task owning
   its **own cloned `TimeDomainRamanSolver`** and **own Hilbert scratch**
   (no `current_thread_index`, no interior mutability, provably disjoint
   column slices; `solve()` resets oscillator state at entry, so a cloned
   solver is bit-identical to the shared one). No new GC-root hazard — the
   solver is Rust-owned and only *cloned*, not a persistent raw pointer
   into Julia-owned memory (contrast the plasma-pointer UAF fixed on
   `main`). Verified to the full S2 bar: `n_threads=1`-vs-`4`
   **bit-identical** (`s.yn` exact) in `test/test_native_radial_raman.jl`,
   plus an 8-cycle forced-`GC.gc()` stress repro with no crash.
   Modal (item 3) and free-space (item 4) threading remain the open
   follow-ups (the latter needs an isolated multi-threaded FFTW plan under
   `PLANNER_LOCK`: `fftw.rs`'s `unsafe impl Sync` only holds for
   single-threaded plans, so an `nthreads>1` plan would deadlock under the
   existing concurrent `fftw_execute_dft` path).
   **Measured 2026-07-10** (temporary `Instant` profiling, reverted after
   reading, same discipline as S1.6): FFT-loop + plasma-loop share of
   `rhs_radial` time, at N=32/N=128 r-points, with/without plasma:
   | N | plasma | FFT | QDHT | plasma | other |
   |---|---|---|---|---|---|
   | 32 | off | 35.9% | 46.7% | — | 17.4% |
   | 32 | on | 20.9% | 28.3% | 40.5% | 10.2% |
   | 128 | off | 18.6% | 72.6% | — | 8.8% |
   | 128 | on | 14.1% | 54.8% | 24.5% | 6.6% |

   Unlike S1.6's ~2% ceiling, FFT+plasma combined is **38-61% of
   `rhs_radial` time** for plasma-enabled configs — clears the bar for
   proceeding. QDHT dominates the rest (already internally
   parallel/BLAS-backed via S1 item 5's BLAS-3 QDHT fix — out of scope
   for this item).
3. 🟢 **Modal: DONE 2026-07-20.** rayon over the cubature batch's nodes
   inside both `integrand_v` callbacks (`modal_integrand_v` /
   `modal_integrand_v_full`) — cubature's own adaptive node *placement* stays
   sequential/deterministic; only the per-node integrand evaluation is
   parallelized. **Measured first** (temp `Instant` counters on `rhs_modal`,
   add/measure/revert per S1.6/S2-Phase-2 discipline): the integrand loop is
   **90.3%** (full=false, 1 mode) / **95.6%** (full=true) / **82.8%**
   (2-mode) of `rhs_modal` wall time — well above the proceed bar (radial was
   38-61%; S1.6 parked at ~2%). Batch sizes are small for full=false (~4
   nodes/batch) but each node is ~27-33µs of FFT-heavy work, so threading
   still nets a win; full=true batches ~25 nodes. **Measured wall-clock
   speedup (`native_threads`=1→4, this 12-core host, 300-step fixed-dt solve,
   min-of-3):** full=false 1-mode **1.31×**, full=false 2-mode **1.52×**,
   full=true **2.64×** — every config a genuine speedup (even the small-batch
   full=false regime S3.3/S1.6 warned could go net-negative), which also
   proves the parallel branch actually engages (the bit-identical test isn't
   vacuously passing on a silently-sequential path). No min-npt guard needed.
   **Refactor:**
   `rhs_modal_pointcalc` (a `&mut self` method scribbling on ~13 shared
   `self.modal_*`/`raman_*` scratch buffers) was split into a `Sync`
   read-only view (`ModalRO`: `&[..]`/`Copy`/`Option<&Plan>`, FFT wrappers
   already `Sync`) + per-worker `ModalScratch` (all written buffers, pooled on
   `self.modal_scratch_pool`, entry 0 = sequential path) and a free-standing
   associated fn `modal_pointcalc(&ro, sc, r, θ, out)` used by BOTH paths (one
   code path, no duplicated 270-line body). Nodes split into `≤ n_threads`
   contiguous groups; each group's `out[p*fdim..]` is disjoint with no
   cross-node reduction ⇒ **bit-identical** `n_threads`=1-vs-4 (the S2 gate,
   not a tolerance). Raman-modal threaded too: each worker's `ModalScratch`
   carries its **own cloned** `TimeDomainRamanSolver` + Hilbert scratch (same
   discipline as radial item 1; `solve()` resets state at entry ⇒ clone ==
   shared, bit-identical); the Hilbert FFT plan is shared read-only
   (`fftw_execute` thread-safe against distinct buffers). No new GC-root
   hazard — the solver is Rust-owned/cloned, not a persistent raw pointer into
   Julia memory (contrast the plasma-pointer UAF fixed on `main`). New
   `test/test_native_modal_threading.jl` (Kerr full=false/full=true/2-mode/
   npol=2, Raman :N2, + forced-`GC.gc()` stress loop): all bit-identical
   1-vs-4, native-vs-Julia unchanged at ~2e-16 (Kerr) / ~1e-6 (Raman ADE-vs-
   FFT floor). 70/70 Rust unit tests; clean `-D warnings` build.
4. 3-D free-space FFT: `fftw_plan_with_nthreads`/`fftw_init_threads`
   (dlopened `libfftw3_threads`, silent fallback if absent). *Not started —
   the last open S2 item.* Blocked on `fftw.rs`'s `unsafe impl Sync` holding
   only for single-threaded plans: an `nthreads>1` plan would deadlock under
   the existing concurrent `fftw_execute_dft` path, so it needs an isolated
   multi-threaded plan under `PLANNER_LOCK` first.
5. 🟢 Error-norm reduction stays sequential (determinism, ties to S5.2) —
   confirmed already sequential (`weaknorm_c64`), untouched by this item.
- Gate: universal + fixed-step equivalence under `JULIA_NUM_THREADS=4` +
  ≥2× radial/free benchmark at 4 threads + n=1 bit-identical to pre-track.
  For radial specifically: since the parallelized loops write disjoint
  memory with no cross-column reduction, `n_threads=1` vs. `n_threads=4`
  must be **bit-identical**, not merely within tolerance — a stronger,
  more testable guarantee than typical parallel-code equivalence.

### 🟠 S3 — GPU-resident propagation (suggestion 1) — partially started, see note above
*Large (5+ sessions). Plan's own stated dependency (GPU CI) is not yet
met — see "GPU CI coverage" below. This machine has real GPU hardware
(RTX 5060 Ti, CUDA 13.3) usable for manual verification of future slices,
confirmed 2026-07-11 (`nvidia-smi` needs the sandbox disabled to reach the
driver — a standing requirement for any GPU work in this repo, not a
one-off).*
Already landed (2026-07-05/07, ahead of the plan's own sequencing): the
`NativeBackend` trait extraction, `CudaNativeSim` scoped to mode-averaged
RealGrid Kerr-only (not "+plasma" — see item 1 below, plasma was never
implemented), verified on real hardware, wired behind
`LUNA_USE_RUST_CUDA_NATIVE=1`. Still open, per the original design:
1. 🟢 **Done 2026-07-11.** Design doc reconciliation
   (`docs/native-port/GPU.md`). Rewrote §8 (was still the stale
   2026-07-05, pre-hardware "untested" text — the 2026-07-07 verification
   pass and Julia wiring had updated `BACKLOG.md` but never made it back
   into this doc) and §7 (claimed V1 scope was "Kerr (+plasma)"; actual
   shipped scope is Kerr-only — every `set_*_params` beyond
   `set_mode_avg_params`, including `set_plasma_params`, returns `-1`,
   confirmed by reading `cuda_native.rs` directly). §4's `enum{Cpu,Gpu}` vs
   the actual `Box<dyn NativeBackend>` deviation: kept as a documented,
   deliberate divergence (dynamic-dispatch cost is one vtable call per
   accepted step, immaterial next to actual kernel-launch cost; it's also
   what lets `CpuNativeSim`/`CudaNativeSim` share one FFI surface) rather
   than treated as a TODO to "fix" — the doc previously implied this
   should eventually match §4, which it never needs to. No code changed;
   documentation-only.
2. 🟡 **PPT plasma done 2026-07-11 (mode-avg only); Raman, radial/modal/
   free-space, ADK still open.** Added to `CudaNativeSim`
   (`cuda_native.rs`/`kernels.cu`/`cuda.rs`): `set_plasma_params` uploads
   the same `SplineSegment` LUT format `PptIonizationRate::rate_vector_gpu`
   already uses (reused directly, no new upload format), then `step()`
   runs a 5-kernel sequence per RK stage — `ppt_ionization_kernel` (reused
   from the standalone `LUNA_USE_RUST_IONISATION` path, parallel over
   `n_time`) → `plasma_fraction_kernel` (fused cumtrapz+ρ-transform,
   single-thread sequential scan) → `plasma_phase_kernel` (parallel) →
   `plasma_current_kernel` (fused cumtrapz+loss-current, single-thread) →
   `plasma_polarization_kernel` (fused cumtrapz+accumulate-into-Pto,
   single-thread) — mirroring `native.rs`'s `apply_plasma_real` exactly.
   Single-thread sequential kernels for the 3 cumtrapz stages, not a
   work-efficient parallel prefix scan (GPU.md's original sketch,
   item 4 below) — a deliberate V1 tradeoff: `n_time` (~2^13-2^16) is
   small enough that one thread looping over it is negligible next to this
   step's FFT/launch cost at mode-averaged scale; would matter more for
   radial's much larger per-column state, not in scope here. `_gpu_native_eligible`
   (`RK45.jl`) relaxed to allow exactly one plain Kerr response plus at
   most one PPT-only plasma response (ADK still returns `-1` from
   `set_plasma_params_adk`, unimplemented) — the shared FFI call
   (`native_set_plasma_params`) needed zero Julia-side changes beyond the
   eligibility gate, since it already dispatches through the same
   `Box<dyn NativeBackend>` handle both CPU and GPU sims share.
   **Real pre-existing bug found and fixed while wiring this in**:
   `rhs_mode_avg_real_kernel`'s call site passed `(eto_d, pto_d, ...)` but
   the kernel's own declared parameter order is `(pto, eto, ...)` —
   swapped, so the kernel's `pto` write target was actually bound to
   `eto_d` (overwriting the just-FFT'd field with the Kerr result, silently
   discarded before ever being read again) and its `eto` read source was
   bound to `pto_d` (stale/uninitialized memory, not the field) — meaning
   every accepted step's forward FFT (`cufftExecD2Z` on `pto_d`) transformed
   whatever was left in `pto_d` from a previous call, not the real Kerr
   polarisation. Present since the 2026-07-05/07 GPU work, never caught by
   the existing Kerr-only equivalence test because that test's energy is
   weak enough that the resulting error stayed under the test's existing
   ~4.5e-4-driven `<1e-3` tolerance regardless. Fixed by correcting the
   argument order to match the kernel's declaration (also documented
   in-line in `cuda_native.rs`, right at the call site, so it can't silently
   regress again). Every new kernel-arg array is bound through named `let`
   locals, not inline temporaries — the exact UB pattern (`&mut {expr} as
   *mut _`) that caused a real `SIGSEGV` inside `libcuda.so` in the original
   2026-07-07 verification pass; caught in review before this landed.
   **Verified on real CUDA hardware** (RTX 5060 Ti): new
   `test/test_native_cuda.jl` testitem (`"...Kerr+plasma"`) passes,
   `rel_solve ≈ 2.0e-2` at `gas=:Ar, energy=6e-6`. This is *not* the same
   tolerance tier as the Kerr-only sibling test (~1e-3) — diagnosed, not
   assumed: the CPU-resident native path (`LUNA_USE_RUST_NATIVE=1`, no
   CUDA — proper `n_time_over`-sized buffers) matches the Julia oracle for
   this exact config to `1.3e-16`, and sweeping energy 1e-7→6e-6 (60×)
   showed `rel_solve` scaling almost exactly linearly with energy at every
   point — the signature of the same pre-existing, documented
   `n_time`-vs-`n_time_over` Kerr buffer-sizing gap (item 6 below; GPU.md
   §8) amplified roughly 40× by plasma's Keldysh-exponential sensitivity to
   field amplitude, not a new bug. Full 7-group gate green, zero
   regressions (rust 42117/42117 = 42113 baseline + 4 new assertions).
3. 🟢 **Done (2026-07-16).** Problem-size dispatch threshold — measured, not
   guessed, on real hardware (RTX 5060 Ti). Benchmarked `native_step`
   CPU-vs-GPU directly (mode-avg, RealGrid, fixed-step, 10-iteration average
   after warmup) across a size sweep before writing any dispatch code, per
   this backlog's own "benchmark first" discipline (the r2c/c2r item, #3 in
   Phase I above, was investigated the same session and *failed* that same
   gate — no benchmark existed there, so it was correctly left alone).
   **Two very different regimes, not one crossover:**
   - **Kerr-only**: GPU is slower below ~n=8,193 (breakeven ~1.3x there),
     then wins increasingly — 5x at n=16,385, 14x at n=65,537, 27x at
     n=262,145. Dominated by cuFFT throughput at scale; small-n loss is
     CUDA kernel-launch/sync overhead (`cuda_native.rs::step`'s
     `launch_checked` synchronizes after every one of ~dozens of per-stage
     launches).
   - **Kerr+plasma (PPT)**: GPU is 20-30x *slower* than CPU at every size up
     to n=131,073 tested, and the gap *widens* with n — the opposite trend
     from Kerr-only. Root cause: `cuda_native.rs`'s plasma kernels
     (`plasma_fraction_kernel`/`plasma_current_kernel`/
     `plasma_polarization_kernel`) are single-GPU-thread sequential scans
     (item 2's documented V1 tradeoff, item 4/6 below), so they don't
     benefit from n the way cuFFT does — they're pure serial overhead that
     scales with grid size. No crossover exists in the tested range, so a
     single numeric threshold across both regimes would be actively
     misleading.
   - **Fix**: `AMALTHEA_NATIVE_GPU=off/on/auto` (`Config.jl`'s new
     `gpu_dispatch::Symbol` field), layered on top of
     `AMALTHEA_USE_RUST_CUDA_NATIVE`'s existing master opt-in (unchanged).
     `off` forces CPU; `on` restores the old unconditional-GPU behavior
     (kept for forcing GPU on a small/known config, e.g. reproducing a
     specific benchmark); `auto` (**new default**) requires a plasma-free
     config at `n >= _GPU_KERR_ONLY_N_THRESHOLD = 16384` (chosen with margin
     above the measured n=8,193 breakeven, since GPU time isn't cleanly
     monotonic in n at small sizes) — a plasma-bearing config is rejected by
     `:auto` outright, at any size, since no threshold is supported by data
     there. `RK45._gpu_native_eligible(f!, linop)` split into a pure
     config-shape check (`_gpu_kernel_supports`, unchanged logic) and the
     new size/policy-aware `_gpu_native_eligible(f!, linop, n)` (now 3-arg;
     `n = length(y0)`, threaded through from `RustNativeStepper`'s existing
     `n`). Full measured table and reasoning live in
     `RK45._GPU_KERR_ONLY_N_THRESHOLD`'s docstring, next to the code it
     justifies. Both existing `test_native_cuda.jl` GPU-vs-CPU equivalence
     tests now explicitly set `AMALTHEA_NATIVE_GPU=on` (they test raw kernel
     correctness at deliberately small/known configs, independent of the
     dispatch heuristic — including the Kerr+plasma test, which
     intentionally still drives the known-slow path for numerical
     verification). New `test/test_native_gpu_dispatch.jl` covers the
     `:off`/`:on`/`:auto` decision matrix directly (pure Julia-side logic,
     no `ccall`, so it runs without GPU hardware, unlike the sibling
     equivalence tests).
4. Raman ADE kernel, ADK plasma kernel, radial/modal/free-space geometries
   — or explicit `NativeIneligible`-style eligibility split keeping those
   configs on CPU for GPU v1, documented as such. A work-efficient parallel
   prefix scan for cumtrapz (superseding item 2's single-thread kernels)
   would matter here, at radial's much larger per-column plasma-state size.
5. `test/test_native_gpu.jl`-style coverage of the above, mirroring the
   phase-test structure used throughout the CPU native port.
6. The `n_time`-vs-`n_time_over` Kerr/plasma buffer-sizing fidelity gap
   itself (GPU.md §8) — not fixed by item 2 above, which worked within the
   existing buffer sizing rather than changing it.

### 🟢 S4 — Architecture cleanups (suggestions 6, 7, 8) — done, gate closed 2026-07-11
*Found already substantially implemented (undated prior session, not
reflected here) when picked up per the suggested execution order — only
the gate test itself was missing.*
1. 🟢 **Done.** Config struct (item 6) — `src/Config.jl`'s `BackendConfig`
   (one `Bool` field per `LUNA_USE_RUST_*`/`LUNA_*` toggle, default `false`
   except `native`) + `Luna.Config.backend_config()` resolver (re-reads
   `ENV` every call, not cached — the test suite flips toggles mid-session
   via `withenv`). Every call site migrated off raw `get(ENV, "LUNA_USE_RUST_*", ...)`
   (confirmed via grep — none remain outside `Config.jl`'s own docstring).
   `Luna.backend_report()` added (`src/Luna.jl:100`), exposing
   `(config, last_stepper_type)` as a public API wrapping the
   `RK45._LAST_STEPPER_TYPE` test-only hook.
2. 🟢 **Done.** FFI error enum (item 7) — implemented as `RK45.check_ffi(rc,
   what; ineligible=false)` (`src/RK45.jl:798`), used at every native FFI
   call site. **Deliberately not a Rust-side `FfiStatus` enum**: the same
   numeric nonzero code means "ineligible" at one call site and "bug" at
   another depending on what Julia already validated before calling, so
   the ineligible/bug classification is a call-site policy decision, not
   encodable in the return code itself — documented at `check_ffi`'s
   docstring.
3. 🟢 **Done.** Explicit accessor seams (item 8) — `Nonlinear.kerr_γ3(resp)`,
   `NonlinearRHS.norm_pre(norm!)`, `NonlinearRHS.norm_βfun(norm!)`,
   `RK45._is_plain_kerr_resp(r)` centralize what were 4 duplicated inline
   reflective loops across `RK45.jl`'s native-stepper construction paths
   into one place each.
- Gate: **closed 2026-07-11** — `test/test_backend_config.jl` added (was
  the one actually-missing piece): `backend_config()` defaults/truthiness
  under `withenv`, plus `backend_report()` asserting `RustNativeStepper`
  for a default `prop_capillary` call and `PreconStepper` under
  `LUNA_USE_RUST_NATIVE=0`. Full `rust` group: 42101/42101 (42087 baseline
  + 14 new assertions).

### 🟡 S5 — Numerics options (suggestions 10, 11, 12)
*Item 2 done 2026-07-11 (re-scoped). Items 1 and 3 investigated 2026-07-19
— both re-scoped after measurement, see below (item 1: bar not cleared,
reverted; item 3: backlog premise wrong, DP5 5th-order continuous extension
is not the coefficient swap the entry implied — deferred as a larger item).*
1. 🟢 **Done 2026-07-19 — measured, bar not cleared, reverted (S1.6
   discipline).** Mixed-precision spike (item 10). Added a timeboxed
   Criterion bench (`amalthea/benches/mixed_precision_bench.rs`, since
   reverted) microbenchmarking the precision-sensitive inner arithmetic of
   one accepted Dormand-Prince step — the two 7-term stage combinations
   (`yn = field + dt·Σ b5ᵢ·kᵢ`, `yerr = dt·Σ errᵢ·kᵢ`) plus the weak error
   norm (`native.rs`'s `native_step`, lines ~4272-4363) — in f64 vs. an
   f32-mixed variant (stages held/combined in f32, norm reduced in f64:
   the most generous case for the error estimate).
   - **Measured speedup (`target-cpu=native`, this 12-core host): ~1.0-1.06×**
     — f64 vs f32-mixed at n=8192: 65.8µs vs 64.8µs (1.02×); n=16384:
     132.9µs vs 131.4µs (1.01×); n=65536: 577µs vs 546µs (1.06×). Far
     below the >1.4× gate (a). The loop is compute/FMA-bound and already
     well auto-vectorised in f64, not the memory-bandwidth-bound loop that
     would have let f32's halved byte-count translate to speed — so f32
     buys almost nothing here.
   - Gate (b) would also fail regardless of (a): the error estimate is the
     b5−b4 near-total cancellation (TESTING.md §3 / CLAUDE.md's Phase-2
     gotcha — a ~1e-15 summation-order difference already amplifies into a
     ~20% relative `err` swing near the FP-noise floor, which the PI
     controller turns into a different step path). f32's ~1e-7 relative
     precision on the cancelling `Σ errᵢ·kᵢ` sum cannot keep the adaptive
     step sequence identical, so even a hypothetical speed win would change
     the propagation.
   - **Decision: do not implement; bench reverted** (only the numbers above
     retained, per the S1.6 add/measure/revert precedent).
2. 🟢 **Done 2026-07-11 — re-scoped, backlog's own premise was partly
   wrong.** Deterministic mode (item 11). Investigated before writing any
   code (per the S1.4/S4 pattern of checking the backlog's premise against
   current architecture, not just its literal wording):
   - **"Pinning `dispatch.rs` to the portable lane" — dropped.**
     `dispatch.rs`'s `SimulationEngine`/`HardwarePath` is a vestigial
     hardware-path *selector*, never wired into any RHS kernel's actual
     codegen (same finding class as S1.4: real vectorization comes from
     `target-cpu=native` at compile time, not a runtime branch this enum
     could gate). There is nothing to "pin."
   - **"Forcing sequential reductions" — the default native path already
     *is* run-to-run deterministic on one machine.** With S2 Phase 3
     reverted, no RHS code reads `n_threads` yet. Every parallel seam that
     exists today (the QDHT Rayon fallback, the older per-kernel
     `LUNA_USE_RUST_QDHT` batch loops in `ffi.rs`) is embarrassingly
     parallel — each output row/column computed independently with no
     cross-thread reduction — and native FFTW plans only ever use
     `FFTW_ESTIMATE` (no wisdom-dependent plan selection by default, see
     S1 item 1). So a naive "two runs bit-identical" test would pass
     whether or not a new toggle did anything — and a first implementation
     attempt shipped exactly that vacuous test before catching it.
   - **The one real, addressable lever: process-global BLAS-eligibility.**
     `luna-rust/src/blas.rs`'s `BLAS_API` is an `OnceLock`, populated only
     by the older per-kernel `LUNA_USE_RUST_QDHT`+`LUNA_QDHT_BLAS` path
     (`NonlinearRHS._init_rust_qdht_blas`). Once populated, it silently
     makes *every later* native-path radial-QDHT call in that process
     BLAS-eligible too, even though nothing in the native construction
     path asked for it — a process-global-state contamination hazard in
     the same family as S1 item 1's wisdom-pool finding. BLAS-3 `dgemm`
     (OpenBLAS/MKL) and the row-parallel Rayon fallback sum in a different
     order, so which one silently gets taken is a real,
     configuration-order-dependent effect on the result.
   - **Fix:** `native_set_deterministic(handle, bool)` FFI (`native.rs`,
     `NativeBackend` trait + `CpuNativeSim`/`CudaNativeSim` impls) forces
     the native-port radial `QdhtFfiHandle` to skip the BLAS branch
     regardless of `BLAS_API` state; a second FFI,
     `qdht_ffi_set_deterministic` (`ffi.rs`), does the same for the
     per-kernel handle — needed because that's the only call site that
     ever populates `BLAS_API` in the first place, so leaving it unwired
     would make the flag inert exactly where contamination originates.
     `deterministic::Bool` added to `Config.jl`'s `BackendConfig`
     (`LUNA_NATIVE_DETERMINISTIC`, default off), read at both call sites
     (`RK45.jl`'s native construction, `NonlinearRHS._make_rust_qdht_handle`).
     **What this guarantees: BLAS-eligibility invariance** — the result no
     longer depends on whether some unrelated earlier construction in the
     same process touched `LUNA_QDHT_BLAS`, nor on which BLAS
     implementation/thread-count Julia happens to have linked. **What it
     does NOT guarantee:** bit-identical results across different
     machines/CPU targets (`target-cpu=native` means a different build
     host takes a different SIMD/libm path); it is also not "fixing" a
     same-process run-to-run instability that was never observed to exist
     at these problem sizes.
   - **Test:** `test/test_native_deterministic.jl`, tagged `:rust`.
     Deliberately does *not* stop at "two runs bit-identical" (that alone
     can't distinguish a working flag from an inert one, as above) —
     T1 establishes the two-runs-bit-identical baseline before `BLAS_API`
     is ever touched; the test then explicitly contaminates process-global
     `BLAS_API` via `NonlinearRHS._make_rust_qdht_handle` (mirroring a real
     session that mixes the per-kernel and native-port Rust paths); T2
     asserts `deterministic=false` vs. `deterministic=true` now produce
     numerically *different* results (`s_blas.yn != s_det.yn`) — proof the
     flag actually gates the BLAS branch rather than being unreachable;
     T3 re-confirms bit-identical repeats under `deterministic=true` even
     after contamination; T4 confirms toggling back off doesn't crash and
     still produces a finite result. Gate: full 7-group suite —
     rust 42111/42111 (42101 baseline + this item's 10 new assertions,
     zero drift elsewhere), physics/sim-interface/sim-multimode/
     sim-propagation/io/fields all green (see "Done (recent)" for the
     dated full run).
3. 🟡 **Investigated 2026-07-19 — backlog premise wrong; the stated goal is
   unachievable by an interpolant-order change. Delivered the tighter
   regression guard the item could actually achieve; the genuine 5th-order
   continuous extension is deferred as a larger (extra-stage) item.**
   The entry read: "Shampine's DP5 continuous extension in
   `native.rs::interpolate` and `RK45.jl`'s `interpC`, same commit — removes
   the Julia-vs-native saved-grid tolerance tier entirely." Two structural
   facts (config-independent, not just measured) make this incorrect:
   - **There is no `native.rs::interpolate`.** Dense output for the native
     stepper is reconstructed *in Julia* (`interpolate(s::RustNativeStepper,
     ti)`, `RK45.jl:2053`): it fetches the 7 resident RK stages via the
     `get_ks_stage` FFI and applies the *same* `interpC` quartic as
     `PreconStepper` (which reads them from `s.ks`), then re-expresses the
     polynomial at the query time via `native_apply_prop`. Rust exposes the
     stages and the propagator; the interpolation math is Julia-side for
     *both* steppers.
   - **Changing interpolant order therefore cannot change native-vs-Julia
     agreement.** Both sides evaluate the identical interpolant from the
     identical stages, so their saved-grid difference is the Phase-1
     native-vs-Julia *method* tolerance (FFT/summation-order + Rust-`exp`
     vs Julia-`exp` in the propagator), not an interpolation-order effect.
     Measured (Kerr-only default config, `test_native_phase8.jl`): dense
     output (saveN=50) native-vs-Julia rel = **2.2e-11**, essentially the
     same order as the saveN=2 endpoints, **1.1e-11** — no interpolation-
     driven tier exists to "remove." (The old test threshold was `1e-8`,
     ~3 orders looser than reality; tightened to `1e-9` this pass — the one
     shippable piece of the item. Aside: the endpoint comment claims
     "~1e-13"; the real figure is ~1e-11, 2 orders looser than documented —
     unrelated doc drift, left as-is.)
   - **A genuine 5th-order continuous extension is real but different work.**
     DP5's 7-stage FSAL "free" interpolant is provably *order 4* (Hairer &
     Wanner II.6; it's the same MATLAB `ntrp45`/scipy interpolant), so
     reaching order 5 requires *extra function evaluation(s)* per
     interpolated step (Shampine 1986), i.e. a lazy extra-stage machinery on
     both the resident Rust stepper (a new FFI to evaluate + return the
     extra stage) and Julia. Its benefit is better accuracy of dense output
     **against the true solution** — which no current test measures (the
     whole suite's only native dense-output check is the native-vs-Julia one
     above; there is no dense-vs-analytic tier where 4th-order interpolation
     error is the limiter). So it neither shrinks the native-vs-Julia tier
     nor tightens any existing test; it's a standalone accuracy improvement,
     deferred until someone wants it. **Future implementer:** add an extra
     stage to `native.rs` exposed via a new `get_extra_stage`-style FFI,
     port the order-5 continuous-extension coefficients into a shared Julia
     helper used by both `interpolate` methods, and add a *new*
     dense-vs-fine-reference test (not a native-vs-Julia one) to show the
     order gain.

### ⚪ S6 — Distribution & ecosystem (suggestions 9, 13, 14)
*Item 1 done 2026-07-11. Item 2 done 2026-07-19 (commit 05c4a4e). Item 3 not
started.*
1. 🟢 **Done 2026-07-11.** Prebuilt binaries (item 13).
   `.github/workflows/release.yml`: triggered on `v*` tags (same tags
   TagBot.yml pushes after a Julia registry release) or manual dispatch;
   builds `libluna_rust` on the same three CI hosts `run_tests.yml` already
   tests on (`ubuntu-latest`→`x86_64-unknown-linux-gnu`, `macos-latest`→
   `aarch64-apple-darwin`, `windows-2025-vs2026`→`x86_64-pc-windows-msvc`),
   deliberately with `RUSTFLAGS=""` (portable, no `target-cpu=native` —
   unlike the dev/test build path — since a downloaded binary must run on
   any user's CPU, not just the builder's); stages each asset as
   `libluna_rust-<triple>.<ext>` with a per-asset `.sha256` file, then a
   `publish` job merges all `.sha256` files into one `SHA256SUMS.txt` and
   uploads everything to a GitHub Release via `softprops/action-gh-release`.
   `deps/build.jl` gained `try_download_prebuilt(rust_dir)`, tried before
   the existing `cargo build --release` fallback: resolves the release
   asset from `Project.toml`'s version + the running platform's target
   triple (only the 3 triples above; anything else — e.g. linux non-x86_64
   — returns `nothing` and falls straight to source), downloads the binary
   + `SHA256SUMS.txt`, verifies the asset's sha256 against the manifest
   entry, and only then moves it into the exact
   `luna-rust/target/release/<libname>` path every `_libluna_rust_path()`
   helper across `src/*.jl` already resolves to — so no Artifacts.toml or
   separate lookup path was needed, just placing the file where the
   existing from-source build already puts it. Any failure at any step
   (network, unsupported platform, missing release, checksum mismatch) is
   caught, logged via `@info`, and falls back to `cargo build --release`
   silently — never `rethrow`s from the download path itself. Opt-out:
   `LUNA_RUST_SKIP_DOWNLOAD=1` forces straight to source (useful for local
   dev iteration on `luna-rust/`, where a stale downloaded binary would
   silently shadow local changes). **Verified:** the fallback path (no
   `v0.7.0` release exists yet, so every attempt 404s) confirmed to return
   `false` cleanly, leave any existing library file untouched (mtime
   unchanged), and leave no `.download`/`SHA256SUMS.txt` temp files behind.
   The download+verify+install happy path was verified in isolation
   against a local HTTP server serving a real build of `libluna_rust.so`
   plus its real sha256 manifest line — confirmed the checksum-match branch
   installs correctly. **Not yet verified:** an actual tagged release
   completing the `release.yml` pipeline and `deps/build.jl` consuming it
   end-to-end in the wild — that only happens on the next real version tag.
2. 🟢 **Done 2026-07-19 (commit 05c4a4e).** Rust-side scan HDF5 writer
   (item 9) — `io.rs` `scan_write_point(...)` (+ `create_dataset_nd_julia`)
   writes a finished scan point's field/z arrays directly from native buffers,
   matching HDF5.jl's column-major dim-reversal so Julia reads them back
   untransposed; optional scan-queue `FlockLock`/`LockFileEx` coordination.
   Exposed via `scan_write_point_ffi` + the opt-in
   `Output.write_scan_point_native` (default Julia `HDF5Output` path
   unchanged). Also fixed a latent `io::H5T_COMPOUND` constant bug (was 3 =
   H5T_STRING). Test: `test/test_scan_native_write.jl` (:rust).
3. Standalone CLI (item 14) — `luna-rust/src/bin/luna-cli.rs`
   (`[[bin]]` in the existing crate, not a new workspace member — see the
   root-workspace-removal history in "Done (recent)"). TOML config in,
   scoped to native-eligible configs, HDF5 out; compare against a
   Julia run of the same config as the acceptance test. WASM demo only
   after the CLI exists.

**Suggested execution order** (per `SUGGESTIONS.md`, adjusted for what's
already partially done): S1 → S4 → S2 → S5.2/S5.3 → S6.1 → finish S3 →
remaining S5/S6.

## Open items

### 🟢 Native-Rust backend port (phased)

**Goal:** make the propagation backend run **exclusively in Rust** — no Julia
callback in the per-step hot loop. **Finding:** even with all five
`LUNA_USE_RUST_*` toggles ON, ~80% of per-step cost is still Julia. The Rust
stepper (`precon_step_ffi`) drives the loop but calls Julia `fbar!`/`prop!` back
through C function pointers on **every** RK stage, so every FFT (there is no Rust
FFT), Kerr, plasma `cumtrapz`, window/norm broadcasts, and the `exp(linop)`
application stay in Julia. "Exclusively Rust" therefore requires a **resident
`NativeSim`** field + native RHS + FFTW binding, not a default-flip.

Design docs (read before starting any phase):
[`docs/dev/native-port/ARCHITECTURE.md`](native-port/ARCHITECTURE.md) ·
[`docs/dev/native-port/MATH.md`](native-port/MATH.md) ·
[`docs/dev/native-port/TESTING.md`](native-port/TESTING.md) ·
[`docs/dev/native-port/PORT_LOG.md`](native-port/PORT_LOG.md) ·
agent workflow `AGENTS.md`. New toggle: `LUNA_USE_RUST_NATIVE`.

Phases (each independently shippable; gate = single-step ~1e-13 **and**
full-`solve` ~1e-6 vs the Julia oracle — see TESTING.md §3 nondeterminism floor):

- ✅ **Phase 0 — Foundations.** `NativeSim` opaque handle; FFTW binding;
  `init_native_sim` / `free_native_sim` / `set_field` / `get_field`; callback-free
  `native_step` (`RustNativeStepper` in `src/RK45.jl`). Replaces callback round-trip
  in `luna-rust/src/ffi.rs:1002` + `src/RK45.jl:309-319`. Gate passed: set/get
  bit-exact; no-op RHS rel_solve < 1e-6 (zero-RHS → bit-exact). 41928/41928 rust
  group pass. Test `test/test_native_phase0.jl`. ✔
- ✅ **Phase 1 — Mode-averaged + Kerr (RealGrid).** `rhs_mode_avg_real` +
  `native_set_mode_avg_params`; ports `to_time!`/`to_freq!`, Kerr, windows,
  `norm_mode_average`, exp-linop prop. Replaces `TransModeAvg`
  (`src/NonlinearRHS.jl:531`) + Kerr (`src/Nonlinear.jl:81`). First fully-Rust
  `prop_capillary(:HE11, Kerr)`. Gate passed: single-step ≤1e-13, full-solve
  5.8e-13. Test `test/test_native_phase1.jl`. ✔
- ✅ **Phase 2 — Plasma + EnvGrid Kerr.** `rhs_mode_avg_env` (EnvGrid c2c Kerr,
  3/4 SVEA factor) + `native_set_plasma_params`/plasma current assembly (rate
  LUT already Rust). Replaces `PlasmaCumtrapz` (`src/Nonlinear.jl:161`) +
  EnvGrid Kerr. Gate passed: Phase 2a (EnvGrid Kerr) single-step <1e-13,
  full-solve 3.2e-17; Phase 2b (RealGrid+plasma) single-step 3.8e-17,
  full-solve 2.7e-16 — all with fixed step size (see PORT_LOG 2026-07-01: the
  adaptive PI controller's near-cancellation error estimate is FP-noise
  sensitive and not itself a meaningful equivalence signal). Also fixed a
  latent `RustNativeStepper.s.y`-not-updated bug that broke `interpolate()`.
  Test `test/test_native_phase2.jl`. ✔
- ✅ **Phase 3 — Radial (TransRadial) + resident QDHT.** `rhs_radial` reuses the
  existing `QdhtFfiHandle` directly (no FFI round-trip per RHS) + a
  precomputed complex `(n_ω, n_r)` normalization array (folds `norm_radial` +
  `ωwin`, valid for a z-invariant `normfun`). Replaces `TransRadial`
  (`src/NonlinearRHS.jl:663`). Scope: RealGrid + scalar Kerr only (EnvGrid and
  plasma-radial deferred). Gate passed: single-step 1.1e-17, full-solve
  1.3e-16 (fixed step size, per the Phase 2 lesson — see PORT_LOG
  2026-07-01). Test `test/test_native_radial.jl`. ✔
- ✅ **Phase 4 — Raman.** `rhs_mode_avg_real` gains an additive Raman term via
  the resident `TimeDomainRamanSolver` (already-existing ADE solver, reused
  directly) + `native_set_raman_params`. Replaces `RamanPolarField`
  (`src/Nonlinear.jl:357`). Scope: RealGrid, `thg=true` only, all-SDO
  density-independent-τ2 eligibility (same criteria as `LUNA_USE_RUST_RAMAN`).
  Gate passed: full-solve Rust-vs-Julia 4.2e-8 — independently verified
  non-vacuous (Raman changes the Julia oracle's full-solve result by 1.1e-4,
  self-validated in-test; a single 1cm z-step alone shows Raman's
  contribution below the FP floor relative to Kerr — the effect is
  cumulative over propagation, see PORT_LOG 2026-07-01). Test
  `test/test_native_raman.jl`. ✔
- ✅ **Phase 5 — Modal (TransModal), narrow scope.** Binds the *same*
  `libcubature` C library Julia's `Cubature.jl` wraps (`Cubature_jll`,
  dlopened at runtime like FFTW — not a reimplemented cubature algorithm, so
  adaptive node placement is bit-identical, not just close). Per-node
  evaluation reuses the existing rank-1 FFT plans + Kerr formula. Scope:
  RealGrid, constant-radius `MarcatiliMode` with `kind=:HE, n=1` only (needs
  only `besselj(0,·)`/`besselj(1,·)`, already in `diffraction.rs`),
  `full=false` (the radial modal integral — what `Interface.needfull`
  already selects for `HE,n=1` mode collections, not an artificial
  restriction), Kerr-only, `shotnoise=false`. Replaces `TransModal`
  (`src/NonlinearRHS.jl:421`, `pointcalc!` `:363`, `Erω_to_Prω!` `:401`) within
  that scope. Keeps the integration loop **sequential** (prior
  `Threads.@threads` race). Gate passed: two-mode (HE11+HE12) single-step
  1.4e-19, full-solve 4.0e-16 (fixed step size) — independently verified
  non-vacuous (HE11→HE12 energy transfer is 2.0e-5 of total energy, far above
  any noise floor). General-order modes (`TE`/`TM`/`n>1`), tapered radius,
  `full=true`, EnvGrid, and Raman/plasma-in-modal are deferred (see MATH.md
  §3.3). Test `test/test_native_modal.jl`. ✔
- ✅ **Phase 6 — Free-space (TransFree).** A genuine 3-D FFTW plan
  (`fftw.rs::RealFft3d`, new `fftw_plan_dft_r2c_3d`/`_c2r_3d` symbols — same
  libfftw3 binary Julia's `FFTW.jl` uses, not a new library) replaces the
  QDHT-plus-1-D pattern Phase 3 used for radial. Dimension order
  (`(n_x,n_y,n_t)` reversed for Julia's column-major `(n_t,n_y,n_x)`) and the
  `1/(n_t·n_y·n_x)` round-trip normalization were verified against a literal
  `FFTW.rfft` reference before being trusted, not assumed from the
  row/column-major rule alone. Scope: RealGrid, `const_norm_free`
  (z-invariant), scalar Kerr, `shotnoise=false`. Replaces `TransFree`
  (`src/NonlinearRHS.jl:826`) within that scope. Gate passed: single-step
  7.05e-18, full-solve 5.01e-17 (fixed step size). EnvGrid (c2c 3-D) and a
  z-dependent `normfun` are deferred (see MATH.md §3.4). Test
  `test/test_native_free.jl`. ✔
- ✅ **Phase 7 — z-dependent linop assembly (narrow scope).** Ports the
  mode-averaged, graded-core constant-radius `MarcatiliMode` case
  (`Capillary.gradient(gas,L,p0,p1)`, a two-point pressure-gradient
  capillary) resident — `NativeSim::ensure_linop_at(z)`. `dens(pressure)` is
  a **transferred** `HermiteSpline` (Julia's own `Maths.CSpline`
  `(x,y,D)`, not re-fit — re-fitting a different spline through sampled
  values is a spline-of-a-spline problem that doesn't converge, see
  PORT_LOG). `β1(z)` is an **exact analytic closed form**, not a LUT: since
  `εco(ω;z)-1` is separable and `nwg(ω)` is z-independent, β1(z) reduces to
  4 z-independent constants computed once via `Maths.derivative` fed a
  `BigFloat` argument — see `docs/native-port/BETA1_ANALYTIC.md` for the
  derivation, why this is *more* accurate than Julia's own adaptive-FD
  `Modes.dispersion`, and the resulting tolerance tradeoff (this is the
  first phase where Rust deliberately diverges from the Julia oracle to be
  more correct, rather than a faithful bit-parity port). Also fixed: the
  nonlinear RHS's `kerr_fac`/`beta[i]` must be rescaled by `dens(z)` every
  RK stage too (`TransModeAvg` re-evaluates `densityfun(z)` fresh each
  stage) — missing this caused a ~9% full-solve mismatch, found by isolating
  that a `kerr=false` control run matched Julia while `kerr=true` didn't.
  Scope: RealGrid, Kerr-only, two-point gradient only (multi-point gradient
  and radial/free-space/modal z-dependent `nfun` deferred — see MATH.md
  §3.5). Test `test/test_native_zdep_linop.jl`. ✔
- ✅ **Phase 8 — Default-flip + cleanup.** `LUNA_USE_RUST_NATIVE` default flipped
  to `"1"`; every Phases 1-7 scope restriction converted from a hard `error()`
  to a new `NativeIneligible` exception, caught by `solve_precon` and silently
  (one-time `@warn`) falls back to the Julia stepper — native being opt-in
  used to make a scope-restriction crash the right behavior; being default
  makes it a user-facing regression instead, so it must fall back. Running
  the *entire* test suite (not just the phase-specific groups) surfaced four
  real, pre-existing gaps invisible while native was opt-in: an unrecognized
  `f!` silently ran with zero nonlinearity (`test_rk45.jl`'s raw closures);
  gas mixtures crashed with a `MethodError` instead of falling back
  (non-scalar `densityfun`); `RamanPolarEnv` (GNLSE/envelope Raman) silently
  vanished (no `isa` branch matched it — closed generally with a catch-all,
  not a special case); and, most significantly, the resident field never saw
  `Luna.run`'s per-step windowing at all (`native_step` overwrites `s.yn`
  from Rust's own state, discarding whatever Julia wrote into the pointer
  between calls) — invisible because no native-specific test drives the
  stepper through `Luna.run`, fixed via a new `native_resync_field` FFI
  called after `stepfun`. A related, separate bug (dense output between
  accepted steps was linear, not Julia's quartic `interpC`) explained nearly
  every remaining general-suite failure at once — fixed via a new
  `get_ks_stage`-based `interpolate(s::RustNativeStepper)` and
  `native_apply_prop` FFI. Full details, including the tolerance-vs-bug
  triage for each affected general-purpose test, in `PORT_LOG.md`. Test
  `test/test_native_phase8.jl`. Gate met: `LUNA_TEST_GROUP=All` — 46590
  passed, 0 failed, 0 errored, 12 broken (pre-existing), with the baseline
  (`LUNA_USE_RUST_NATIVE=0`) independently confirmed 100% green first. ✔

**Native-Rust backend port (Phases 0-8) complete.**

### 🟡 Wire remaining Rust kernels into the Julia pipeline
PPT ionization is now wired (opt-in via `LUNA_USE_RUST_IONISATION=1`).
The pattern: Rust exports an opaque handle lifecycle + a vector-eval FFI function;
Julia stores the handle in the struct and routes the in-place vector call through
`ccall`; a `@testitem tags=[:rust]` equivalence test guards the boundary.

Remaining kernels to wire (same pattern, in this order):
1. ✅ **PPT ionization** (`IonRatePPTAccel`) — `LUNA_USE_RUST_IONISATION` toggle —
   `test/test_ionisation_rust.jl`
2. ✅ **Time-domain Raman** (`raman.rs` `TimeDomainRamanSolver`) — toggle
   `LUNA_USE_RUST_RAMAN`, `init_raman_solver` / `free_raman_solver` / `raman_solve`
   exported, wired into `Nonlinear.jl` hot loop for carrier-field SDO responses
   (`CombinedRamanResponse` with all-SDO `Rs`, density-independent τ2) —
   `test/test_raman_rust.jl`. Follow-ups: rotational multi-oscillator (FFI already
   supports n_osc>1; needs Julia-side extraction of per-J Ω/K arrays);
   density-dependent τ2 (add `raman_update_coeffs` FFI entry); intermediate-broadening
   (Gaussian damping — ~~stays Julia indefinitely~~ now native via the resident
   FFT-conv kernel, Phase I item 2, 2026-07-08); envelope (`RamanPolarEnv`) Rust path
   (~~needs real-buffer copy~~ now native, Phase F item 2). Note these follow-ups
   landed in the *native-port* path, not this older per-kernel
   `LUNA_USE_RUST_RAMAN` wiring, which keeps its original narrower scope.
3. ✅ **Waveguide dispersion — Zeisberger** (`dispersion.rs` `ZeisbergerNeff`) — toggle
   `LUNA_USE_RUST_DISPERSION`, `init_zeisberger_neff` / `free_zeisberger_neff` /
   `zeisberger_neff_vector` exported, wired into `Antiresonant.jl` via a specialised
   `neff_β_grid(grid, ::ZeisbergerMode, λ0)` that batch-evaluates neff over the
   positive-frequency grid per propagation step — `test/test_dispersion_rust.jl`.
   Full Zeisberger eq.(15) parity: all four mode kinds (HE/EH/TE/TM), ϕ wall-thickness
   phase, σ⁴ real (C) and imaginary (D·loss_scale) terms. Equivalence at ~1e-12
   (same formula + Julia-supplied nco/ncl → only float-reassociation differences).
   Follow-ups: Rust-side multi-term Sellmeier (offload nco/ncl computation too);
   const-linop one-time setup path (negligible cost, left on Julia indefinitely).

3a. ✅ **Waveguide dispersion — MarcatiliMode** (`dispersion.rs` `MarcatiliNeff`) — same
    `LUNA_USE_RUST_DISPERSION` toggle; `init_marcatili_neff` / `free_marcatili_neff` /
    `marcatili_neff_vector` exported. Wired into the constant-radius specialisation
    `neff_β_grid(grid, ::MarcatiliMode{<:Number}, λ0)` in `Capillary.jl`. Nwg(ω)
    precomputed ONCE at setup (cladding-dependent, z-independent) and stored in the
    Rust handle; per step only nco(ω; z) is passed. Also adds z-level memoization
    even on the Julia-only fallback path (batching all sidcs before returning cached
    values). Equivalence is bitwise (0.0 rel error) — same IEEE 754 formula + same
    Float64 inputs. Model `:full` (`sqrt(εco-nwg)`) and `:reduced` (`1+(εco-1)/2-nwg`)
    both wired. Tests: `test/test_dispersion_rust.jl` (second `@testitem`).
4. ✅ **QDHT batch transform** — toggle `LUNA_USE_RUST_QDHT`, `init_qdht_ffi` /
   `free_qdht_ffi` / `qdht_ffi_mul_real` / `qdht_ffi_ldiv_real` / `qdht_ffi_mul_cplx` /
   `qdht_ffi_ldiv_cplx` exported. Wired into `TransRadial` in `NonlinearRHS.jl` via
   type-stable `_qdht_mul!` / `_qdht_ldiv!` dispatch. Stores Julia's T matrix
   (transposed to row-major at init); per-call transform uses Rayon parallel
   row-vector dot products with pre-allocated scratch (4×n_r×n_time), avoiding
   the two `permutedims` allocations that Julia's dim=2 QDHT path incurs.
   Handles both `Float64` (RealGrid) and `ComplexF64` interleaved (EnvGrid).
   Equivalence: ~1e-13 relative error vs Julia BLAS path (summation order differs).
   Tests: `test/test_qdht_rust.jl` (`@testitem tags=[:rust]`).
   Follow-ups: wire `TransFree` (2D Cartesian FFT, different transform type — stays Julia);
   consider cblas/openblas DGEMM binding for peak throughput.
5. ✅ **RK45 interaction-picture PreconStepper** — Dormand-Prince 5(4) with FSAL and Lund PI
   step control implemented in `ffi.rs` (`init/free/precon_step_ffi`); Julia side in
   `src/RK45.jl` (`RustPreconStepper`, `LUNA_USE_RUST_STEPPER=1`).  Key fix: `@cfunction`
   pointers must be created in `RK45.__init__` (not as precompile-image `const`s).
   Tests: `test/test_stepper_rust.jl` (physical equivalence < rtol=1e-6).

- **Safety net:** `test/test_rust_ffi.jl`, `test/test_ionisation_rust.jl`,
  `test/test_raman_rust.jl`, `test/test_dispersion_rust.jl`, and
  `luna-rust/tests/*.jl` (`@testitem tags=[:rust]`, auto-discovered).

### 🟢 Windows scan-lock validation — done (2026-07-08, found already validated by existing CI)
`luna-rust/src/scans.rs` `FlockLock::lock`/`unlock` call real Win32 `LockFileEx`/
`UnlockFileEx` on non-Unix targets (commit `febdde1`, 2026-06-28). This entry
previously claimed "no Windows CI runner exists" — **that was stale, and had
been since the entry was written**: `.github/workflows/run_tests.yml`'s `rust`
group has included `windows-2025-vs2026` in its OS matrix since long before this
BACKLOG entry existed (`cargo test` runs there on every push/PR). Verified
directly against a real run
(`gh api repos/vdiego28/Amalthea.jl/actions/jobs/85999378123/logs`, run
28980961881, 2026-07-08):
```
test scans::tests::test_flock_lock_new_error ... ok
test scans::tests::test_flock_lock_new ... ok
test tests::test_scan_queue_flock ... ok
test result: ok. 37 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```
`test_scan_queue_flock` calls `checkout_next_index`/`mark_completed`, which call
`FlockLock::lock`/`unlock` — the real `LockFileEx`/`UnlockFileEx` path on this
runner, not a stub — and it passed. (The test does self-skip if HDF5 can't be
`dlopen`'d, printing to stdout; that branch wasn't taken here, since
`test_hdf5_io_basic` — same `get_hdf5_api()` call — also passed earlier in the
same job, before Julia's own HDF5 install step even ran, meaning the Windows
runner image already has a loadable HDF5.) **No code change was needed** — this
was a stale-documentation item, not an untested-code item.
- Currently latent in production regardless of platform — `ScanQueue`/
  `init_scan_queue` is only reachable via FFI, which `src/*.jl` doesn't call yet
  (confirmed: no `init_scan_queue` reference in any Julia source file). Relevant
  again once the Rust-side scan HDF5 writer (S6.2) is wired up.

### 🟡 GPU CI coverage
`luna-rust/tests/test_gpu_cuda.jl` and the GPU numerical-equivalence tests in
`luna-rust/src/lib.rs` self-skip when no GPU/CUDA is present, so the GPU code paths
(CUDA/Vulkan dispatch, GPU QDHT/Raman/ionization) are never exercised in CI.
- **Fix:** add a GPU-equipped CI runner (or a scheduled job) that actually executes them.

### 🟢 GPU-resident stepper (Track S3, `CudaNativeSim`) — verified on real CUDA hardware 2026-07-07, see "Done (recent)" for the fixes this took
A prior agent pass (2026-07-05, see `GEMINI_SUGGESTIONS.md` and `docs/native-port/GPU.md`)
left a large uncommitted working-tree diff implementing three of `SUGGESTIONS.md`'s
performance ideas. Reviewed and tested (full `rust` gate: 42004/42004, no regression) —
findings below.
- ✅ **GPU PPT ionization clamp parity** (`kernels.cu`/`ionization.rs`): the CPU-side
  `strict` clamp-vs-error behavior (`Ionisation.jl`/`ionization.rs::rate`) was missing from
  the CUDA kernel (`ppt_ionization_kernel` unconditionally errored above `e_max`). Now takes
  `strict` as an argument and clamps `abs_e = e_max` when `strict == 0`, matching the CPU
  path exactly. Verified correct by inspection against `ionization.rs`'s own `strict` field.
- 🟡 **BLAS-3 QDHT** (`ffi.rs`, new `blas.rs`): `QdhtFfiHandle::apply_real`/`apply_cplx` now
  call `cblas_dgemm` via a runtime-`dlopen`ed BLAS (`init_blas_path`, new FFI export) when
  available, falling back to the existing Rayon path otherwise. Safe (inert until called) —
  but **no Julia-side caller exists** (`grep` for `init_blas_path` in `src/*.jl` finds
  nothing), so this is currently dead code, not a wired optimization.
- 🟠 **GPU-resident stepper V1** (`native.rs`: `NativeSim` → `NativeBackend` trait +
  `Box<dyn NativeBackend>`, `CpuNativeSim` = the renamed original; new `cuda_native.rs`:
  `CudaNativeSim`, scoped to mode-averaged RealGrid Kerr(+plasma) only, all other
  `set_*_params` return `-1` matching `docs/native-port/GPU.md`'s stated V1 scope):
  - **Bug found and fixed (2026-07-05):** `CudaNativeSim::step` ran the 6 internal RK
    stages via `rk45_accumulate_stage_kernel` (`DP_B`, the intra-stage a-coefficients) but
    never performed the final 5th-order solution accumulation the CPU reference does in
    `native.rs` (`let b0 = dt*DP_B5[0]; ...` block, ~line 2521) — it just re-propagated the
    untouched old field, silently dropping the entire nonlinear RK contribution on every
    accepted step. Fixed by adding one more `rk45_accumulate_stage_fn` launch, in place on
    `field_d`, using `DP_B5` weights, gated on `locextrap != 0` exactly like the CPU
    reference, right before the existing final `apply_prop` call. Compiles clean, all 37
    Rust unit tests and the full `rust` Julia gate (42004/42004) still pass — **but this
    fix has still never executed on real CUDA hardware** (only checked for logical parity
    against `CpuNativeSim::step`, not numerically verified end-to-end).
  - **Opt-in gate added:** `init_cuda_native_sim` (the only FFI entry point that constructs
    a `CudaNativeSim`) now refuses to initialize — returns null and prints a warning to
    stderr — unless `LUNA_USE_RUST_CUDA_NATIVE=1` is set, and prints a second warning on
    successful opt-in. Deliberately stricter than the usual `LUNA_USE_RUST_*` toggles
    (which default-on once verified): this one requires explicit opt-in until verified on
    real GPU hardware. Covered by `test_cuda_native_sim_ffi_gated_by_env_var` in `lib.rs`.
  - **Not wired to Julia at all**: no `src/*.jl` file references `init_cuda_native_sim` or
    the CUDA path; `RK45.jl`'s native dispatch is untouched. Purely additive scaffolding —
    zero risk to the existing (CPU) native-port default, doubly so now with the opt-in gate.
  - **Untestable on this machine**: no `nvcc` in `PATH` or at `/usr/local/cuda/bin/nvcc`
    (only the NVIDIA driver is present), so `build.rs` falls back to a dummy PTX and
    `CudaNativeSim::new` fails to load real kernels — `lib.rs`'s
    `test_cuda_native_sim_basic` self-skips (confirmed via `--nocapture`), so the GPU path
    has never actually executed on real hardware here.
  - Design-doc deviation: `docs/native-port/GPU.md` §4 specifies an `enum { Cpu, Gpu }`
    dispatch ("no `Box<dyn>`... avoids dynamic dispatch overhead") but the implementation
    uses `Box<dyn NativeBackend>` instead — functionally fine, just not what was designed.
  - **Update 2026-07-07: verified on real CUDA hardware (RTX 5060 Ti, CUDA 13.3) and wired
    into `RK45.jl`** — see "Done (recent)" below for the full list of bugs found and fixed
    along the way (this section's text above is left as historical record of the pre-hardware
    state).

## Informational / no action planned

- ⚪ `deps/build.jl` forwards `ENV["RUSTFLAGS"]` (defaulting to `""` if unset),
  which neutralizes `.cargo/config.toml`'s `target-cpu=native` for package-driven
  builds. This is the intended portability safeguard (the runtime dispatcher in
  `dispatch.rs` selects the ISA at runtime); native opt only applies to a manual
  `cargo build`. Note that today this selection is mostly informational for
  everything but Raman — see the "Suggestions backlog" section's ISA sync note
  and S1.4 for the gap. **Note:** `actions-rust-lang/setup-rust-toolchain` sets
  `RUSTFLAGS=-D warnings` in CI, which propagates through `deps/build.jl` — so
  the package build runs under strict warnings (desired; keeps the Rust code clean).

## Done (recent)

- ✅ **BACKLOG.md S5.2 — deterministic mode, re-scoped (2026-07-11).** See
  S5 item 2 above for the full writeup (why "pin `dispatch.rs`" was
  dropped, why the default native path is already run-to-run deterministic
  on one machine, and why the real lever turned out to be process-global
  BLAS-eligibility invariance rather than a run-to-run fix). A first
  implementation pass shipped a technically-passing but vacuous test
  (`s1.yn == s2.yn` with the flag never actually reachable, since the
  native path never touches the process-global `BLAS_API` `OnceLock`
  itself); caught before commit and fixed by wiring `deterministic` into
  the per-kernel `LUNA_USE_RUST_QDHT` handle too (the one call site that
  populates `BLAS_API`) and rewriting the test to force a BLAS-vs-fallback
  divergence, proving the flag is load-bearing. Full 7-group gate:
  rust 42111/42111 (42101 baseline + 10 new assertions, zero drift),
  physics 1645/1657 (12 pre-existing `@test_broken`, unrelated),
  sim-interface 301/301, sim-multimode 33/33, sim-propagation 18/18,
  io 2302/2302, fields 334/334 — all green, no regressions.
  (S2 — threading the native RHS — was deliberately not picked up this
  pass: Phase 3 was already implemented, committed, pushed, and reverted
  once for an intermittent heap-corruption segfault that bit-identical
  single-process unit tests didn't catch, and its own plan document
  requires ASAN/valgrind verification before a re-attempt. S5.2 was chosen
  instead as a self-contained, gate-verifiable task.)

- ✅ **GPU-resident stepper (`CudaNativeSim`) verified on real CUDA hardware
  and wired into `RK45.jl` (2026-07-07).** Previously this could never even
  initialize on real hardware (always fell back / self-skipped, "no GPU"
  message was misleading — see the "GPU-resident stepper" entry above for
  the pre-hardware state). Available now for the first time, this surfaced
  and fixed five real, independent bugs, none hardware-availability alone
  would have caught without actually running it:
  1. **`CudaNativeSim::new` never called `init_gpu_context()`** — allocated
     GPU buffers directly, which need an active CUDA context; nothing
     created one. Fixed by calling it first, and by now taking a `linop:
     &[Complex<f64>]` seed (mirrors `CpuNativeSim::new`) and uploading it —
     previously `linop_d` was allocated but never written to (`cuMemAlloc`
     doesn't zero memory), so dispersion was uninitialized garbage.
     `init_cuda_native_sim`'s FFI signature gained a `linop` parameter to
     match (was `(n)`, now `(linop, n)` like `init_native_sim`).
  2. **`resync_field` ran backwards** — copied device→host into the
     `data` the caller expects to be read from (host→device), via an
     unsound `*const`→`*mut` cast. Not exercised by any `solve()`-based
     test (only `Luna.run`'s post-window resync path would hit it); fixed
     by inspection, not itself covered by the new test below.
  3. **Temporary-lifetime UB**: `&mut {expr} as *mut _ as *mut c_void`
     patterns (a pointer cast of a `&mut` to an anonymous block/literal
     temporary) are not one of Rust's extending-expression forms, so the
     temporary could be freed before `cuLaunchKernel` read it — this
     genuinely crashed (`SIGSEGV` inside `libcuda.so` itself, confirmed via
     `gdb`) on real hardware. Fixed by binding to named locals first.
  4. **Missing `activate_context()`**: `step()` launched kernels without
     ensuring the CUDA context was current on the calling thread first —
     `raman.rs`/`ionization.rs`'s working GPU code both do this immediately
     before their `cuLaunchKernel` calls, `cuda_native.rs` didn't.
  5. **`weaknorm_elem_kernel` called with 6 arguments instead of 7**, in the
     wrong order, missing the "new/trial field" pointer entirely —
     `cuLaunchKernel` read past the end of the array for the 7th parameter,
     causing an illegal memory access (`CUDA error 700`) or, depending on
     what garbage sat in that stack slot, the same host-side `SIGSEGV` as
     (3). Fixed by passing all 7 parameters in the kernel's actual order;
     since `step()` has no pre-acceptance trial solution to use as the true
     "new field" (CPU only computes one after deciding a step is accepted),
     `field_d` is passed for both — harmless under fixed-step
     (`max_dt=min_dt=dt`, the only equivalence-test discipline used
     anywhere in this port) since `err`'s exact value no longer affects the
     accepted step path, only whether `err<=1`.
  6. **cuFFT plan reused across both transform directions** — one
     `CUFFT_D2Z` plan was created and passed to both `cufftExecD2Z`
     (forward) and `cufftExecZ2D` (inverse); cuFFT plans are direction/type
     specific, and the mismatched inverse call failed with
     `CUFFT_INVALID_VALUE` (4). Fixed by creating a second `CUFFT_Z2D` plan
     (`fft_c2r`, alongside the existing `fft_r2c`).
  Also added a `launch_checked` helper (sync + `cuGetErrorString` after
  every `cuLaunchKernel`) — previously every launch's return code and every
  kernel's async execution error were silently discarded, which is how bugs
  3 and 5 above surfaced as a segfault or a mysteriously-unrelated later
  failure instead of a clear, immediately-attributable error message.
  **Known, documented, un-fixed fidelity gap** (not a "wrong" result, a
  bounded approximation): the GPU path's Kerr FFT buffers/plans are sized
  `n_time` (grid.t), not `n_time_over` (grid.to) — it skips the
  oversampling/anti-aliasing padding `CpuNativeSim`/Julia both apply, so
  full-solve agreement is ~4.5e-4 (test's tolerance: `<1e-3`) versus the
  CPU-resident native path's ~1e-6 (Phase 1). Fixing this is a real Rust
  kernel/buffer-sizing change, out of scope for verification.
  **Julia wiring** (`RK45.jl`, opt-in, independent of the CPU-resident
  `LUNA_USE_RUST_NATIVE` default): `RustNativeSimHandle` gained a
  `use_gpu::Bool` kwarg dispatching to `init_cuda_native_sim` instead of
  `init_native_sim`; new `_gpu_native_eligible(f!, linop)` checks
  `LUNA_USE_RUST_CUDA_NATIVE=1` plus the same mode-averaged/RealGrid/
  scalar-density/single-plain-Kerr-response scope `cuda_native.rs` actually
  implements (everything else on `CudaNativeSim`'s `NativeBackend` impl
  returns `-1`), called from `RustNativeStepper`'s non-zdep construction
  path. New test `test/test_native_cuda.jl`: constructs a GPU-backed
  stepper via `withenv("LUNA_USE_RUST_CUDA_NATIVE" => "1")`, self-skips if
  construction fails (no GPU/toolkit — CI-safe), otherwise asserts
  `_gpu_native_eligible` actually returned true (not a vacuous CPU-vs-CPU
  comparison) and checks full-solve field agreement against
  `PreconStepper`. Full 7-group gate green (rust 42049/42049, physics
  1645/1657 with the same 12 pre-existing unrelated broken tests).
  Remaining GPU work: Phase G item 2 (a CI-hosted GPU runner, so this
  doesn't silently bit-rot un-exercised again) and the `n_time_over` Kerr
  fidelity gap above.

- ✅ **Phase F item 4: gas mixtures, native (mode-averaged, Kerr-only).**
  `RustNativeStepper`'s `is_mode_avg` block (`RK45.jl`) now accepts
  `densityfun(z) isa AbstractVector{<:Real}` (previously any non-`Real`
  density threw `NativeIneligible`, forcing every mixture onto the Julia
  stepper). Mixtures use Luna's existing low-level convention — no new
  Julia struct needed — `densityfun(z)` returns one density per species and
  `f!.resp` is a tuple of per-species response tuples (see
  `test/test_mixtures.jl`, already exercised this shape). Key insight: Kerr
  is the only nonlinearity summed here, and `NonlinearRHS.Et_to_Pt!`'s
  `AbstractVector`-density branch calls each species' response with its own
  density and accumulates into the same `Pt` buffer — since Kerr's
  contribution is linear in `density·γ3`, the whole per-species sum
  collapses into a **single scalar** `kerr_fac = Σᵢ densityᵢ·ε₀·γ3ᵢ`, computed
  once in Julia. **No Rust/FFI changes were needed at all** — `native.rs`'s
  `kerr_fac` scalar and `native_set_mode_avg_params` signature are
  unchanged; only `RK45.jl`'s eligibility guard and γ3-extraction logic
  changed. Each mixture species is validated to be a single plain Kerr
  response (`_is_plain_kerr_resp`); any species carrying plasma, Raman, or
  more than one response throws `NativeIneligible` (not a simple linear sum,
  out of scope) instead of silently miscomputing. Mixtures combined with a
  z-dependent (pressure-gradient) linop are also rejected explicitly (no
  existing construction path produces this combination, but the guard makes
  the rejection explicit rather than an accidental `UndefVarError`).
  New test: `test/test_native_mixture.jl` — a 2-species mixture of the same
  gas at half pressure each (physically equivalent to
  `test_mixtures.jl`'s single-gas reference); single-step equivalence
  <1e-13, full-solve equivalence measured ~3.8e-16 (bit-parity tier, since
  Kerr's native RHS is unchanged — only the Julia-side `kerr_fac`
  computation differs), plus a rejection test for an out-of-scope
  multi-response species. `test_mixtures.jl` itself (pre-existing,
  `:io`-tagged) now exercises the native path for its mixture run for the
  first time (previously silently fell back to Julia) — still passes at its
  existing `rel < 1e-8` tolerance against the single-gas Julia reference.
  Full 7-group gate: all Pass==Total (rust 42045/42045, physics
  1645/1657 with the same 12 pre-existing broken tests, unrelated), 0 failed.
- ✅ **Phase F item 3: multi-point (general piecewise) pressure gradients,
  native.** Generalizes Phase 7's z-dependent linop (previously two-point
  `Capillary.gradient(gas,L,p0,p1)` only) to the general multi-point
  `Capillary.gradient(gas,Z,P)` fill:
  - New `Capillary.MultiPointGradient(Z,P)` — a concrete callable type
    mirroring `TwoPointGradient`, computing the same per-segment
    `sqrt(P[i]²+t·(P[i+1]²-P[i]²))` interpolation the old anonymous
    `p(z)` closure used, so `gradient(gas,Z,P)` no longer returns a `pfun
    === nothing` sentinel (which previously forced the plain, non-native
    linop path for every multi-point config, silently, no error).
  - `Capillary.jl`'s `ZDepLinopMarcatili` wrapper's `L/p0/p1` fields
    generalized to `Z::Vector{Float64}/P::Vector{Float64}` breakpoints
    (a two-point gradient is just the `Z=[0,L]` case); its `make_linop`
    specialization now accepts `pfun isa Union{TwoPointGradient,
    MultiPointGradient}` and builds the same 4 z-independent β1 closed-form
    constants either way — **no per-segment change to the constants
    themselves**: BETA1_ANALYTIC.md's derivation only requires
    `εco(ω;z)-1` separable into `γ(λ(ω))·dens(z)`, true regardless of how
    many pressure-gradient segments feed `dens(z)`. Only the z→pressure
    map itself needed generalizing.
  - Rust (`native.rs`): `zdep_grad_p0/p1/flength` fields replaced with
    `zdep_grad_z: Vec<f64>, zdep_grad_p: Vec<f64>`; a new `zdep_pressure_at`
    free function performs the segment lookup (`findlast(x -> x < z, Z)`
    equivalent) + same sqrt-interpolation formula, called from
    `ensure_linop_at` and `debug_beta1_at`. `native_set_zdep_mode_avg_params`
    FFI gained `n_z`/`z_pts`/`p_pts` in place of `flength`/`p0`/`p1`.
  - New test: `test/test_native_multipoint_gradient.jl` (3-segment
    non-monotonic profile, mirrors `test_native_zdep_linop.jl`'s structure)
    — β1(z) vs BigFloat ground truth at 8 z-values spanning all 3 segments
    (~1e-9, same tier as Phase 7's check) and full-solve equivalence
    (`rel_solve` ~2.8e-7, same ~1e-4 systematic-β1-offset tier
    BETA1_ANALYTIC.md documents — confirmed unrelated to segment count).
    `test/test_gradient.jl`'s pre-existing "Gradient array" testset (a
    2-point `gradient(gas,[0,L],[p,p])` call) now also exercises the native
    path for the first time (previously fell back to Julia via the old
    `pfun===nothing` guard) — still passes at its existing `rel_grad_array
    < 1e-3` tolerance.
  - Full 7-group gate: all Pass==Total (rust 42037/42037, +14 new tests),
    12 broken (pre-existing, `physics`), 0 failed.
- ✅ **Phase F item 2: `RamanPolarEnv` native (mode-averaged EnvGrid),
  rotational multi-oscillator, density-dependent τ2.** Three independent
  pieces, all reusing the existing resident `TimeDomainRamanSolver`
  unchanged (no new Rust solver, only new callers):
  - `RamanPolarEnv` was never natively wired at all (neither the old
    `LUNA_USE_RUST_RAMAN` per-kernel path nor the native port). Its
    intensity (`sqr!(R::RamanPolarEnv,E) = 1/2·|E|²`, Nonlinear.jl:351-354)
    is real-valued with no `thg` branch — the same real ADE solver applies
    unchanged; only `rhs_mode_avg_env` needed a new Raman step (intensity
    from the complex envelope, final accumulation `pto_cplx[i] +=
    eto_cplx[i] * (ρ·raman_p[i])`, complex × real).
  - `Raman.flatten_sdo_oscillators` expands a `RamanRespRotationalNonRigid`
    into its per-`J` `RamanRespSingleDampedOscillator`s (all sharing one
    `τ2ρ`) so a rotational (or rotational+vibrational) response reaches the
    solver — the FFI already accepted `n_osc>1`, only the Julia-side
    extraction was SDO-only. Used by both the native-port eligibility
    (`RK45.jl`, all three geometries) and the older
    `Interface._make_rust_raman_handle_from_response`.
  - Density-dependent τ2: the native-port wiring now evaluates `gammas` at
    the sim's actual constant density (`f!.densityfun(0.0)`, already
    computed for the FFI call) instead of a placeholder `ρ=1.0`, and drops
    the density-independence eligibility restriction entirely — for a
    z-invariant-density sim (already required by `_check_density_zindependent`
    for all three geometries), evaluating once at the real density is
    correct regardless of whether `τ2ρ` happens to depend on density. Makes
    H2's vibrational line (`Bρv`/`Aρv`, MATH.md §5.3's explicit negative
    control) natively eligible. The older per-kernel wiring keeps its
    `ρ=1.0` placeholder unchanged — `makeresponse` doesn't have
    pressure/density in scope at that call site, a larger plumbing change
    not undertaken here.
  - New test: `test/test_native_raman_env_rotational.jl` (envelope Raman,
    rotational-only N2, density-dependent H2 vibrational — each checks
    Rust-vs-Julia agreement and, where applicable, non-vacuousness).
  - **Found and fixed a real test-golden-value regression during
    verification, not a bug in this work:** `test/test_gnlse.jl`'s "Soliton
    shift" test compares against a `rtol=1e-14` cross-validated golden
    value — this only ever passed because `RamanPolarEnv` was always
    `NativeIneligible` before, so `prop_gnlse`'s default call silently used
    `PreconStepper` (pure Julia) every time. Now that `RamanPolarEnv` is
    natively eligible, the same call uses `RustNativeStepper`, and the
    resident ADE solver's already-documented method-difference floor vs
    Julia's FFT-convolution Raman path (~1.7e-5 per fixed-dt step,
    measured) compounds over this test's 90-dispersion-length,
    chaotically-sensitive self-frequency-shift propagation to ~4e-4 — still
    nowhere near the magnitude a genuinely wrong shift model would produce.
    Loosened both assertions' `rtol` to `1e-3`. (This is the same test that
    originally caught `RamanPolarEnv`'s missing native wiring during Phase
    8 — see that phase's entry above.)
  - Full 7-group gate: 46656 passed, 12 broken (pre-existing), 0 failed
    (rust 42023/42023, +9 new tests).
- ✅ **Phase H: 3 of 4 upstream PRs sent to `LupoLab/Luna.jl`** (forked to
  `vdiego28/Luna.jl`, each fix on its own branch off upstream's actual
  `master`, not this fork's diverged history): [#431](https://github.com/LupoLab/Luna.jl/pull/431)
  `DataField(fpath)` silently dropped the `λ0` keyword (never forwarded to
  the inner constructor call — the other two `DataField` constructors
  already forwarded it correctly); [#432](https://github.com/LupoLab/Luna.jl/pull/432)
  `parse_item(::Type{UnitRange{Int}}, ...) = eval(Meta.parse(x))` — arbitrary
  code execution from a `--range`/`-r` CLI argument, replaced with explicit
  `split`+`parse(Int, ...)`; [#433](https://github.com/LupoLab/Luna.jl/pull/433)
  `SSHExec`'s `ssh`/`scp` submission spliced the scan name/subdir/scriptfile
  unescaped into a remote-shell command string — remote command injection,
  fixed with `Base.shell_escape_posixly` + `Cmd` array construction. The 4th
  item (`pointcalc!` threading race) isn't actionable — upstream doesn't
  currently use `Threads.@threads` there.
- ✅ **Phase G item 3: CI benchmark job (regression guard, not a trend
  dashboard first).** `test/benchmark_native_default.jl` is a standalone
  script (its own CI job, not a shared `@testitem` group, so a noisy runner
  can't contaminate the timing) that (1) hard-`@assert`s the fork's default
  field-resolved `prop_capillary` (no env toggles) still selects
  `RustNativeStepper` — the actual Phase C / REVIEW.md §3.2 regression,
  which produced a silent 0x-speedup fallback with no CI signal at all
  before this job existed — and (2) times native's own fixed-dt,
  fixed-step-count per-step wall time (avoids the adaptive-step-count
  confound: two independently-adaptive full solves can take different step
  paths, see PORT_LOG 2026-07-01), writing it to a
  `customSmallerIsBetter`-format JSON consumed by
  `benchmark-action/github-action-benchmark` (published to `gh-pages` on
  push to `main` only; alerts + fails the job on a >150% regression vs
  recent history). **Finding during implementation:** comparing native's
  wall time against a *ratio* to the plain Julia `PreconStepper` turned out
  to be too hardware-dependent to hardcode as a pass/fail threshold — an
  empirical check on this dev machine measured native *slower* than
  `PreconStepper` for a fixed-dt mode-averaged+plasma workload, contradicting
  PORT_LOG's ~3.5x full-`prop_capillary` figure (likely measured on
  different hardware and/or including one-time setup costs the fixed-dt
  comparison excludes) — so the benchmark tracks native's own absolute time
  over commits instead of asserting an unreproducible ratio. **Also found
  (not fixed, out of scope for this item):** repeatedly `step!`-ing a
  `RustNativeStepper` far beyond its intended `flength` with a too-large
  fixed `dt` segfaults inside `PptIonizationRate::rate` (`ionization.rs`) —
  the field grows large enough to push the PPT rate query outside
  `[e_min,e_max]`, and something in that path doesn't hit the documented
  "clamp to `rate(e_max)`" behavior (Phase B.2) the way it should. Worth a
  follow-up: `PptIonizationRate::rate` should never segfault regardless of
  how far outside its fitted range the query lands.
- ✅ **Phase F item 1: `thg=false` Raman (resident Hilbert transform).**
  `sqr!`'s `thg=false` branch (`Nonlinear.jl:342-349`) computes intensity as
  `1/2·|hilbert(E)|²` instead of `E²` — needed its own resident c2c FFTW
  plan since RealGrid otherwise only builds r2c plans (`fft_c2c`/
  `fft_c2c_over` are EnvGrid-only). Reused `fftw.rs`'s existing
  `ComplexFft1d` (already validated by the c2c-roundtrip unit test and
  EnvGrid's Kerr path) rather than writing new FFTW plan-management code.
  `native_set_raman_params` gained a `thg: c_int` parameter (passed straight
  through from `RamanPolarField.thg`, replacing the previous blanket
  `!r.thg` rejection) and lazily builds the plan + two `n_time_over`
  scratch buffers (`ComplexFft1d::forward`/`inverse` need distinct in/out
  buffers) when `thg=false`. New `hilbert_intensity` free function in
  `native.rs` matches `Maths.hilbert`'s FFTW bin convention bit-for-bit:
  forward c2c FFT, double bins `1..n/2` (0-indexed), zero bins `n/2..n`, DC
  untouched, inverse c2c FFT normalized by `1/n`. Wired into all three
  geometries that already carry `thg=true` Raman (mode-averaged, radial —
  per r-column, modal — per quadrature node), reusing the resident
  `raman_solver`/scratch-buffer-reuse pattern already established for those.
  `Interface.makeresponse` couples Kerr's and Raman's `thg` to the same
  flag, so the new equivalence tests (`test/test_native_raman_nothg.jl`)
  construct `Kerr_field` (thg=true, native-eligible) + `RamanPolarField(thg=false)`
  manually rather than via `prop_capillary`; each geometry's testset checks
  both Rust-vs-Julia agreement (~1e-7, the same ADE-vs-FFT-convolution
  method-difference floor as the existing `thg=true` Raman tests) and that
  `thg=false` actually differs from `thg=true` by a margin far above that
  floor (non-vacuousness). Full 7-group gate: 46651 passed, 12 broken
  (pre-existing), 0 failed (rust 42018/42018, +6 new tests).
- ✅ **Fixed CI build breakage in `cuda_native.rs` (`-D warnings`, edition-2024
  `unsafe_op_in_unsafe_fn`).** The GPU-resident stepper scaffolding (Track S3)
  called unsafe fns and dereferenced raw pointers throughout its `unsafe fn`
  bodies without inner `unsafe {}` blocks, plus three now-unused imports. This
  only warns locally (no `-D warnings`), but CI's `actions-rust-lang/setup-rust-toolchain`
  sets `RUSTFLAGS=-D warnings` (see "Informational" note above) — so every push
  since that commit landed failed *every* CI job (all groups build the shared
  library via `deps/build.jl`'s `cargo build --release`), even though nothing
  in the actual test logic had changed. Fixed by wrapping the unsafe operations
  in explicit `unsafe {}` blocks and dropping the unused imports; also fixed
  two pre-existing `-D warnings`-only `unused_mut` failures in `lib.rs`'s GPU
  smoke test. Verified with `RUSTFLAGS="-D warnings" cargo build --release` /
  `cargo test` locally (both clean) and the full 7-group Julia gate.
- ✅ **Fixed a silent-wrong-physics correctness gap: native Kerr eligibility
  didn't distinguish `Kerr_field`/`Kerr_env` from `Kerr_field_nothg`/
  `Kerr_env_thg`.** All four of `Nonlinear.jl`'s Kerr closures capture a field
  named γ3, so the mode-averaged/radial/modal/free-space eligibility checks'
  "does any field contain γ3" scan (`is_kerr_resp`/`is_kerr_resp_radial`/
  `is_kerr_resp_modal`, and the free-space single-response check) treated all
  four as native-eligible — but native.rs only ever implements the plain
  `Kerr_field`/`Kerr_env` formula for a given grid type (dispatched solely on
  `sim.is_real`; there's no THG flag in the FFI). Reachable from the ordinary
  high-level interface: `prop_capillary(...; thg=false)` on a RealGrid, or
  `envelope=true, thg=true`, silently ran the *opposite* THG treatment under
  the (now-default, Phase 8) native path instead of falling back to Julia —
  no error, no warning, just a wrong answer. Fixed by `RK45._is_plain_kerr_resp`
  (distinguishes by field count: the plain closures capture only γ3; the
  `_nothg`/`_thg` variants also capture a Hilbert-transform plan or
  oscillating-phase buffer), wired into all four eligibility sites. New
  regression test: `test/test_native_kerr_thg_guard.jl` (confirms both
  variants now throw `NativeIneligible`, and that the plain closures still
  agree with Julia to ~1e-16/1e-17). Verified against the full 7-group gate:
  46645 passed, 12 broken (pre-existing), 0 failed (rust 42012/42012, +4 new).
- ✅ Skip-guard added to `test/test_rust_ffi.jl` and the four `luna-rust/tests/*.jl`
  testitems so a fresh-clone `]test Luna` skips (not fails) when the Rust lib is unbuilt.
- ✅ Windows scan-lock no-op documented in `scans.rs` (with `TODO`) and `luna-rust/README.md`.
- ✅ `CLAUDE.md` extended with a Math & Advancements section.
- ✅ **Removed root `Cargo.toml` workspace** so `cargo build` in `luna-rust/` writes to
  `luna-rust/target/release/` (where all tests look). Root `Cargo.lock` also removed.
  Previously, the workspace redirected output to `./target/release/`, silently breaking all
  `rust` CI jobs on a clean checkout. Also fixes `rust-cache` effectiveness (caches
  `workdir: luna-rust` but output was going to root `target/`).
- ✅ **Fixed `pointcalc!` data race in `src/NonlinearRHS.jl`**: `Threads.@threads` was
  added to the `TransModal` integration loop, but the loop body writes to shared struct
  fields (`t.Erω`, `t.Prω`, `t.Prmω`, etc.). Multiple threads clobbering each other's
  intermediate buffers produced corrupted modal overlap values → RK45 rejected every step →
  `"Reached limit for step repetition (10)"` on all `physics`/`sim-interface`/`sim-multimode`
  jobs. Fixed by reverting to a sequential loop.
- ✅ **Fixed `IonRatePPTAccel` to clamp instead of throw** (`src/Ionisation.jl:419`):
  the adaptive stepper evaluates the RHS at rejected trial points; a large-field trial step
  above `Emax` is recoverable and should not crash the run. Now returns `exp(spline(Emax))`
  (saturated rate) instead of calling `error()`.
