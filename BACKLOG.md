# Backlog

Deferred work and known issues for Luna-Rust.jl. Severity: 🔴 correctness · 🟡 robustness/CI · ⚪ informational.

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

5. EnvGrid modal. Substantially larger than npol=2 was — needs a genuine new
   RHS variant in `native.rs` (complex time/polarisation buffers,
   `fft_c2c_over` instead of `fft_r2c_over`, `KerrScalarEnv!`/
   `KerrVectorEnv!` formulas — note `KerrVectorEnv!`
   (`3/4·((Ex²+2/3Ey²)Ex + 1/3·conj(Ex)·Ey²)`, `src/Nonlinear.jl:124-133`) is
   a genuinely different formula from RealGrid's `KerrVector!`, not just a
   complex-typed port of it), a new `is_real`/response-kind flag threaded
   through `native_set_modal_params` and `rhs_modal_pointcalc` to select the
   FFT plan and Kerr closure, and a new eligibility path in `RK45.jl` to
   distinguish `Kerr_env` from `Kerr_field` (today's `is_kerr_resp_modal`
   just greps for a `γ3` field name, which both closures have). Combines
   multiplicatively with the npol axis. Comparable in scope to Phase D.3
   (EnvGrid free-space). Remains out of scope for this session.

Gate (E.4): all 7 test groups green — 46637 passed, 12 broken (pre-existing),
0 failed/errored (rust group: 42004/42004, includes 3 new tests).

### Phase F — Native scope: Raman completions + z-dependence (🟡, medium)
1. `thg=false` Raman (needs a resident Hilbert transform — one extra c2c
   FFT pair on the existing plans).
2. `RamanPolarEnv` (envelope Raman) native + the existing
   `LUNA_USE_RUST_RAMAN` follow-ups (rotational multi-oscillator per-J
   extraction; density-dependent τ2 via a `raman_update_coeffs` FFI).
3. Multi-point pressure gradients (piecewise Phase 7 — per-segment analytic
   β1 constants; `ensure_linop_at` selects the segment).
4. Gas mixtures: per-species density vector + summed susceptibilities in
   the native RHS (removes the `densityfun isa Real` guard).

### Phase G — Platform & CI robustness (🟡, small-medium)
1. Windows scan locking via `LockFileEx`/`UnlockFileEx` (existing item
   below; needs a Windows CI runner to validate).
2. GPU CI: a scheduled CUDA-equipped job running the currently
   self-skipping GPU equivalence tests (existing item below).
3. CI benchmark job: track the Phase C benchmark over time so native-path
   regressions (like §3.2) show up as perf cliffs, not silence.

### Phase H — Upstream contributions (⚪, small)
Send back as PRs to LupoLab/Luna.jl: the `DataField(fpath)` λ0-forwarding
fix, the `parse_item` eval-injection fix, the `Cmd`-array/shell-escaping
scan hardening, and the `pointcalc!` race fix if upstream ever re-adds
threading. Reduces future divergence surface.

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
[`docs/native-port/ARCHITECTURE.md`](docs/native-port/ARCHITECTURE.md) ·
[`docs/native-port/MATH.md`](docs/native-port/MATH.md) ·
[`docs/native-port/TESTING.md`](docs/native-port/TESTING.md) ·
[`docs/native-port/PORT_LOG.md`](docs/native-port/PORT_LOG.md) ·
agent workflow [`AGENTS.md`](AGENTS.md). New toggle: `LUNA_USE_RUST_NATIVE`.

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
   (Gaussian damping — stays Julia indefinitely); envelope (`RamanPolarEnv`) Rust path
   (needs real-buffer copy for complex→real conversion).
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

### 🔴 Real file locking for parameter scans on Windows
`luna-rust/src/scans.rs` `FlockLock` is a **no-op on non-Unix targets** (`#[cfg(not(unix))]`
returns an empty struct; `lock()`/`unlock()` do nothing). Concurrent scans on Windows are
therefore not process-safe and may race on the shared HDF5 queue file.
- **Fix:** implement locking via Windows `LockFileEx` / `UnlockFileEx`.
- **Blocker:** needs a Windows runner to validate; untestable from Linux/macOS.
- Currently latent — `ScanQueue` is only reachable via FFI, which `src/` doesn't call yet.

### 🟡 GPU CI coverage
`luna-rust/tests/test_gpu_cuda.jl` and the GPU numerical-equivalence tests in
`luna-rust/src/lib.rs` self-skip when no GPU/CUDA is present, so the GPU code paths
(CUDA/Vulkan dispatch, GPU QDHT/Raman/ionization) are never exercised in CI.
- **Fix:** add a GPU-equipped CI runner (or a scheduled job) that actually executes them.

### 🔴 GPU-resident stepper (Track S3, `CudaNativeSim`) — uncommitted, bug fixed, gated behind explicit opt-in, still unverified on real hardware
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
  - **Do not wire this into `RK45.jl`/the runtime dispatcher until it's been run and
    verified against the Julia oracle on real CUDA hardware** (per
    `[[feedback_verify_actual_gate_not_subset]]` — a GPU path is a gate of its own, not a
    subset of the CPU gate that happens to pass). The env-var gate is a stopgap, not a
    substitute for that verification.

## Informational / no action planned

- ⚪ `deps/build.jl` forwards `ENV["RUSTFLAGS"]` (defaulting to `""` if unset),
  which neutralizes `.cargo/config.toml`'s `target-cpu=native` for package-driven
  builds. This is the intended portability safeguard (the runtime dispatcher in
  `dispatch.rs` selects the ISA at runtime); native opt only applies to a manual
  `cargo build`. **Note:** `actions-rust-lang/setup-rust-toolchain` sets
  `RUSTFLAGS=-D warnings` in CI, which propagates through `deps/build.jl` — so
  the package build runs under strict warnings (desired; keeps the Rust code clean).

## Done (recent)

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
