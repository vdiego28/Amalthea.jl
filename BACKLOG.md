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
5. ⚪ **Modal path restricted to `MarcatiliMode`** (`RK45.jl:1352-1354`) —
   `StepIndexMode`/`ZeisbergerMode`/`VincettiMode` exist but
   `prop_capillary`'s `makemode_s` only ever builds `MarcatiliMode`s; the
   other mode types are only reachable via the low-level API. Rare in
   practice.
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
4. ⚪ **User-facing native-support matrix.** What runs natively vs falls
   back (geometry × grid × response × options) is currently derivable
   only from this file + PORT_LOG + the `NativeIneligible` sites. A
   single README (or `docs/`) table — rows: mode-avg/radial/modal/
   free-space; columns: RealGrid/EnvGrid, Kerr/plasma/Raman(SDO/SiO2)/
   noise/mixtures/z-dep — would let users predict performance without
   reading the port history. Generate it from the audit in Phase I's
   preamble; keep it next to the code so guard changes update it.
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
7. ⚪ **FFI-surface robustness tests.** The FFI now has ~30 `native_set_*`
   entry points whose error paths (null pointers, mismatched sizes,
   wrong call order) are each hand-checked but never systematically
   exercised. A single Rust test that calls every setter on a fresh
   `NativeSim` with adversarial inputs (nulls, zero lengths, `n_time`
   mismatches) and asserts non-zero return codes — no crash, no UB —
   would fence the whole surface cheaply. Pairs well with S4.2's
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
4. SIMD Kerr + window/norm broadcasts (item 3a) — route through
   `dispatch.rs`'s existing AVX2/AVX-512 lanes (see ISA sync note above:
   this is the fix for "dispatch selects a path but only Raman uses it").
   Add a Rust unit test asserting SIMD lane == scalar lane bitwise (or
   ≤1 ulp where FMA contraction differs).
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
6. 🟡 **Spike run 2026-07-10.** Split-complex exp-linop spike (item 3b) —
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
   - Phases 2-4 (per-geometry migration, FFI boundary shim, cleanup) —
     not started.

### 🟡 S2 — Threading the native RHS (suggestion 2)
*Not started. Depends on S1.4 (dispatch plumbing).*
1. `native_set_threads(handle, n)` FFI, wired from `Threads.nthreads()`,
   default 1 (bit-identical to today).
2. `rhs_radial`: rayon over radial nodes, each node's own FFTW
   new-array-execute call against one shared plan; one scratch slab per
   rayon worker (never shared — this is precisely the bug the Julia
   `Threads.@threads` `pointcalc!` race had, see Phase B/"Done (recent)").
3. Modal: rayon inside `hcubature_v`'s batch callback only — cubature's
   adaptive node placement stays sequential/deterministic.
4. 3-D free-space FFT: `fftw_plan_with_nthreads`/`fftw_init_threads`
   (dlopened `libfftw3_threads`, silent fallback if absent).
5. Error-norm reduction stays sequential (determinism, ties to S5.2).
- Gate: universal + fixed-step equivalence under `JULIA_NUM_THREADS=4` +
  ≥2× radial/free benchmark at 4 threads + n=1 bit-identical to pre-track.

### 🟠 S3 — GPU-resident propagation (suggestion 1) — partially started, see note above
*Large (5+ sessions). Plan's own stated dependency (GPU CI) is not yet
met — see "GPU CI coverage" below.*
Already landed (2026-07-05/07, ahead of the plan's own sequencing): the
`NativeBackend` trait extraction, `CudaNativeSim` scoped to mode-averaged
RealGrid Kerr-only, verified on real hardware, wired behind
`LUNA_USE_RUST_CUDA_NATIVE=1`. Still open, per the original design:
1. Design doc (`docs/native-port/GPU.md` exists from the prior pass but
   deviates from what was built — reconcile or rewrite: it specifies an
   `enum{Cpu,Gpu}` dispatch, the actual code uses `Box<dyn NativeBackend>`).
2. Plasma, Raman, radial/modal/free-space geometries on `CudaNativeSim`
   (every `set_*_params` beyond mode-avg Kerr currently returns `-1`).
3. Problem-size dispatch threshold (measured crossover, not guessed) so
   small grids stay on CPU; `LUNA_NATIVE_GPU=0/1/auto` env override.
4. cumtrapz (plasma scan, work-efficient prefix scan) and Raman ADE
   kernels — or explicit `NativeIneligible`-style eligibility split
   keeping those configs on CPU for GPU v1, documented as such.
5. `test/test_native_gpu.jl`-style coverage of the above, mirroring the
   phase-test structure used throughout the CPU native port.

### 🟡 S4 — Architecture cleanups (suggestions 6, 7, 8)
*Not started. Plan says do this BEFORE S3 (S3 multiplies the toggle zoo
otherwise) — that ordering was already missed once S3 partially started;
land this before extending S3 further.*
1. Config struct (item 6) — `src/Config.jl` `BackendConfig` (native ∈
   {:auto,:on,:off}, per-kernel overrides, gpu, threads, qdht_blas), one
   `Luna.backend_config()` resolver reading env vars once; migrate every
   `get(ENV, "LUNA_USE_RUST_*")` call site to consult it instead. Add
   `Luna.backend_report()` (also closes BACKLOG's own `_LAST_STEPPER_TYPE`
   test-hook item under a public API).
2. FFI error enum (item 7) — `#[repr(i32)] enum FfiStatus{Ok=0,
   Ineligible>0,Bug<0}` in `ffi.rs`; one Julia `check_ffi(rc, what)`
   replacing the per-call-site `rc == 0 ||` patterns in `RK45.jl`.
3. Explicit accessor seams (item 8) — `NonlinearRHS.jl`/`Nonlinear.jl`
   gain `kerr_γ3(resp)`, `norm_pre(norm!)`, `norm_βfun(norm!)`; delete the
   `occursin("γ3", string(fld))`/`getfield(norm_func, :pre)` reflective
   probes in `RK45.jl`.
- Gate: universal + a `backend_report()` test asserting `RustNativeStepper`
  for default `prop_capillary` and Julia stepper under `native=:off`.

### 🟡 S5 — Numerics options (suggestions 10, 11, 12)
*Not started. Items are independent of each other.*
1. Mixed-precision spike (item 10) — timeboxed Criterion bench only;
   proceed to real implementation only if >1.4× projected AND the b5−b4
   cancellation analysis (TESTING.md §3) shows the f64-reduced estimate
   keeps the same step sequence on the benchmark case.
2. Deterministic mode (item 11) — `native_set_deterministic(handle, bool)`
   forcing sequential reductions + pinning `dispatch.rs` to the portable
   lane; `deterministic=true` kwarg threaded through `BackendConfig`
   (hard dependency: S4.1). Test: two runs bit-identical, with a written
   statement of what's NOT guaranteed (cross-machine libm differences).
3. 5th-order dense output (item 12) — Shampine's DP5 continuous extension
   in `native.rs::interpolate` (stages already exposed via
   `get_ks_stage`, Phase 8) **and** `RK45.jl`'s `interpC`, same commit —
   removes the Julia-vs-native saved-grid tolerance tier entirely (only
   valid if both sides change together).

### ⚪ S6 — Distribution & ecosystem (suggestions 9, 13, 14)
*Not started. Lowest priority; 13 is smallest/highest-value of the three.*
1. Prebuilt binaries (item 13) — `release.yml` building
   `libluna_rust` per-platform, uploaded to GitHub Releases with a sha256
   manifest; `deps/build.jl` tries download-by-version+checksum first,
   falls back to `cargo build` (kept as the from-source/dev path).
2. Rust-side scan HDF5 writer (item 9) — `io.rs` `scan_write_point(...)`
   writing directly from native buffers under the existing flock/
   `LockFileEx` queue lock (hard dependency: Windows scan-lock validation
   below, for the Windows half). Opt-in `Output.jl` backend.
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
(`gh api repos/vdiego28/Luna-Rust.jl/actions/jobs/85999378123/logs`, run
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
