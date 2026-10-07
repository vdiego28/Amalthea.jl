# Native-Rust Backend Port — Testing & Equivalence

> Status: design doc for the phased port. Phases 0-8 are implemented and
> passing (see `docs/dev/native-port/PORT_LOG.md`) — the native-Rust backend
> port is complete, and the follow-on scope phases (BACKLOG.md Phases D-I,
> all ✅ 2026-07-08) extended it to essentially every configuration the
> high-level API can construct. The testing discipline below (tolerance
> tiers, fixed-step full-solve, non-vacuousness / triangulation) applied
> unchanged to all of them.
> Companion docs: [ARCHITECTURE.md](ARCHITECTURE.md), [MATH.md](MATH.md),
> [PORT_LOG.md](PORT_LOG.md).

Every phase ships **only** when an equivalence test proves the native path
reproduces the Julia path within the tolerance tier justified below. The Julia
path is the **oracle** — this is the central reason it is retained as a fallback
(ARCHITECTURE §4.3).

## 1. How to write an equivalence `@testitem`

Mirror `test/test_stepper_rust.jl`. Every Rust test is a `@testitem` tagged
`:rust` with a **skip-guard** so a fresh clone without the built `.so` skips
(not fails):

```julia
using TestItems

@testitem "Native <phase> equivalence" tags=[:rust] begin
    import Test: @test, @testset
    using Amalthea
    import Logging: with_logger, NullLogger
    import LinearAlgebra: norm

    # ── skip guard (copy verbatim) ──────────────────────────────────────────
    libname = if Sys.iswindows(); "amalthea.dll"
              elseif Sys.isapple(); "libamalthea.dylib"
              else; "libamalthea.so"; end
    libpath = joinpath(@__DIR__, "..", "amalthea", "target", "release", libname)
    if !isfile(libpath)
        @warn "Skipping: shared library not found at $libpath. " *
              "Build with `cargo build --release` in amalthea/."
        return
    end

    # ── run BOTH paths and compare the final spectrum ───────────────────────
    @testset "<geometry> equivalence" begin
        args = (radius, L, gas, pres)
        kw = (; λ0, τfwhm=τ, energy, modes=:HE11, loss=false,
                saveN=2, trange=0.5e-12, λlims=(200e-9, 4e-6))

        out_julia = withenv("AMALTHEA_USE_RUST_NATIVE" => "0") do
            with_logger(NullLogger()) do
                prop_capillary(args...; kw...)
            end
        end
        out_rust = withenv("AMALTHEA_USE_RUST_NATIVE" => "1") do
            with_logger(NullLogger()) do
                prop_capillary(args...; kw...)
            end
        end

        rel = norm(out_rust["Eω"][:,end] - out_julia["Eω"][:,end]) /
              norm(out_julia["Eω"][:,end])
        @test rel < 1e-6        # tier — see §2, justify per phase
    end
end
```

Notes:

- Toggle **both** paths explicitly via `withenv`, never by relying on the
  default or mutating global state. Native defaults on, so omitting the
  `"0"` around the oracle produces a vacuous native-vs-native comparison.
- `prop_capillary` **requires** `λlims`; it does **not** accept `stepfun`,
  `rtol`, or `atol` as kwargs (learned the hard way — see PORT_LOG seed).
- Compare the final-z spectrum `Eω[:,end]`. For stronger checks also compare an
  intermediate save and a derived observable (e.g. spectral energy).
- Prove the feature under test changes the explicit Julia oracle by more than
  the comparison tolerance. The CUDA test failed this rule: a zero-nonlinearity
  backend passed because its `1e-3` tolerance exceeded the entire `4.5e-4`
  nonlinear effect.
- Place the file in `test/` and register any new testitem in the corresponding
  `test/*_test_timings.txt` manifest used by the parallel CI scheduler.
  The serial `@run_package_tests` runner discovers testitems automatically.

### Backend observability and dispatch gates

`RustNativeStepper` exposes the internal diagnostic
`RK45._native_backend(s)`, which returns exactly `:cpu` or `:cuda` from the
backend selected at construction. Use this accessor when a test needs to
prove dispatch; `s isa RustNativeStepper` proves only that the resident API
was used, not which resident implementation owns the field. The accessor does
not query CUDA or change eligibility.

Pure eligibility, unsupported-response, capacity-boundary, and CPU-fallback
assertions must run before any CUDA-device gate. A supported configuration may
be constructed with `gpu_dispatch=:off` or a below-threshold `:auto` policy to
prove `:cpu` on every host. A `gpu_dispatch=:on` test on a CPU-only host should
assert only the pure eligibility result; it must not attempt a supported CUDA
construction. After the hardware gate succeeds, assert `:cuda` before reading
GPU stages or comparing trajectories. Z-dependent native constructors are
CPU-only and should assert `:cpu` explicitly.

## 2. Tolerance tiers (and the reason each applies)

Pick the **tightest** tier the math justifies. A test that passes at a looser
tier than its math allows is hiding a bug.

| Tier | Threshold | When it applies | Example (wired) |
|------|-----------|-----------------|-----------------|
| **bitwise** | `== 0.0` | identical IEEE-754 formula **and** identical Float64 inputs | Marcatili neff (`test_dispersion_rust.jl`) |
| **reassociation** | `~1e-13` | same formula, summation/BLAS order differs (FFTW parity, QDHT, dot products) | QDHT (`test_qdht_rust.jl`), Zeisberger (~1e-12) |
| **method/spline** | `~1e-8` | LUT/spline interpolation or a different-but-equivalent algorithm | PPT ionization (`test_ionisation_rust.jl`) |
| **FFT-method + floor** | `~1e-6` | FFT method differences **and** the run-to-run nondeterminism floor (§3) | RK45 stepper (`test_stepper_rust.jl`) |
| **deliberate divergence** | `~1e-4` (config-dependent) | Rust computes a *more accurate* value than Julia's own oracle on purpose, and the resulting small systematic (non-random) offset accumulates coherently over propagation length/bandwidth | Phase 7 β1(z) (`test_native_zdep_linop.jl`, see `BETA1_ANALYTIC.md`) |
| **different backend** | `~1e-8`-`1e-6` (config-dependent) | Two configs in the *same* comparison legitimately execute on different steppers (one `NativeIneligible`, one not) — as of Phase 8 this is possible for the first time, since native is the default rather than opt-in | `test_mixtures.jl` (mixture vs single-gas), `test_tapers.jl` (Function-radius vs constant-radius) |

This last tier is different in kind from the others: it is not "we haven't
converged Rust to match Julia yet," it is "Rust and Julia will never
converge further, because Rust is right and Julia's own value has a real,
repeatable, tiny error against the true derivative." Before reaching for
this tier, do the two checks that prove it isn't secretly the other three
tiers in disguise: (1) a `kerr=false`/linear-only control run should show
the *same* magnitude as the full nonlinear run (proves it's not a bug in
some other piece of the RHS), and (2) an independent BigFloat/higher-precision
ground truth should confirm Rust's value, not just Julia's, is the accurate
one. See `BETA1_ANALYTIC.md` §4 for a worked example of both checks.

**Per-phase target.** Because the native port binds the **same FFTW** (so
transforms are bit-parity) and copies Julia's coefficient arrays in, most phases
should land in the **reassociation tier (~1e-13)** for a single deterministic
step. Whole-`solve` comparisons that accumulate many adaptive steps fall to the
**~1e-6 floor tier** (§3). State both numbers in the PORT_LOG entry: the
single-step tightness (proves the math) and the full-run tolerance (proves the
trajectory).

## 3. The run-to-run nondeterminism floor (critical)

**The Julia stepper alone varies ~2e-8 run-to-run** for a typical capillary
setup, even single-threaded, because FFTW's summation order is not reproducible
across invocations. This is a hard floor: **no equivalence threshold for a full
`solve` can sit below ~2e-8**, regardless of how perfect the native code is.
This is why `test_stepper_rust.jl` uses `1e-6` (comfortably above the floor),
not `1e-10` (numerically impossible).

Two consequences for the port:
1. **Full-run equivalence tests use the ~1e-6 tier.** Do not tighten them below
   the floor; a "failing" test there is measuring FFTW noise, not a port bug.
2. **For tight per-step checks**, compare a **single deterministic RHS/step
   evaluation** (same input field, one `fbar!`/one `step!`) rather than a full
   adaptive run. A single evaluation has no accumulated step-sequence divergence
   and should hit ~1e-13. This is the test that actually proves the math is right.

### Tighter local checks
For local debugging, pin FFTW to one thread and `:estimate` planning to reduce
(not eliminate) variance:

```julia
import FFTW
FFTW.set_num_threads(1)
# tests already use :estimate planning (see CLAUDE.md)
```

The step controller also matters: once two paths' `err` estimates differ by even
1 ULP near an accept/reject boundary, they take different step sequences and
diverge within the tolerance band (MATH §2.2). Single-step comparison sidesteps
this entirely.

**Phase 2 postmortem — this bit us for real.** Phase 2a's full-solve test
initially failed at 9.64e-5 (vs the 1e-6 tier) despite the single-step test
passing at <1e-13. Root cause: the embedded RK45 error estimate is a
near-total cancellation (`b5-b4=0` in the Butcher tableau) — early in a weakly
nonlinear propagation, `err` sits at the ~1e-15 floor, where FP-summation-order
noise between Julia and Rust shows up as a ~20% *relative* disagreement in
`err`. The PI controller amplifies that into a different `dtn` choice, and the
two adaptive integrators diverge onto different step sequences that land at
different z — so the "full-solve" comparison was comparing the field at two
different points in space, not detecting a state-accumulation bug. Confirmed
by forcing `max_dt=min_dt=dt` on both steppers: agreement collapsed to
~1e-17 all the way to `flength`.

**Recommended full-run test shape (adopted in `test_native_phase{1,2}.jl`):**
construct both steppers with `max_dt=dt, min_dt=dt` for the full-solve
testset specifically (leave the single-step testset as-is). This forces an
identical step-size sequence — sidestepping the adaptive-path-divergence
confound entirely — while still exercising genuine multi-step state
accumulation, which is what the full-run tier is supposed to test. Apply this
to every future phase's full-solve test, not just the ones where it happens to
bite (Phase 1/2b's `err` values were "healthy" — far from the cancellation
floor — so their raw-`yn` full-solve tests happened to pass anyway; that's
coincidence of regime, not immunity to the same mechanism).

## 4. Per-phase acceptance criteria

| Phase | Status | What to test | Test file | Single-step tier | Full-run tier |
|-------|--------|--------------|-----------|------------------|---------------|
| 0 | ✅ done | set/get round-trip bit-exact; no-op RHS reproduces Julia stepper | `test/test_native_phase0.jl` | bitwise (round-trip) | ~1e-6 |
| 1 | ✅ done | mode-avg + Kerr `prop_capillary(:HE11)`, RealGrid | `test/test_native_phase1.jl` | <1e-13 (achieved) | 2.75e-16 (fixed dt) |
| 2 | ✅ done | EnvGrid Kerr (2a) + plasma/RealGrid (2b) | `test/test_native_phase2.jl` | <1e-13 (achieved) | 3.19e-17 / 2.73e-16 (fixed dt) |
| 3 | ✅ done | radial + resident QDHT (RealGrid + scalar Kerr) | `test/test_native_radial.jl` | 1.1e-17 (achieved) | 1.3e-16 (fixed dt) |
| 4 | ✅ done | Raman (carrier SDO, thg=true, all-SDO eligibility) | `test/test_native_raman.jl` | 0.0 (see note) | 4.2e-8 (achieved) |
| 5 | ✅ done | modal + overlap cubature (`HE,n=1`, `full=false`, Kerr-only) | `test/test_native_modal.jl` | 1.4e-19 (achieved; ~1e-10 tier) | 4.0e-16 (achieved; fixed dt) |
| 6 | ✅ done | free-space 3-D FFT (RealGrid, const_norm_free, Kerr-only) | `test/test_native_free.jl` | 7.05e-18 (achieved) | 5.01e-17 (achieved; fixed dt) |
| 7 | ✅ done | z-dependent linop (mode-avg, graded-core, two-point pressure gradient) | `test/test_native_zdep_linop.jl` | <1e-9 (β1 vs BigFloat truth, achieved); ~1e-12 (`dtn`/`err`, achieved) | <1e-3 tier (measured ~2.7e-7 post-Phase-8-precision-fix, deliberate-divergence, see §2) |
| 8 | ✅ done | default-flip: existing suite green with native as default | `test/test_native_phase8.jl` + full suite | — | 46590 pass / 0 fail / 0 error / 12 broken (pre-existing), 46602 total |
| S5.3 | ✅ done | order-5 dense output + deferred FSAL carry on Julia/CPU-native; measured CUDA order-4 fallback | `test/test_native_dense_order5.jl` | CPU order → 5; CUDA order → 4; native-vs-Julia ~1e-17 | full gate green |

Phase 5's single-step tier is documented looser (~1e-10) than the FFTW-only
phases, not because cubature node placement is algorithm-dependent — it binds
the *same* `libcubature` C library Julia calls (`Cubature.jl` is a thin
`ccall` wrapper, confirmed via `Cubature.Cubature_jll.libcubature` and
`nm -D libcubature.so`), so node placement is bit-identical by construction.
The tier is looser because the per-node mode-field synthesis needs
`besselj(0,·)`, and the existing Rust `j0` (`diffraction.rs`) agrees with
`SpecialFunctions.besselj` to ~1e-15 *absolute*, not bitwise — verified
standalone before implementing, not assumed. In practice the achieved
number (1.4e-19) came in far under even the tight tier at this problem size;
the ~1e-10 ceiling is the honest one to keep asserting.

Phase 4's single-step result (`0.0`, exact) is **not a vacuous test** — verified
via a three-cell diagnostic (PORT_LOG 2026-07-01): Raman's raw per-step RHS
contribution is ~2e-16 relative to Kerr's at these test parameters (right at
the FP floor for a single 1cm z-step — physically expected, since
Raman-induced spectral changes are cumulative over propagation, unlike Kerr
self-phase-modulation). The full-solve testset is the meaningful gate and is
self-validating: it independently asserts Raman changes the Julia oracle's
result (1.1e-4, far above any noise floor) *before* asserting Rust matches
Julia on that changed result (4.2e-8). Any future Raman-adjacent full-solve
test should include the same "does this feature actually change the
reference result" sanity assertion — a passing comparison between two paths
that both silently exclude the feature under test proves nothing.

Phase 7's full-run tier (~1e-3, measured ~2.7e-7 post-Phase-8-precision-fix,
see `BETA1_ANALYTIC.md` §6) is the widest of any phase, and is the **only**
phase where the widening is deliberate rather
than a limitation to work around — see the "deliberate divergence" tier in
§2 and `BETA1_ANALYTIC.md`. The single-step tier is still tight (β1 itself
is verified to <1e-9 against a BigFloat ground truth, and `dtn`/`err`
agreement is ~1e-12); it is specifically the *coherent accumulation* of
β1's tiny systematic offset from Julia's own value, over a broad-bandwidth,
multi-step propagation, that produces the wider full-run number. A
narrower-bandwidth or shorter-fibre config would show a smaller number
without any code change.

Phase 8's gate is the widest in *scope* (the entire suite, not a phase-specific
subset) but not in *tolerance* — most of its failures turned out to be real
bugs (see PORT_LOG), fixed properly rather than tolerance-widened. Only two
tests legitimately needed a tolerance change, and for a reason specific to
Phase 8: a config comparison where the two sides now execute on genuinely
different backends (native vs `NativeIneligible`-fallback Julia) for the
first time — see the "different backend" tier above. Before reaching for
that tier on any future test failure, check first whether both sides of the
comparison are actually eligible for the same backend; if they are, a
failure is a real bug, not a tolerance problem.

**GPU-specific acceptance rule.** A self-skipping CUDA test passing on a
CPU-only CI runner proves only that the skip guard works. Every GPU correctness
slice requires all of:

1. a CPU-native and Julia-oracle control showing the intended nonlinear effect;
2. a GPU stage-derivative check whose scale is comparable to CPU native, not
   approximately zero;
3. a full-solve tolerance tighter than that measured nonlinear effect;
4. a recorded run on real CUDA hardware; and
5. eventually, a standing CUDA CI job so the path cannot rot silently.

Adaptive-controller changes additionally require a deliberately rejected
trial whose field remains unchanged, a controller-selected retry, and an
adaptive trajectory against CPU native/Julia. The 2026-07-27
`test_native_cuda.jl` extension is the reference: both Kerr and Kerr+PPT reject
and retry, with full adaptive trajectory differences of `5.42e-15` and
`2.24e-15` on the RTX 5060 Ti.

### CPU optimization/concurrency gate (2026-08-24)

**Embedded fourth-order mode (PLANS §18).** Test both propagation modes against
an independent nonlinear analytic solution and test each solver against a fresh
restart from its own accepted endpoint. False-mode interpolation must converge
to y4 from inside the interval and retain fourth-order dense convergence.
Exercise nonzero linear operators, rejected attempts/retries, and repeated
dense queries before the next step. A Julia/native parity check alone cannot
detect shared invalid k7 reuse. The y4/y5 gap must exceed the continuity test's
tolerance. Default fifth-order FSAL and dense-order gates remain required.

Allocation removal is accepted only with the existing rejected-step,
`locextrap=false`, dense-output, and callback/window lifecycle suites green.
The matched medium audit additionally records Julia-visible allocations: the
optimized native fixed-step cells are 96 bytes each, down from
16,616–787,000 bytes; complete adaptive solves are 480–1,088 bytes, down from
49,608–18,885,744 bytes.

QDHT policy coverage must exercise real and complex forward/inverse transforms,
round trips, invalid FFI modes, and a resident radial trajectory under
`off`/`auto`/`on` plus deterministic override. On the current workload,
`auto == on`, `off == deterministic`, and the two kernels agree at
`rtol=1e-12` while differing bitwise as expected from summation order.

Raman SIMD coverage compares the scalar oracle with the dispatched kernel at
1, 2, 3, 4, 5, 49, 50, and 65 oscillators, multiple time lengths, and
adversarial signs at `2e-13`. AArch64 code must cross-compile cleanly and the
Apple quick test supplies the runtime NEON gate on Apple hardware.

Julia modal batching requires exact sequential/four-thread callback arrays
under `FFTW_ESTIMATE`, forced-GC repeats, and proof that stateful closures stay
sequential. Queue tests require simultaneous scans, exact-once result files,
failure marking, stable queue removal, one-thread worker topology, concurrent
resident handles, and no leaked `Distributed` workers.

## 5. Commands

Use the recorded local gate for work-item validation:

```bash
# CPU build + Cargo tests + Rust/native Julia group
python3 test/validate.py

# Include affected groups, or run all eight
python3 test/validate.py --groups rust physics --max-workers 4
python3 test/validate.py --all --max-workers 4

# Required CUDA build/tests: run on the host, outside the agent sandbox
PATH=/usr/local/cuda-13.3/bin:$PATH python3 test/validate.py --cuda
```

The wrapper builds into `amalthea/target/release`, disables prebuilt downloads,
and verifies Julia loads both the package and RK45 library from this checkout.
It also runs `julia --startup-file=no --project test/run_rust_hdf5.jl` as a
required standalone HDF5 gate, recorded in `rust-hdf5-tests.log`. The launcher
passes Julia's resolved HDF5 library and dependency search paths to a fresh
Rust process with `AMALTHEA_REQUIRE_HDF5_TESTS=1`; unavailable HDF5 fails this
gate. Its five serial tests check real/complex dataset readback, exact buffer
lengths, scalar and empty datasets, rank/extent validation, queue-state
preservation and safe initializer failures. This includes HDF5 initialization
before Julia has initialized the library in the child. Rejected transfers
must leave existing values and caller-buffer sentinels unchanged. The same
gate runs in the Linux, macOS and Windows Rust CI jobs after Julia dependency
setup. A separate required integration test starts four independent native
queue workers, verifies concurrent initialization and exactly-once claims, and
checks persisted completion/failure states and preservation of existing claims
on reopening. It runs against both a new queue and a queue with saved progress;
child execution is bounded and failure logs are retained. This tests native
file locking separately from Julia's own distributed queue. Ordinary Cargo-only tests keep
HDF5 optional; a skipped optional test is not HDF5 acceptance.
The Julia native writer selects `HDF5.API.libhdf5` during module initialization
when `AMALTHEA_HDF5_LIB` is unset, so custom depots work without scanning
unrelated cached versions. An explicit override is preserved. Exercise
`test_scan_native_write.jl` in a fresh process without that override when
changing this integration.
CPU mode sets `AMALTHEA_CUDA_BUILD=off` and strict-CUDA tests off. `--cuda`
sets the build to `required` and `AMALTHEA_REQUIRE_CUDA_TESTS=1`, and always
includes the Rust group: missing real PTX or GPU dispatch fails validation.
It does not globally force GPU dispatch for CPU control tests. CUDA commands
still require host execution; the wrapper does not request sandbox escalation.
Explicit RUSTFLAGS are preserved; unset means empty for portability. Encoded
Cargo flags and a cross-compilation target override are removed so this gate
builds a host-loadable library with the recorded RUSTFLAGS.
In a restricted agent sandbox, Julia's normal precompile depot may be
read-only. Prefix `JULIA_DEPOT_PATH` with a writable temporary depot while
retaining the installed depot as a later entry. The `io` group also needs host
execution: its `Distributed` queue tests bind loopback sockets and its output
test writes under `~/.luna/output_test`. A sandbox failure at either boundary
is not a scientific test result; retain that failed evidence and rerun the
affected group on the host.

**Execution deferrals.** When the lead reserves CPU/GPU resources and defers
full validation, respect that limit rather than launching this build wrapper.
Bounded syntax, discovery and source-selected checks may document a partial
checkpoint, with thread limits, timeout, command, elapsed time and scope saved
in the evidence directory. Loading selected definitions with their real small
dependencies can check constructor/API behavior without initializing the
simulation engine; it does not prove full-module integration, process/queue
behavior, scientific equivalence or native ABI compatibility. Mark those gates
pending in BACKLOG. For [PLANS §24](PLANS.md#24-low-load-scan-and-upstream-maintenance-2026-09-20),
the affected command is `python3 test/validate.py --groups rust io fields --max-workers 1`;
the new argument/API items are registered in the `io` and `fields` manifests.

The follow-on [analytic audit, PLANS §25](PLANS.md#25-analytic-audit-of-propagation-frames-and-spectral-diagnostics-2026-09-20)
adds five unit tests directly to `amalthea/src/stepper.rs` and the `fields`
item `test/test_processing_math.jl`. The public standalone stepper must match
an independent DOPRI polynomial with nonzero composing linear evolution at
roundoff, retain accurate nonlinear rotation under refinement, and reject/retry
without changing the field. Diagnostics use a centered-impulse FFT and an
analytic chirp for phase, plus Gaussian Fourier pairs and independent half-width
roots for mode-summed bandwidth. Keep phase roundoff thresholds separate from
sampled FWHM interpolation error; refine sampling rather than widen a failing
bound. A one-job isolated mini-crate and source-selected Julia runs provide
bounded evidence only. Their failure/timeout and final subset results must be
retained; they do not replace the full recorded `rust io fields` gate.

The [Raman cancellation audit, PLANS §26](PLANS.md#26-raman-exponential-integrator-cancellation-audit-2026-09-20)
adds `amalthea/src/raman_math_tests.rs`, included by the normal Rust unit-test
runner. Its independent impulse-response integrals test all four forcing
weights at `2e-13` relative accuracy, including tiny coupling values; analytic
homogeneous transitions use `2e-14`, and scalar/available SIMD trajectories
for exactly interpolated affine drives use `2e-12`. Require finite results
before maximum-error reductions and a nonzero response. These are oscillator
subsystem checks, not optical propagation acceptance. Preserve the existing
Julia FFT-convolution/native single-step and full-solve gates, including their
Raman-on/off feature-effect assertions. The bounded CPU harness omits the CUDA
include, dispatcher and device method; it cannot validate GPU dispatch or ABI.
The combined maintenance gate is
`python3 test/validate.py --groups rust physics io fields --max-workers 1`.
Because the shared constructor supplies CUDA too, also run the affected strict
CUDA gate with `--cuda --groups rust --max-workers 1`, using the host-execution
and `AMALTHEA_REQUIRE_CUDA_TESTS=1` rules above. A CUDA pass requires its own
hardware run; CPU results do not establish it. Existing Python wheel validation
remains a separate work item.

For a static repository-hygiene review ([PLANS §27](PLANS.md#27-repository-hygiene-review-2026-09-20)),
check `git diff --check`, parse Python/TOML/JSON/YAML without importing the
scientific package, validate shell syntax with `bash -n`, and inspect test
discovery/timing coverage without starting workers. Use `git check-ignore
--no-index --stdin` with synthetic paths to verify ignore rules and template/
baseline exceptions. A read-only `rustfmt --check --edition 2024 --config
skip_children=true` pass can identify style debt; preserve its failures rather
than claiming an all-green hygiene gate or reformatting unrelated pending
changes. Local Markdown file-target checks do not validate anchors, external
links or the documentation build. Narrow credential-pattern checks cover
only the selected working files, not history or ignored files, and do not
constitute a security audit. None of these checks substitutes for numerical
acceptance or warrants running deferred scientific tests.

Every run creates a unique directory under `.rust_test_logs/validation/`
(or under the parent supplied with `--log-dir`). `summary.json` records the
revision, dirty status, selected groups, platform, effective relevant environment,
library SHA-256, commands, durations, exit codes, and final status. Version
logs, `inventory.json`, command logs, and `workers/` retain complete output.
Each worker JSON sidecar records its exact Julia command, assigned items,
thread overrides, duration, and exit code.
Build or preflight failure stops dependent tests; Cargo test failure still
allows Julia tests to collect evidence. A failed or interrupted run is not a
passing gate. Logs are ignored local artifacts: retain/share the evidence
bundle when handing work to another machine, and record concise measured
results in PORT_LOG. Exit codes do not establish numerical tolerances or
replace feature-sensitivity and actual-backend assertions.

To check the validation tooling itself without Julia or CUDA, run
`python3 -m unittest discover -s test -p 'test_*py'`. These tests also run in
the existing Python CI job.

Lower-level commands remain useful for focused diagnostics (they do not create
the wrapper's evidence bundle):

```bash
AMALTHEA_CUDA_BUILD=off cargo build --release --manifest-path amalthea/Cargo.toml
LUNA_TEST_GROUP=rust julia --project test/runtests.jl
python3 test/parallel_group_tests.py --group rust --max-workers 2
python3 test/run_full_gate.py
python3 test/parallel_group_tests.py --group rust --max-workers 10 --update-timings-only
julia --project -e 'using Pkg; Pkg.test("Amalthea")'
```

Main-gate `LUNA_TEST_GROUP` values: `physics`, `rust`, `sim-interface`,
`sim-multimode`, `sim-propagation`, `io`, `fields`, `examples`, `All`
(default). `test/test_groups.txt` is the canonical maintained-group list.
`test/run_full_gate.py` and GitHub Actions both use the same item-level LPT
scheduler; the serial `runtests.jl` command remains the simplest oracle when
checking aggregate assertion counts.

GitHub Actions adds the scheduler's `--ci` option, which preserves the former
`julia-actions/julia-runtest` bounds-check, deprecation-warning,
compiled-module, inlining, and user-code-coverage settings. Each worker writes
a distinct LCOV trace beside its log. Local timing and full-gate runs omit
`--ci` to retain their existing lower-overhead command; add it locally only
when reproducing the hosted Julia invocation:

```bash
python3 test/parallel_group_tests.py --group physics --max-workers 2 --ci
```

## 6. Definition of done for a native work item

### Internal Python foundation gate

The separate `python-native/` package tests grids, portable FFTs, and the
low-level callback solver described below. GNLSE and constant capillary envelopes
have Python and resident Rust evaluation; full capillary coverage remains
unfinished. Build/install it in an isolated
Python environment with `RUSTFLAGS="" maturin build --release` from that
directory, then install the resulting wheel. From the repository root export
independent setup fixtures:

```bash
julia --startup-file=no --project python-native/tools/export_grid_oracle.jl /tmp/amalthea-grid-oracle
AMALTHEA_GRID_ORACLE=/tmp/amalthea-grid-oracle python -m pytest python-native/tests -q -s
```

Use the environment's Python executable for pytest. No environment variable
means the Julia-fixture test skips; an acceptance run must supply the exported
fixtures. It asserts all axis/window max-relative errors below `1e-13`, exact
selection masks, and matching shapes. Portable FFT checks cover odd/even
lengths, direction/normalization, DC/Nyquist, input ownership, repeated use,
Hilbert phase, and causal convolution. The documented final propagation gates
still apply when the driver/physics are implemented.

Also build an sdist (`maturin sdist`), extract outside the checkout, rebuild
with `CARGO_NET_OFFLINE=true` after development dependencies are cached, and
run that extracted suite against its installed wheel. Record the actual wheel
tag: a locally produced `manylinux_2_34` wheel is not evidence for the planned
`manylinux_2_28` release baseline. These are internal artifact checks.

### Internal GNLSE and hosted artifact gate

For the Python GNLSE evaluator, additionally export
`julia --startup-file=no --project python-native/tools/export_gnlse_oracle.jl /tmp/amalthea-gnlse-oracle`
and set `AMALTHEA_GNLSE_ORACLE=/tmp/amalthea-gnlse-oracle` alongside the grid and
solver oracle variables when running the installed suite. This compares twelve
independently prepared configurations at setup/RHS/single-interval/fixed-solve
`1e-13` and adaptive-solve `1e-6`, including nonzero beta0/beta1, actual high-level
entrypoint parity, and independently measured Kerr/Raman/shock/loss sensitivity.
SiO2 response/RHS/fixed-step comparisons use an independently refined Julia
normalization, because the default Julia quadrature contributes a measured
8.58e-12 relative normalization error. Export the unmodified response and its
normalization too; verify the discrepancy follows the scalar normalization and
retain default high-level Julia comparisons at the 1e-6 full-solve tier. The
closed-form Python normalization must agree at 1e-13 with both refined Julia
and independent time-domain quadrature with refined tolerance/breakpoints and
an explicitly enforced error estimate. Do not loosen the tight gate to absorb
the unrefined oracle error. See the SiO2 design in PYTHON_NATIVE_PLAN.md.
The standalone Linux CI job runs the installed suite and the source-rebuilt
suite outside the checkout with all nineteen oracle variables supplied. It also
runs twelve propagation examples and the carrier-pulse/PPT/molecular setup examples with
networking disabled and Julia/Cargo absent from PATH.
A prepared workflow is not hosted execution evidence; record its exact commit
and run before claiming hosted acceptance. These remain internal local-platform
checks, not manylinux release validation or complete optical coverage.

For rich pulse inputs, export
`julia --startup-file=no --project python-native/tools/export_pulse_oracle.jl /tmp/amalthea-pulse-oracle`
and supply `AMALTHEA_PULSE_ORACLE` alongside the other three oracle variables.
The installed suite compares seven independent pulse preparations and full
trajectories, including input propagators and nonuniform spectral data, at
1e-13 setup/interval and 1e-6 full-solve tiers. Retain omitted-pulse/propagator
sensitivity and independent interpolation-refinement checks. The standalone
CI job supplies this fourth oracle directory and runs `examples/pulses.py`
from the source-rebuilt wheel with networking disabled.

### Resident portable GNLSE gate

For carrier-resolved input preparation, export
`python-native/tools/export_real_pulse_oracle.jl` with the root Julia project
and supply `AMALTHEA_REAL_PULSE_ORACLE`. Compare eight independently prepared
RealGrid pulse cases (Gaussian/sech, CEP/GDD, peak power, mixed/data/custom
inputs) for axes, spectra, real fields, analytic intensity and both energy
conventions at 1e-13. Require the instantaneous-square intensity alternative
to differ by more than the asserted tolerance. Retain analytic odd/even
Hilbert tests including DC/Nyquist and shape/ownership/exception checks.
Run `examples/real_pulse.py` from the source-rebuilt installed wheel offline.
This proves input preparation; carrier-resolved trajectories have a separate
acceptance gate.

The optical and pulse oracle tests run both explicit Python and native paths.
`test_resident.py` adds fourth/fifth-order fixed and adaptive lifecycle checks,
nontrivial accepted windows, rejection/retry/restart/stopping, analytic dense
convergence, private configuration validation, and a monkeypatch guard proving
that native/auto never call Python RHS/filter methods. Cargo's portable tests
cover direct DFT/input preservation/reuse, a resolved RealGrid Kerr third
harmonic, analytic Hilbert intensity, and serial/four-worker modal fixed-node
identity for complex and real polarized fields (real includes Hilbert Raman).
Run `python3 test/validate.py --all` for the shared CPU seam, retaining FFI
checks. CPU-only changes here do not change CUDA implementations or ABI.

### Gas material setup gate

Export `julia --startup-file=no --project python-native/tools/export_material_oracle.jl
/tmp/amalthea-material-oracle` and supply `AMALTHEA_MATERIAL_ORACLE` with the
other four fixture variables. All sixteen gases compare density, inverse
pressure, polarizability, refractive index, and available default gamma3 at
1e-13 relative tolerance. Require Python CoolProp 8.0.0 and independently
require the Julia fixtures to record CoolProp 7.2.0 and CODATA2014 constants.
Keep both version checks and the numerical comparisons. Include zero
pressure, 273.15/293.15/330 K, 0.1–50 bar, 200–3000 nm, scalar/array broadcasting,
alias distinctions, sensitivity and thermodynamic round trips. This proves
setup coverage, not capillary trajectories. The installed offline gate also
calls the gas thermodynamic interface without network access.

### Native work-item acceptance

For constant capillary envelopes, export `tools/export_capillary_oracle.jl`
from `python-native/` with the root Julia project and supply
`AMALTHEA_CAPILLARY_ORACLE`. Seven independently prepared cases cover
full/reduced, loss/Kerr controls, HE12, finer temporal sampling and molecular
Kerr-only propagation. Input/beta/density/Aeff/energy/RHS and identical-input
single-interval gates stay at 1e-13; independently prepared complete fixed,
adaptive and high-level trajectories stay at 1e-6. The interval oracle must
use the same accepted-step window as both Python paths: floating-point range
endpoints can sample the filtered left state of the next interval. Assert
Kerr/loss and changed-mode/gas effects exceed the full-solve tolerance.
Both explicit backends and the source-rebuilt offline capillary example are
required. These checks cover the internal constant-envelope slice, not
plasma, molecular Raman, variable profiles or modal propagation.

For constant carrier capillaries, export `tools/export_real_capillary_oracle.jl`
and supply `AMALTHEA_REAL_CAPILLARY_ORACLE`. Eight independent cases cover
full/reduced, loss/Kerr, HE12, fine grids, THG-off and fourth-order controls.
Use native/Python paths where eligible and Python for THG-off. Preserve 1e-13
setup/RHS/identical-input dense intervals and 1e-6 independently prepared full
trajectories; Kerr, THG and loss effects must exceed the trajectory tolerance.
The real resident lifecycle suite covers windows, rejection/retry/restart,
stopping, configuration ownership and analytic dense convergence for the
real scalar equation dE/dz=a*E+E^3. Run `examples/carrier_capillary.py` from the
source-rebuilt installed wheel offline. This Kerr/loss gate uses plasma=False;
default carrier plasma is covered by the separate plasma gate below.

The mode setup exporter is `python-native/tools/export_mode_oracle.jl`; supply
its directory as `AMALTHEA_MODE_ORACLE`. The installed mode tests cover forty
HE/TE/TM, full/reduced, loss/profile combinations, refined effective area,
independent spatial normalization, silica literal data and interpolation,
extra gas/vacuum cores and cutoff, callback shape/ownership/exceptions, and
group velocity at 1e-13. All seven derivative orders compare identical Julia
samples at 1e-13, including adaptive bound/step calculations. This does not
establish independent higher-derivative accuracy: retain the raw independently
evaluated derivative discrepancies and diagnose cancellation before closing
that gate. The analytic refinement test uses exp(2x), whose derivative signal
depends on the requested order, 100-digit arithmetic and ten spacing halvings.
It requires the refined derivative error below 1e-13, convergence at each
halving, and the float64 result within measured input perturbation plus a
gamma_n summation bound. The bound must remain below one thousandth of the
derivative signal, excluding vacuous zero/wrong-order results.
This verifies the inherited method's accuracy limit without weakening the
same-sample gate. Complete capillary trajectory acceptance remains separate.

For the standalone callback driver, additionally export and supply its solver
oracle directory:

```bash
julia --startup-file=no --project python-native/tools/export_solver_oracle.jl /tmp/amalthea-solver-oracle
AMALTHEA_GRID_ORACLE=/tmp/amalthea-grid-oracle AMALTHEA_SOLVER_ORACLE=/tmp/amalthea-solver-oracle python -m pytest python-native/tests -q -s
```

These development-only exporters use the Julia fallback, not the new Python
code. Driver acceptance covers first-interval dense equivalence (`1e-13`),
fixed/adaptive trajectories (respectively `1e-13`/`1e-6`), independent nonlinear
analytic accuracy, order-4/order-5 dense convergence, rejected trials, restart,
nonuniform nonlinear filters, exception identity, and field-array ownership.
Exact saved-position and filtered-output checks guard against NumPy/Julia
range rounding placing nominal boundary samples on opposite sides of a filter.
An exported-oracle skip is not a passing acceptance gate. Rebuild/install the
sdist outside the checkout and run its nested `python-native/tests` suite and
`python-native/examples/solver_analytic.py` against the installed wheel.

For the complete hosted reference artifact, prepare the root Julia project,
then run `python python-native/tools/export_oracles.py --output NEW_DIRECTORY`.
The output directory must not exist. This exports all twenty fixture families
with Rust offloads disabled; a completion manifest records the revision,
normalized-LF source hashes, output hashes, Julia version and CoolProp 7.2.0.
Keep the directory and its exporter logs on failure. An incomplete export must
not be reused as a complete reference artifact.

Before installed-wheel tests, run
`python python-native/tools/check_validation.py oracles DIRECTORY` in the
matching checkout. The command verifies provenance and prints all required
environment paths. In GitHub Actions, add `--github-env "$GITHUB_ENV"` to set
those paths for subsequent steps only after verification succeeds. Produce
JUnit XML using pytest's `--junitxml=REPORT.xml`, then require
`python python-native/tools/check_validation.py pytest REPORT.xml`. Empty
collection, failures, errors and skips are all incomplete acceptance. Upload
the XML and complete test logs even when pytest fails. The transport/acceptance
regressions run with `python test/test_python_native_validation.py -v`; their
synthetic fixtures are not scientific validation.

The maintained platform runner builds CPU-only checkout and source-rebuilt
wheels with `python python-native/tools/wheel_validation.py build --output
NEW_DIRECTORY`. The directory must be outside the checkout. Install maturin in
the build interpreter's environment; Linux additionally needs ziglang and
auditwheel, Windows needs delvewheel, and macOS uses Xcode's otool. The runner
requires a native supported host and CPython 3.11–3.14, records exact commands,
tool/interpreter versions and hashes, verifies the source archive and complete
wheel package inventory, and enforces the documented platform tag. Bytecode
caches are excluded from both distributions. A `built` manifest is only build
evidence.

Run `python python-native/tools/wheel_validation.py test --manifest
DIRECTORY/build.json --oracles ORACLE_DIRECTORY` with the same checkout and
build interpreter. It verifies both manifests, then creates fresh environments,
installs binary-only dependencies and the HDF5 extra, checks installed module
paths/libraries, runs all complete examples with OS-enforced network denial,
and runs checkout solver/output smoke plus the full source-rebuilt suite.
Isolated interpreters use explicit UTF-8 mode. JUnit skips/errors/failures fail
acceptance. `validation.json`, per-command logs, XML and offline-example JSON
are retained separately from build evidence; failed examples retain their
partial stdout and original exception. Do not edit source or reuse these
environments while an acceptance run is active.

The hosted platform jobs pass `--ephemeral-ci`: Linux uses the runner's sudo
network namespace, and Windows creates temporary interpreter-specific firewall
rules with cleanup. Windows firewall setup refuses ordinary or self-hosted
execution. macOS requires its deny-network process sandbox. Unavailable network
isolation is an incomplete offline gate. Actual ARM64/macOS/Windows execution
and the oldest-glibc runtime gate remain required; local matrix parsing or
cross-compilation alone cannot establish them.

A native work item is complete when **all** hold:

1. The native path is selected by its toggle and runs the full geometry with no
   Julia callback in the hot loop (verify: no `@cfunction` round-trip for that path).
2. A single-step equivalence test passes at the tier in §4.
3. A full-`solve` equivalence test passes at the ~1e-6 floor tier.
4. The pre-existing Julia-path tests still pass (no regression).
5. A `PORT_LOG.md` entry records both achieved tolerances, the FFI symbols added,
   and any gotchas.
6. The test is non-vacuous: the feature changes the oracle by more than the
   asserted equivalence tolerance, or another direct assertion proves the
   relevant intermediate quantity is present.


### Standalone ADK setup gate

Export `python-native/tools/export_adk_oracle.jl` with the root Julia project
and supply `AMALTHEA_ADK_ORACLE`. Test sixteen supported material identifiers
(including atomic H), both occupancies, threshold modes and cycle averaging.
Compare first potentials in three units, CODATA constants, coefficients and
resolved rates at 1e-13; thresholds must match exactly. For samples whose
exponential factor is subnormal, use its floating-point spacing amplified by
the positive prefactor as an absolute error bound; do not claim relative
precision below underflow. Record Julia's NaN at exactly zero with threshold
False separately: Python returns the physical zero limit. Independently
compare the n*=1 hydrogenic formula with 100-digit Decimal arithmetic. Require
nonzero rates, signed symmetry, occupancy/cycle-average sensitivity, input
ownership, repeated setup and invalid-input handling. Exercise the installed
rate evaluator with networking disabled. These gates prove setup only;
plasma trajectory acceptance remains separate.


### Standalone PPT setup gate

Export `python-native/tools/export_ppt_oracle.jl` and supply
`AMALTHEA_PPT_ORACLE`. Compare all supported material setups, numeric l=2,
Stark/dipole, msum, cycle average, sum/integral, occupancy and Cnl branches at
1e-13. Sensitivity uses a separately exported Julia baseline at the same
wavelength. Direct Julia PPT at zero does not terminate; the exporter marks
it unevaluated with a NaN sentinel and explicit metadata. Test Python's zero
limit separately. Numeric Julia setup requires floating-point zero correction
arguments due to its constructor type constraint.

Compare phi on both sides of x=26 against refined BigFloat Julia quadrature
and independent 100-digit scaled-integral quadrature. Compare raw independently
generated table nodes/rates and identical-sample spline queries at 1e-13,
including nonuniform inputs and removed zero-rate nodes. The spline must use
Maths.CSpline's normalized-knot derivative system, not FITPACK's not-a-knot
boundary condition. Retain default-size 65536-node construction and independent
interpolation refinement. Test cache hits, parameter isolation, corrupt bytes,
valid-archive corruption, atomic concurrent writes, callback serial order,
original exceptions, invalid outputs and max-terms failure. Run the installed
`examples/ppt.py` with networking disabled. Setup does not establish plasma
trajectory equivalence.

### Constant carrier plasma gate

Export `python-native/tools/export_plasma_capillary_oracle.jl` and supply
`AMALTHEA_PLASMA_ORACLE` with the other eleven fixtures. Eight cases cover
ADK/PPT, plasma/Kerr/THG controls, preionisation, a finer grid and fourth-order
stepping. Keep the 1e-13 input/RHS/identical-input dense gates and 1e-6 complete
fixed/adaptive/high-level gates. Record plasma, Kerr, THG, preionisation and
ADK/PPT effects above 1e-5. Include rejected trials against a Julia solve with
identical controls (accepted-step filtering makes cadence part of the problem),
default/material selection, existing rate models, output ownership/NPZ and
explicit unsupported configurations.

For plasma intermediates, record independently prepared rate, fraction,
current and polarization discrepancies. Compare current and polarization
at 1e-13 with identical Julia intermediate inputs. Bound cancellation-amplified
polarization discrepancies by cumulative absolute current perturbations plus
the floating-point summation bound, which must remain below 1e-6 of the signal.
Compare the final integral independently with 100-digit Decimal arithmetic.
This conditioning check does not replace the full RHS/dense/trajectory gates.
Run `examples/plasma_capillary.py` from the actual source-rebuilt installed
wheel offline, alongside the seven previous examples. For resident plasma,
extend all eligible cases to both paths and test actual Python callback
avoidance. Cover nonuniform supplied tables, ADK occupancy/cycle-average,
private configuration validation, box ownership across facade moves, repeated
construction, and explicit direct/custom/threshold-free fallback selection.
Run `python3 test/validate.py --all` for the safe resident CPU integration and
FFI gate. No shared ionisation formula or CUDA implementation changes here.

### Molecular Raman setup gate

Export `python-native/tools/export_molecular_raman_oracle.jl` and supply
`AMALTHEA_MOLECULAR_ORACLE` with the other twelve fixtures. Compare all six
complete molecular parameter sets, rotation/vibration toggles, temperatures,
rotational truncation/ranges, density-dependent lifetimes and Planck-windowed
samples at 1e-13. Preserve separate rotational grouping and absolute couplings.
Compare an analytic N2O rotor's populations/frequencies/couplings with 100-digit
arithmetic. Require component/temperature and applicable density effects above
1e-5. Explicitly retain Julia failures for missing O2 lifetimes, invalid
truncation and nonfinite zero-density vibrational origins. Test array ownership,
fresh density evaluations, empty components and invalid axes/options. Run
`examples/molecular_raman.py` from the source-rebuilt installed wheel offline.
This proves setup coverage; capillary Raman trajectories have a separate gate.

### Constant molecular capillary Raman gate

Export `python-native/tools/export_raman_capillary_oracle.jl` and supply
`AMALTHEA_RAMAN_CAPILLARY_ORACLE` with the other thirteen fixtures. Force Julia
FFT-convolution evaluation in the exporter. Compare all six complete gas models
on both grids at 1e-13 for initial fields, full RHS, density-scaled impulse
samples and identical-input dense intervals; use 1e-6 for independent complete
fixed/adaptive/high-level trajectories. Cover rotation/vibration, temperature,
Kerr-off Raman, fourth-order stepping, finer time sampling, carrier THG-off,
envelope THG (including its finer oversampling) and ADK/PPT+Raman combinations.
Require the oracle Raman, component, temperature, Kerr, THG and plasma effects
above 1e-5. Native-eligible cases must run on both native/Python paths and prove
native callback avoidance. Retain explicit fallback and missing-data/undefined
oracle errors; test empty components and owned NPZ output. Check the portable
carrier convolution against direct causal summation across repeated calls to
expose stale padded-tail data. Run `python3 test/validate.py --all` for the
shared CPU change and `examples/raman_capillary.py` from the source-rebuilt
installed wheel with networking disabled.

### Variable linear-operator callback gate

Export `python-native/tools/export_variable_solver_oracle.jl` and supply
`AMALTHEA_VARIABLE_SOLVER_ORACLE` with the other fourteen fixtures. Compare
matrix-field fixed trajectories and identical-input dense intervals at 1e-13,
and complete adaptive solves at 1e-6, for both orders with nonzero start position.
Fixed-step callback positions and accepted endpoints must agree exactly with
Julia, including dense extra-stage revisits and direct left-endpoint returns.
Retain non-vacuous variable/nonlinear/filter controls above 1e-5, rejected
trials, and the matching fifth-order tightly filtered adaptive repetition
failure. Test constant-callable equality, array ownership, serial callbacks,
teardown, invalid/nonfinite arrays and original exceptions at initialization,
stage and dense positions. Independently verify the endpoint exponential's
discrete formula and first-order error under refinement; do not claim fifth-
order accuracy for variable-L physics from the nonlinear tableau alone.
Run `examples/variable_solver.py` from the source-rebuilt installed wheel
offline. This changes the Python extension only; capillary profile/density
wiring is a separate next gate.

### Exact scalar capillary profiles gate

Export `python-native/tools/export_profile_capillary_oracle.jl` and its
companion `export_profile_limits_oracle.jl` into one directory; supply it as
`AMALTHEA_PROFILE_CAPILLARY_ORACLE` with the other fifteen fixtures. Compare
rising/falling/multipoint pressure, arbitrary pressure/radius functions and
simultaneous profiles on both grids, with all six complete molecular gas
models. Include plasma, THG, component/temperature, full/reduced, finer-grid
and fourth-order controls. At off-grid positions compare density, area, neff,
beta0/beta1, complete RHS and molecular impulses at 1e-13. Isolate assembled
linear-phase cancellation using identical Julia neff/beta inputs and exact
linear operators at the requested dense-stage positions (no interpolation);
retain same-input dense intervals at 1e-13 and independently prepared complete
trajectories/high-level entrypoints at 1e-6. Show profile/response effects above
1e-5 and constant-callable equivalence at the full-solve tier.

Compare pressure/density spline nodes and evaluation against Julia, including
same-input normalized derivatives, zero endpoints and equal nonzero pressures;
refine the thermodynamic sampling independently. Exercise saturated loss
clamping with a nonlinear-free solve: the variable and constant paths retain
Julia's distinct amplitude attenuation bounds, 3000 and 1500 per metre.
Check supported zero-density Raman and explicit undefined H2/all-zero-gradient
failures. Verify actual stage positions, serial callbacks, shape/finiteness,
original exceptions, copied arrays/NPZ metadata and repeated teardown. Run all
PPT regressions after extracting the shared cubic helper. Require the full
installed and source-rebuilt suite, plus `examples/profile_capillary.py` offline.


### Scalar gas and response mixture gate

Export `python-native/tools/export_mixture_capillary_oracle.jl` and supply
`AMALTHEA_MIXTURE_CAPILLARY_ORACLE` with the other sixteen fixtures. The oracle
uses Julia's low-level susceptibility sum and per-species response tuples.
Require independent density, neff, area, beta0/beta1, complete RHS and Raman
impulse comparisons at 1e-13, with identical linear inputs for dense intervals
and variable phase assembly. Compare full fixed/adaptive trajectories at 1e-6.
Cover both grids, distinct/duplicate/single-species mixtures, full/reduced loss,
THG, fourth-order and finer-grid controls, and Raman components/temperature,
independent ADK/PPT plasma and preionisation with simultaneous profiles/tapers.
Require species, response and profile effects above 1e-5 and density-equivalent
split-gas agreement at the full-solve tier. All eligible constant Kerr cases
must also use native evaluation and prove callback avoidance.

Check per-species default selection/overrides, fresh Raman broadening, exact
callback positions, serial execution, exceptions/teardown, malformed inputs
and owned NPZ metadata. Keep quantum-noise, envelope-plasma and undefined
molecular responses explicit failures. Run all prior scalar and GNLSE tests
after factoring scalar responses. Require the full installed/source-rebuilt
suite and `examples/mixture_capillary.py` offline. Python-only evaluation and
setup reuse the completed shared CPU gate; shared engine changes require a
new recorded `python3 test/validate.py --all` gate.

For the combined PPT/ADK + Raman + multipoint-gradient/taper stress case,
retain the coarse 1e-5 m adaptive diagnostic and export paired 2.5e-6/1e-6 m
refinements plus default controls. Record step counts and compare the 1e-6 m
case at the unchanged 1e-6 field gate. Same-input dense and original fixed-step
checks remain at their tighter tiers. Do not claim the coarse adaptive run
meets this gate; variable-L endpoint propagation is step-position dependent,
and nonlinear rtol alone does not establish full-equation accuracy.

### Modal geometry and global quadrature gate

Export `python-native/tools/export_spatial_oracle.jl` and supply
`AMALTHEA_SPATIAL_ORACLE` with the other seventeen fixtures. Compare independent
normalization, physical-field synthesis and fixed-node complete-array projection
at 1e-13 across HE/TE/TM, polarization pairs, selected x/y components, full polar
and Cartesian domains and reduced integrals, including varying radius and
generic custom normalization. Compare independently refined Julia h/p integrals
and SciPy integrals, recording the global error, actual refinement and node
counts. Bounded subdivision passes check global convergence before requiring
SciPy's stricter component convergence; all restarted nodes count toward the budget. Test independent analytic complex-array integrals with both Gk21 and
Genz-Malik rules; require the original global L2 criterion even when mocked
component tolerances individually pass. Bound callback evaluations and reject
nonconvergence. Verify analytic power/orthogonality, nonzero higher-mode transfer
and polarization effects, exact boundary rules and common-domain validation.

Check original exceptions, finite/shape constraints, normalization validity,
fresh position evaluation, serial callbacks, owned retained arrays and teardown.
Run the full installed/source-rebuilt suite and custom-mode setup example
offline. This validates a dependency of modal propagation; it does not replace
the later same-input temporal RHS/dense and end-to-end trajectory gates.


### Modal propagation gate

Export `python-native/tools/export_modal_capillary_oracle.jl` and supply
`AMALTHEA_MODAL_CAPILLARY_ORACLE` with the other eighteen fixtures. The exporter
checks Julia's global L2 quadrature criterion at every RHS call. Cover both
grids, HE/TE/TM mode averages/collections, polarization pairs, full/reduced
integrals, custom modes, Raman, ADK/PPT/preionisation, species mixtures,
gradients/tapers, fourth-order and default solver controls. Compare initial
fields, node synthesis, response/FFT/projection and transferred-linear,
fixed-node dense intervals at 1e-13; full trajectories with the public 1e-3 spatial default remain at 1e-6
against the independently refined Julia reference.
Independent assembled linear arrays retain the finite-difference frame check;
fixed-node dense checks transfer the exact linear inputs.

For cancellation-prone plasma integrals retain the scalar gate: rates, same-fraction current and same-current polarization at 1e-13, with a
forward bound on propagated input/rounding error that remains below 1e-6.
Near-zero fractions use the derived integrated-rate/exponential subtraction
bound, and currents use the phase-integral/loss bound. Refine a vector integral
independently with 100-digit arithmetic. Compare FFT
and projection from identical polarization arrays at 1e-13. Do not replace
these checks with a relaxed relative tolerance on the cancelled integral.
Refine Gk21 quadrature with Gk15 and tighter budgets; compare default and
refined complete trajectories with the independent Julia oracle. Require
physics controls to change Julia by more than 1e-5.

Check generic and analytic custom normalization, original exceptions, serial
callbacks, actual positions, teardown, mode assignment, complete modal input
arrays and owned outputs. Run installed/source-rebuilt suites and the complete
`custom_mode.py`/`modal_capillary.py` examples offline. Initial-step bounds are
covered by `export_solver_oracle.jl`: initial dt may lie outside min/max bounds;
subsequent fixed bounds make the schedule comparison independent of adaptive
error-estimate rounding. The shared CPU/FFI gate remains valid when only the
Python package/extension changes; shared engine or Julia edits require rerun.

### General nonlinear-response callback gate

Export `python-native/tools/export_callback_oracle.jl` and set
`AMALTHEA_CALLBACK_ORACLE` with the other nineteen fixtures. This adds complete
Cartesian custom-mode propagation on both grids with independently normalized
orthogonal fields, custom dispersion, exact partial-pressure/radius profiles,
and Julia Kerr responses. Assert initial fields, normalization, assembled RHS
and transferred-linear fixed-node dense intervals at 1e-13, independent linear
assembly at the documented finite-difference tier, and complete fixed/adaptive
trajectories at 1e-6. Linear-only and constant-profile oracle controls must
change the output by more than 1e-5.

`test_callbacks.py` additionally replaces GNLSE Kerr/Raman with a complete-array
convolution, and capillary scalar/vector Kerr and plasma with custom functions.
Reuse independent GNLSE/modal oracles, preserving 1e-13 same-input/dense and
1e-6 trajectory gates. Cover response mixtures, density updates, THG, ownership
of every field/context/returned array, serial ordering, actual z/spatial points,
original exceptions, malformed outputs, forced-native rejection before calls,
nested simulations and repeated construction/destruction. Exception traceback
ownership is released before checking teardown; otherwise the test itself
retains the callback frame. Run the complete installed/source-rebuilt suites
and `examples/custom_response.py` offline. This unit changes Python evaluation
only; the prior full shared CPU/FFI gate remains applicable.

### Safe native modal point gate

`test_native_points.py` compares scalar/vector portable point arrays with the
independent modal nodes, repeated/zero/changing input batches, retained arrays,
complete RHS and transferred-linear fixed-node dense intervals at 1e-13.
Compare complete Kerr/Raman/mixture modal trajectories against Julia at 1e-6.
Check constructor and batch dimension/overflow/finiteness guards, teardown and
recovery after invalid results. Prove auto dispatch bypasses Python temporal
responses and excluded/custom/profile paths retain Python evaluation. Metadata
must distinguish native points from the surrounding SciPy modal evaluator.
Run the full shared CPU/FFI gate after extracting the temporal kernel, plus all
installed/source-rebuilt Python tests and offline examples. Record complete-solve
performance, counts and field agreement under matched inputs/controls before
promoting automatic acceleration; the frozen audit remains untouched.

### Optional HDF5/result output gate

Install `amalthea-native[hdf5]` and run `test_output.py`. Use independent NumPy
and h5py readers to require exact numeric arrays/dtypes, original modal axes,
UTF-8/nested JSON metadata, and grid-appropriate inverse FFT reconstruction.
Cover real simulation results from GNLSE, scalar carrier and custom-response
modal paths, repeated overwrite/close behavior, and serialization errors that
leave existing files intact. In a fresh process forbid h5py imports and verify
base package import, simulation and NPZ still work; HDF5 must report the missing
extra. Test actual checkout/source-rebuilt wheels and packaging extra metadata,
then run the complete `output_processing.py` example offline. Physics formulas
and stepping are unchanged, so exact round trips establish this I/O gate;
the preceding CPU and numerical acceptance remain applicable.

### Linux minimum-glibc runtime gate

`test/standalone_wheels/glibc228.py` tests Linux x86_64 wheels in a pinned,
read-only Debian glibc 2.28 userspace using bubblewrap. Run with normal host
namespace access; no Docker daemon or system installation is required. Prepare
the temporary root filesystem once, then use a compatible development-only
CPython installation, a completed maintained build manifest and a binary
wheelhouse for the same interpreter:

```bash
python3 test/standalone_wheels/glibc228.py prepare --output NEW_ROOT_DIRECTORY
python3 test/standalone_wheels/glibc228.py probe --rootfs ROOT_DIRECTORY/rootfs --interpreter PYTHON_HOME --output NEW_PROBE_DIRECTORY
python3 test/standalone_wheels/glibc228.py test --rootfs ROOT_DIRECTORY/rootfs --interpreter PYTHON_HOME --manifest BUILD_DIRECTORY/build.json --oracles ORACLE_DIRECTORY --wheelhouse WHEELHOUSE --output NEW_TEST_DIRECTORY
```

The test verifies vendor/archive, source/build/wheel/reference provenance and
actual loaded libc 2.28. Only explicit artifact, test, example and dependency
directories are mounted. Install both wheels into fresh environments with
`--no-index --only-binary=:all:`, run all seventeen examples with no network or
toolchain PATH, then require the checkout smoke and complete source-wheel suite
to pass without skips. Repeat on CPython 3.11–3.14. `smoke` accepts the same
arguments without `--oracles` and records only `smoke_passed`; it does not
establish numerical acceptance. Preserve failed setup/run directories and
never substitute host execution after a namespace failure. The host kernel and
CPU remain current; this gate establishes the glibc userspace boundary only.
Helper regressions: `python3 test/test_glibc228_validation.py -v`.

Before release, verify the four completed helper outputs against the candidate
checkout and the downloaded release's actual Linux x86_64 source-wheel bytes:

```sh
python3 test/release.py verify-glibc --repository MATCHING_CHECKOUT \
  --gates GATE_ROOT --builds BUILD_ROOT --oracles ORACLE_DIRECTORY \
  --assets RELEASE_ASSETS --output NEW_GLIBC_REPORT.json
```

`GATE_ROOT` contains `3.11/` through `3.14/`, retaining `validation.json`,
`probe.json`, both raw JUnit/example results and every helper `.log`/`.log.json`.
`BUILD_ROOT` contains `linux-x86_64-3.11/` through `linux-x86_64-3.14/`, each with
the exact runtime-tested `build.json`, `sdist/` and `wheels/{checkout,source}/`.
Preserve each manifest's bytes when moving this evidence: recorded producer
paths are resolved by basename under the supplied directories. Independently
exported complete oracles are allowed when their source/revision and manifest
hash match the runtime gate. Original hosted export hashes need not match a
fresh export at the same candidate.

This offline command rejects smoke-only, failed or incomplete runs; wrong
versions, sources, reference/wheel hashes or glibc probes; skipped/erroring
JUnit; partial numerical commands; and missing example or command evidence.
It applies the collector's shared numerical count floors and all seventeen
examples to both wheel kinds. The new JSON report retains inspected file hashes
and per-cell errors, with a nonzero exit for incomplete evidence. Existing
reports and producer records are preserved. This verifies only the four
minimum-glibc cells; the other platform, CI and publication checks remain
required. Synthetic regressions in `python3 test/test_release.py -v` establish
tool behavior, never platform or release acceptance.

### Separate post-repair Python performance gate

Use `test/python_performance/run.py` with the final installed source wheel,
matching build/reference manifests and an output directory outside the checkout;
see its README for the complete invocation and metric boundaries. `--smoke`
performs real numerical checks but never establishes controlled performance.
Accepted runs refuse active heavy validation, use two warmups and 10–30
randomized samples, and require relative MAD <=3% and bootstrap 95% CI
half-width <=5% for the complete setup-plus-solve workload. Each sample's field
must pass 1e-6 against independently prepared Julia; feature controls must exceed
1e-5. Scalar same-input RHS and custom/built-in callback agreement remain at
1e-13. Modal global integration retains its separate fixed-node/refinement gate.
Record actual fallback and adaptive counts. Keep excluded old-Rust method
comparisons and temporal refinement evidence; excluded paths contribute no
accepted speedup. Do not alter the frozen CPU audit. Evidence tooling checks:
`INSTALLED_PYTHON -I test/test_python_performance.py -v`.

### Collect a downloaded hosted wheel matrix

Download the artifacts from one workflow run and inspect them against a checkout
at that run's exact commit. `MATCHING_CHECKOUT` must contain the tested source;
the collector may live in the working checkout. Retain the full run metadata:

```sh
gh run view RUN_ID --repo vdiego28/Amalthea.jl \
  --json databaseId,url,headSha,status,conclusion,jobs > run.json
gh run download RUN_ID --repo vdiego28/Amalthea.jl --dir downloaded-artifacts
python3 test/standalone_wheels/collect.py \
  --artifacts downloaded-artifacts --repository MATCHING_CHECKOUT \
  --run-json run.json --output wheel-evidence.json
```

The default requires sixteen platform/interpreter cells. For the Linux-only
subset, explicitly add `--platforms linux-x86_64 linux-arm64`; the report records
that smaller scope. JSON/Markdown retain missing/failed cells, workflow failures,
reference/source/wheel digests, actual JUnit counts and completed offline runs.
Exit zero means that requested wheel matrix passed. It does not close other
workflow failures, Apple diagnostics, controlled performance or publication.
Never substitute synthetic collector fixtures for actual platform results.
Collector regressions: `python3 test/test_wheel_collection.py -v`.
