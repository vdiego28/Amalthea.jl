# Native-Rust Backend Port — Testing & Equivalence

> Status: design doc for the phased port. Phases 0-4 are implemented and
> passing (see `docs/native-port/PORT_LOG.md`); Phase 5 (Modal) is next.
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
    using Luna
    import Logging: with_logger, NullLogger
    import LinearAlgebra: norm

    # ── skip guard (copy verbatim) ──────────────────────────────────────────
    libname = if Sys.iswindows(); "luna_rust.dll"
              elseif Sys.isapple(); "libluna_rust.dylib"
              else; "libluna_rust.so"; end
    libpath = joinpath(@__DIR__, "..", "luna-rust", "target", "release", libname)
    if !isfile(libpath)
        @warn "Skipping: shared library not found at $libpath. " *
              "Build with `cargo build --release` in luna-rust/."
        return
    end

    # ── run BOTH paths and compare the final spectrum ───────────────────────
    @testset "<geometry> equivalence" begin
        args = (radius, L, gas, pres)
        kw = (; λ0, τfwhm=τ, energy, modes=:HE11, loss=false,
                saveN=2, trange=0.5e-12, λlims=(200e-9, 4e-6))

        out_julia = with_logger(NullLogger()) do
            prop_capillary(args...; kw...)
        end
        out_rust = withenv("LUNA_USE_RUST_NATIVE" => "1") do
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
- Toggle both paths via `withenv`, never by mutating global state.
- `prop_capillary` **requires** `λlims`; it does **not** accept `stepfun`,
  `rtol`, or `atol` as kwargs (learned the hard way — see PORT_LOG seed).
- Compare the final-z spectrum `Eω[:,end]`. For stronger checks also compare an
  intermediate save and a derived observable (e.g. spectral energy).
- Place the file in `test/` and add nothing else — `@run_package_tests`
  auto-discovers every `@testitem`.

## 2. Tolerance tiers (and the reason each applies)

Pick the **tightest** tier the math justifies. A test that passes at a looser
tier than its math allows is hiding a bug.

| Tier | Threshold | When it applies | Example (wired) |
|------|-----------|-----------------|-----------------|
| **bitwise** | `== 0.0` | identical IEEE-754 formula **and** identical Float64 inputs | Marcatili neff (`test_dispersion_rust.jl`) |
| **reassociation** | `~1e-13` | same formula, summation/BLAS order differs (FFTW parity, QDHT, dot products) | QDHT (`test_qdht_rust.jl`), Zeisberger (~1e-12) |
| **method/spline** | `~1e-8` | LUT/spline interpolation or a different-but-equivalent algorithm | PPT ionization (`test_ionisation_rust.jl`) |
| **FFT-method + floor** | `~1e-6` | FFT method differences **and** the run-to-run nondeterminism floor (§3) | RK45 stepper (`test_stepper_rust.jl`) |

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
| 5 | ⬜ next | modal + overlap cubature | `test/test_native_modal.jl` | ~1e-10 (cubature) | ~1e-6 |
| 6 | ⬜ | free-space 3-D FFT | `test/test_native_free.jl` | ~1e-13 | ~1e-6 |
| 7 | ⬜ | z-dependent linop assembly | extend the above | ~1e-13 | ~1e-6 |
| 8 | ⬜ | default-flip: existing suite green with native as default | full suite | — | existing |

Phase 5's single-step tier is looser (~1e-10) because adaptive cubature node
placement is itself algorithm-dependent; match Julia's cubature tolerance and
node rule to get as close as possible, and document the achieved number.

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

## 5. Commands

```bash
# Build the library first (required for any :rust test to run, else it skips)
cd luna-rust && cargo build --release

# Run only the Rust/native equivalence group
LUNA_TEST_GROUP=rust julia --project test/runtests.jl

# Rust unit tests
cd luna-rust && cargo test

# Full Julia suite (Phase 8 gate)
julia --project -e 'using Pkg; Pkg.test("Luna")'
```

Valid `LUNA_TEST_GROUP` values: `physics`, `rust`, `sim-interface`,
`sim-multimode`, `sim-propagation`, `io`, `fields`, `All` (default).

## 6. Definition of done for a phase

A phase is complete when **all** hold:
1. The native path is selected by its toggle and runs the full geometry with no
   Julia callback in the hot loop (verify: no `@cfunction` round-trip for that path).
2. A single-step equivalence test passes at the tier in §4.
3. A full-`solve` equivalence test passes at the ~1e-6 floor tier.
4. The pre-existing Julia-path tests still pass (no regression).
5. A `PORT_LOG.md` entry records both achieved tolerances, the FFI symbols added,
   and any gotchas.
