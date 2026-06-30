# Native-Rust Backend Port — Testing & Equivalence

> Status: design doc for the phased port. No phase is implemented yet.
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

## 4. Per-phase acceptance criteria

| Phase | What to test | Test file | Single-step tier | Full-run tier |
|-------|--------------|-----------|------------------|---------------|
| 0 | set/get round-trip bit-exact; no-op RHS reproduces Julia stepper | `test/test_native_phase0.jl` | bitwise (round-trip) | ~1e-6 |
| 1 | mode-avg + Kerr `prop_capillary(:HE11)` | `test/test_native_modeavg.jl` | ~1e-13 | ~1e-6 |
| 2 | plasma + EnvGrid Kerr (`gnlse`/ionising capillary) | `test/test_native_plasma.jl` | ~1e-13 | ~1e-6 |
| 3 | radial + resident QDHT | `test/test_native_radial.jl` | ~1e-13 | ~1e-6 |
| 4 | Raman (carrier SDO) | `test/test_native_raman.jl` | ~1e-13 | ~1e-6 |
| 5 | modal + overlap cubature | `test/test_native_modal.jl` | ~1e-10 (cubature) | ~1e-6 |
| 6 | free-space 3-D FFT | `test/test_native_free.jl` | ~1e-13 | ~1e-6 |
| 7 | z-dependent linop assembly | extend the above | ~1e-13 | ~1e-6 |
| 8 | default-flip: existing suite green with native as default | full suite | — | existing |

Phase 5's single-step tier is looser (~1e-10) because adaptive cubature node
placement is itself algorithm-dependent; match Julia's cubature tolerance and
node rule to get as close as possible, and document the achieved number.

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
