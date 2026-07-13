# docs/dev/BACKLOG.md Phase G.3 — "CI benchmark job: track the Phase C benchmark
# over time so native-path regressions (like §3.2) show up as perf cliffs,
# not silence."
#
# Run standalone (not part of the @testitem suite — a wall-clock number
# needs its own dedicated, uncontended CI job, not a shared test-matrix
# runner): `julia --project test/benchmark_native_default.jl`. Writes
# `test/benchmark_result.json` in `benchmark-action/github-action-benchmark`'s
# `customSmallerIsBetter` format, published/tracked by the "Native-path
# benchmark" CI job (.github/workflows/run_tests.yml) so a regression shows
# up as a break in a graph and a CI alert comment, not silence.
#
# Two independent checks, deliberately separated:
#   1. Stepper *selection*: `prop_capillary` with no environment toggles set
#      (the fork's flagship out-of-the-box config) must pick
#      `RK45.RustNativeStepper` — this is the actual Phase C / REVIEW.md
#      §3.2 bug (silent fallback to the Julia stepper). Hard `@assert`:
#      this must never regress, on any hardware, so it is not left to the
#      noisy timing comparison below.
#   2. Stepper *speed*: native's own per-step wall time for a fixed-dt,
#      fixed-step-count run (no adaptive step-size confound — two
#      independently-adaptive full solves can take different step-count
#      paths and are not a stable basis for comparison; see PORT_LOG
#      2026-07-01 on the adaptive PI controller's FP-noise sensitivity).
#      This is tracked as an absolute number over time, not a ratio against
#      the Julia stepper: an empirical check on 2026-07-05 found the
#      native-vs-Julia wall-time ratio too hardware-dependent to hardcode a
#      pass/fail threshold on (measured *slower* than Julia's PreconStepper
#      on one dev machine for this workload, despite PORT_LOG's ~3.5x
#      full-`prop_capillary` figure) — trend-relative alerting on native's
#      own time avoids baking in an assumption that didn't reproduce.

using Amalthea
import Amalthea: Interface, set_fftw_mode
using Amalthea.RK45: PreconStepper, RustNativeStepper, step!
import Logging: with_logger, NullLogger
import Printf: @sprintf

set_fftw_mode(:estimate)

function main()
libpath = RK45._LIBAMALTHEA_RK45
if !isfile(libpath)
    @warn "Rust library not found — skipping benchmark, writing an empty result."
    write(joinpath(@__DIR__, "benchmark_result.json"), "[]")
else
    radius = 125e-6
    flength = 0.03
    gas = :He
    pressure = 1.0
    λ0 = 800e-9
    λlims = (200e-9, 4e-6)
    trange = 1e-12

    # ── Check 1: stepper selection (the actual Phase C bug) ─────────────────
    with_logger(NullLogger()) do
        prop_capillary(radius, flength, gas, pressure;
                        λ0, λlims, trange, energy=1e-6, τfwhm=30e-15, saveN=2)
    end
    @assert RK45._LAST_STEPPER_TYPE[] <: RK45.RustNativeStepper "Default " *
        "field-resolved prop_capillary (no env toggles) did not select " *
        "RustNativeStepper — this is the Phase C / REVIEW.md §3.2 " *
        "silent-fallback regression."
    println("Stepper selection OK: default workload uses RustNativeStepper.")

    # ── Check 2: native's own per-step wall time, fixed-dt (stable number) ──
    kw = (; λ0, λlims, trange, raman=false, plasma=true, kerr=true,
           shotnoise=false, energy=1e-6, τfwhm=30e-15)
    Eω, grid, linop, transform, FT, output = with_logger(NullLogger()) do
        Interface.prop_capillary_args(radius, flength, gas, pressure; kw...)
    end

    # A fresh stepper per trial (same initial field each time) keeps every
    # trial's cumulative propagation distance bounded to nsteps*dt << flength
    # — reusing one stepper across trials was found to accumulate enough
    # nonlinear phase over many repeated large fixed steps to drive the
    # field magnitude far outside the PPT ionization LUT's fitted range,
    # segfaulting inside PptIonizationRate::rate (a real Rust bug — an
    # out-of-bounds/NaN-index LUT access outside [e_min,e_max] instead of
    # the documented default clamp — worth a follow-up issue, but avoiding
    # it here since it's orthogonal to what this benchmark measures).
    dt = 0.0002
    nsteps = 50
    ntrials = 5

    new_stepper() = RustNativeStepper(transform, linop, copy(Eω), 0.0, dt,
                                       rtol=1e-6, atol=1e-10, max_dt=dt, min_dt=dt)
    step!(new_stepper())  # warm up (JIT; FFTW planning is excluded by set_fftw_plans at construction)

    best = Inf
    for _ in 1:ntrials
        s_ru = new_stepper()
        t = @elapsed for _ in 1:nsteps
            step!(s_ru)
        end
        best = min(best, t)
    end
    per_step_ms = best / nsteps * 1000

    println("Native mode-avg+plasma per-step time (best of $ntrials trials, ",
            "$nsteps steps each): ", per_step_ms, " ms")

    # customSmallerIsBetter format: a JSON array of {"name","unit","value"}.
    # Hand-rolled (no JSON dependency needed for one fixed-shape record).
    json = @sprintf(
        """[{"name": "native mode-avg+plasma per-step (fixed dt)", "unit": "ms/step", "value": %.6f}]""",
        per_step_ms)
    write(joinpath(@__DIR__, "benchmark_result.json"), json)
end
end

main()
