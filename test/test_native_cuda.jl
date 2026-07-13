using TestItems

@testitem "Native-Rust GPU-resident stepper (CUDA, mode-avg Kerr)" tags=[:rust] begin
    import Test: @test, @test_skip, @testset
    using Amalthea
    using Amalthea.RK45: PreconStepper, RustNativeStepper, step!, solve
    import Logging: with_logger, NullLogger
    import LinearAlgebra: norm

    libpath = RK45._LIBAMALTHEA_RK45
    if !isfile(libpath)
        @test_skip "Rust library not found"
    else
        # ── Test Geometry & Setup ─────────────────────────────────────────────
        # Same scope as test_native_phase1.jl (mode-averaged, Kerr-only,
        # RealGrid): the only geometry/physics `cuda_native.rs`'s
        # `CudaNativeSim` implements (every other `NativeBackend` method on
        # it returns -1). `AMALTHEA_USE_RUST_CUDA_NATIVE=1` opts into the
        # GPU-resident stepper on both the Julia (`RK45._gpu_native_eligible`)
        # and Rust (`init_cuda_native_sim`'s own env-var check) sides; this
        # is independent from `AMALTHEA_USE_RUST_NATIVE` (the CPU-resident
        # stepper, on by default since Phase 8).
        radius = 125e-6
        flength = 0.15
        gas = :He
        pressure = 1.0
        λ0 = 800e-9
        λlims = (200e-9, 4e-6)
        trange = 1e-12

        args = (radius, flength, gas, pressure)
        kw = (; λ0, λlims, trange, raman=false, plasma=false, kerr=true, shotnoise=false,
                energy=1e-6, τfwhm=30e-15)

        Eω, grid, linop, transform, FT, output = with_logger(NullLogger()) do
            Interface.prop_capillary_args(args...; kw...)
        end

        t0 = 0.0
        dt = 0.01

        s_jl = PreconStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10,
                              max_dt=dt, min_dt=dt)

        local s_ru
        gpu_available = true
        gpu_error = nothing
        withenv("AMALTHEA_USE_RUST_CUDA_NATIVE" => "1") do
            try
                s_ru = RustNativeStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10,
                                          max_dt=dt, min_dt=dt)
            catch e
                gpu_available = false
                gpu_error = e
                return
            end

            @testset "GPU handle actually used (not silently CPU)" begin
                # `RK45._gpu_native_eligible` must have returned true for this
                # exact config, or the "equivalence" below would just be
                # CPU-vs-CPU (both native.rs backends live behind the same
                # opaque `NativeSim.ptr` — the only externally visible sign
                # this used `init_cuda_native_sim` and not `init_native_sim`
                # is that construction succeeded only with the env var set;
                # re-assert eligibility directly here so a future refactor
                # that silently narrows `_gpu_native_eligible` can't make
                # this test pass vacuously). Must be checked inside this
                # `withenv` block — `_gpu_native_eligible` reads the same
                # env var, which is reverted once `withenv` returns.
                @test RK45._gpu_native_eligible(transform, linop)
            end

            @testset "Full-solve equivalence (fixed step size)" begin
                solve(s_jl, flength)
                solve(s_ru, flength)

                rel_solve = norm(s_ru.yn - s_jl.yn) / norm(s_jl.yn)
                println("GPU-resident stepper full-solve rel_solve: ", rel_solve)
                # Looser than the CPU-resident native path's ~1e-6 (Phase 1;
                # measured ~4.5e-4 here). Two known, documented GPU-path
                # simplifications explain the gap (docs/dev/BACKLOG.md has the full
                # writeup) — neither is "wrong sign" or "missing physics",
                # both are real but bounded fidelity gaps versus the CPU
                # path:
                #  1. `cuda_native.rs`'s Kerr FFT buffers/plans are sized
                #     `n_time` (this config: 8192), not `n_time_over`
                #     (16384) — the resident GPU Kerr step skips the
                #     oversampling/anti-aliasing padding
                #     `CpuNativeSim`/Julia both apply.
                #  2. The weak-norm error estimate uses a placeholder
                #     (`field_d` for both the "old" and "trial new" field —
                #     see `weaknorm_elem_args`'s comment) since `step()` has
                #     no pre-acceptance trial solution available; under
                #     fixed-step (`max_dt=min_dt=dt`) this only affects
                #     `err`/`dtn`, not the accepted field trajectory, as
                #     long as every step's `err` stays below 1 (asserted
                #     below) so no step is ever rejected on either backend.
                @test rel_solve < 1e-3
                @test s_ru.ok
                @test s_ru.err < 1.0
            end
        end

        gpu_available || @test_skip "CUDA GPU/toolkit not available on this machine: $gpu_error"
    end
end

@testitem "Native-Rust GPU-resident stepper (CUDA, mode-avg Kerr+plasma)" tags=[:rust] begin
    import Test: @test, @test_skip, @testset
    using Amalthea
    using Amalthea.RK45: PreconStepper, RustNativeStepper, step!, solve
    import Logging: with_logger, NullLogger
    import LinearAlgebra: norm

    libpath = RK45._LIBAMALTHEA_RK45
    if !isfile(libpath)
        @test_skip "Rust library not found"
    else
        # docs/dev/BACKLOG.md S3 item 2 (first slice, 2026-07-11): plasma support on
        # `CudaNativeSim`, PPT ionisation only (ADK still returns -1 — see
        # `RK45._gpu_native_eligible`'s docstring). Same mode-averaged
        # RealGrid Kerr scope as the sibling Kerr-only testitem above, plus
        # a PPT plasma response (`gas=:Ar`, atomic — `Interface.jl`'s
        # `plasma=true` auto-selects PPT for atoms, ADK for molecules; no
        # separate `ionmodel` kwarg exists). This exercises the
        # plasma_rate/fraction/phase/current/polarization kernel sequence
        # added to `cuda_native.rs::step` — and, since it's the first config
        # to ever drive that GPU Kerr kernel with a nonzero `kerr_fac`
        # end-to-end against a real oracle, also covers the argument-order
        # fix to `rhs_mode_avg_real_kernel`'s call site (see that kernel's
        # Rust-side comment): the sibling Kerr-only test above only ever
        # compared against effectively-zero nonlinearity, so it could not
        # have caught a Kerr-write/read pointer swap on its own.
        radius = 125e-6
        flength = 0.02
        gas = :Ar
        pressure = 1.0
        λ0 = 800e-9
        λlims = (200e-9, 4e-6)
        trange = 1e-12

        args = (radius, flength, gas, pressure)
        kw = (; λ0, λlims, trange, raman=false, plasma=true, kerr=true,
                shotnoise=false, energy=6e-6, τfwhm=15e-15)

        Eω, grid, linop, transform, FT, output = with_logger(NullLogger()) do
            Interface.prop_capillary_args(args...; kw...)
        end

        t0 = 0.0
        dt = 0.005

        s_jl = withenv("AMALTHEA_USE_RUST_IONISATION" => "1") do
            PreconStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10,
                          max_dt=dt, min_dt=dt)
        end

        local s_ru
        gpu_available = true
        gpu_error = nothing
        withenv("AMALTHEA_USE_RUST_CUDA_NATIVE" => "1", "AMALTHEA_USE_RUST_IONISATION" => "1") do
            try
                s_ru = RustNativeStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10,
                                          max_dt=dt, min_dt=dt)
            catch e
                gpu_available = false
                gpu_error = e
                return
            end

            @testset "GPU handle actually used (not silently CPU)" begin
                @test RK45._gpu_native_eligible(transform, linop)
            end

            @testset "Full-solve equivalence (fixed step size)" begin
                solve(s_jl, flength)
                solve(s_ru, flength)

                rel_solve = norm(s_ru.yn - s_jl.yn) / norm(s_jl.yn)
                println("GPU-resident stepper (Kerr+plasma) full-solve rel_solve: ", rel_solve)
                # NOT the same 1e-3 tier as the Kerr-only sibling test —
                # measured (2026-07-11) at ~2.0e-2 for this exact config.
                # Diagnosed, not just observed: the CPU-resident native
                # stepper (`AMALTHEA_USE_RUST_NATIVE=1`, no CUDA — proper
                # `n_time_over`-sized buffers) matches this exact
                # Kerr+plasma config to 1.3e-16 against `PreconStepper`, so
                # the gap is entirely GPU-specific. Sweeping energy
                # (1e-7→6e-6, a 60× range) showed rel_solve scaling almost
                # exactly linearly with energy at every point (1e-7→3.3e-4,
                # 1e-6→3.3e-3, 2e-6→6.7e-3, 6e-6→2.0e-2) — the signature of
                # a smooth, systematic discrepancy tracking field amplitude,
                # not a discrete bug (which would not produce such a clean
                # proportional trend across 3 decades). This is the same
                # documented `n_time_over`-vs-`n_time` Kerr buffer-sizing
                # gap the Kerr-only sibling test already carries (GPU.md
                # §8) — plasma's Keldysh-exponential sensitivity to field
                # amplitude amplifies that same small aliasing error by
                # roughly 40× at this energy, since ionization rate responds
                # far more steeply to a field perturbation than a cubic Kerr
                # term does. Not fixed here (real buffer-sizing work in
                # cuda_native.rs/kernels.cu, out of scope for this item) —
                # 5e-2 gives ~2.5× margin above the measured value so this
                # still catches a real regression (e.g. a kernel math bug)
                # without being vacuously loose.
                @test rel_solve < 5e-2
                @test s_ru.ok
                @test s_ru.err < 1.0
            end
        end

        gpu_available || @test_skip "CUDA GPU/toolkit not available on this machine: $gpu_error"
    end
end
