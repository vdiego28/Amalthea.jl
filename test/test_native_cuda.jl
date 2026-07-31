using TestItems

@testitem "Native-Rust GPU-resident stepper (CUDA, mode-avg Kerr)" tags=[:rust] begin
    import Test: @test, @test_skip, @testset
    using Amalthea
    using Amalthea.RK45: PreconStepper, RustNativeStepper, step!, solve
    import Logging: with_logger, NullLogger
    import LinearAlgebra: norm

    libpath = RK45._LIBAMALTHEA_RK45
    require_cuda = get(ENV, "AMALTHEA_REQUIRE_CUDA_TESTS", "0") == "1"
    if !isfile(libpath)
        require_cuda && error("CUDA tests are required, but the Rust library was not found")
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
        n = length(Eω)

        # ── Non-vacuousness measurement (AGENTS.md §3 step 4 / TESTING.md §1):
        # the entire point of BACKLOG.md S3 item 0 was that the previous
        # `rel_solve < 1e-3` tolerance was *looser* than this config's actual
        # nonlinear share, so a GPU backend computing zero nonlinearity passed
        # anyway. Measure that share directly (Julia oracle, kerr on vs off)
        # and assert both that it's non-negligible and that every tolerance
        # below sits far under it.
        kw_off = merge(kw, (; kerr=false))
        Eω_off, _, linop_off, transform_off, _, _ = with_logger(NullLogger()) do
            Interface.prop_capillary_args(args...; kw_off...)
        end
        s_jl_nl = PreconStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10,
                                 max_dt=dt, min_dt=dt)
        s_jl_lin = PreconStepper(transform_off, linop_off, copy(Eω_off), t0, dt, rtol=1e-6, atol=1e-10,
                                  max_dt=dt, min_dt=dt)
        solve(s_jl_nl, flength)
        solve(s_jl_lin, flength)
        rel_nl = norm(s_jl_nl.yn - s_jl_lin.yn) / norm(s_jl_nl.yn)
        println("Kerr-only config's nonlinear share (Julia oracle, kerr on vs off): ", rel_nl)
        # Measured ≈4.5e-4 — this is the number the old 1e-3 tolerance was
        # supposed to be tighter than, and wasn't.
        @test rel_nl > 1e-4

        s_jl = PreconStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10,
                              max_dt=dt, min_dt=dt)

        local s_ru
        gpu_available = true
        gpu_error = nothing
        # `AMALTHEA_NATIVE_GPU=on` forces the GPU path regardless of the
        # size-based `:auto` dispatch policy (docs/dev/BACKLOG.md S3 item 3)
        # — this test's whole point is exercising the GPU kernel directly at
        # a small, fast config, not the dispatch heuristic (that has its own
        # coverage in test_native_gpu_dispatch.jl).
        withenv("AMALTHEA_USE_RUST_CUDA_NATIVE" => "1", "AMALTHEA_NATIVE_GPU" => "on") do
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
                # env vars, which are reverted once `withenv` returns.
                @test RK45._gpu_native_eligible(transform, linop, length(Eω))
            end

            @testset "Stage-derivative structural check (GPU vs CPU-native)" begin
                # BACKLOG.md S3 item 0's root cause (`set_mode_avg_params`
                # discarding scaling/normalization/window arrays) made the
                # GPU RHS compute ~zero nonlinearity — invisible to a
                # solve-level tolerance that happened to be looser than the
                # true nonlinear effect. This check catches that whole
                # failure class directly: it compares the raw per-stage RK
                # derivative (`ks[i]`, fetched via `get_ks_stage`) between
                # the GPU backend and the CPU-resident native backend
                # (`AMALTHEA_USE_RUST_NATIVE=1`, no CUDA — itself
                # equivalence-tested against the Julia oracle in
                # test_native_phase1.jl to ~1e-13/1e-6), independent of any
                # subsequent RK accumulation, error control, or FFTW/cuFFT
                # accumulated drift over many steps.
                s_cpu_native = withenv("AMALTHEA_USE_RUST_CUDA_NATIVE" => "0") do
                    RustNativeStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10,
                                       max_dt=dt, min_dt=dt)
                end

                get_ks(handle_ptr, idx) = begin
                    kbuf = zeros(ComplexF64, n)
                    rc = ccall((:get_ks_stage, libpath), Cint,
                          (Ptr{Cvoid}, Csize_t, Ptr{ComplexF64}, Csize_t),
                          handle_ptr, Csize_t(idx), kbuf, Csize_t(n))
                    rc == 0 || error("get_ks_stage failed rc=$rc")
                    kbuf
                end

                # Probe immediately after construction, before any `step!` —
                # this is exactly where the second bug found during this
                # item's design review lived: `CudaNativeSim::set_field`
                # never seeded `ks_d[0]`, unlike `CpuNativeSim::set_field`
                # (see docs/dev/native-port/portlog-inbox/gpu-nonlinearity.md),
                # so `ks_d[0]` before the first `step!` was whatever
                # `cuMemAlloc` happened to return.
                k0_cpu_pre = get_ks(s_cpu_native._handle.ptr, 0)
                k0_gpu_pre = get_ks(s_ru._handle.ptr, 0)
                maxk_cpu_pre = maximum(abs.(k0_cpu_pre))
                # Non-vacuousness: the true k1 for this config/dt is large
                # (~1e3), not the ~3.5e-13 the zero-nonlinearity backend
                # measured (BACKLOG.md S3 item 0).
                @test maxk_cpu_pre > 100.0
                # Measured ~1.04e-15 on real hardware (RTX 5060 Ti) — cuFFT
                # vs FFTW-backed CpuNativeSim agree to FP noise for a single
                # deterministic stage evaluation (TESTING.md §2's
                # reassociation tier, ~1e-13, with >1000x margin to spare).
                # 1e-12 is the tightest tier that still leaves ample room for
                # a different GPU/driver/cuFFT version to diverge slightly
                # without being a false positive.
                @test norm(k0_gpu_pre - k0_cpu_pre) / norm(k0_cpu_pre) < 1e-12

                # Probe every stage after one accepted step.
                step!(s_cpu_native)
                step!(s_ru)
                maxk_cpu = 0.0
                maxk_gpu = 0.0
                for idx in 0:6
                    kc = get_ks(s_cpu_native._handle.ptr, idx)
                    kg = get_ks(s_ru._handle.ptr, idx)
                    maxk_cpu = max(maxk_cpu, maximum(abs.(kc)))
                    maxk_gpu = max(maxk_gpu, maximum(abs.(kg)))
                    @test norm(kg - kc) / norm(kc) < 1e-12
                end
                println("Stage-derivative check: max|k_i| CPU=", maxk_cpu, " GPU=", maxk_gpu)
                @test maxk_cpu > 100.0
                @test maxk_gpu > 100.0
            end

            @testset "Adaptive rejection is transactional (GPU vs CPU-native)" begin
                # dt=0.1 is deliberately above this config's acceptance
                # range (measured err≈6). Unlike the fixed-step checks, the
                # controller is free to reject and reduce dt.
                dt_reject = 0.1
                s_cpu_adapt = withenv("AMALTHEA_USE_RUST_CUDA_NATIVE" => "0",
                                      "AMALTHEA_NATIVE_GPU" => "off") do
                    RustNativeStepper(transform, linop, copy(Eω), t0, dt_reject,
                                      rtol=1e-6, atol=1e-10,
                                      max_dt=0.2, min_dt=0.0)
                end
                s_gpu_adapt = RustNativeStepper(transform, linop, copy(Eω), t0, dt_reject,
                                                rtol=1e-6, atol=1e-10,
                                                max_dt=0.2, min_dt=0.0)
                field_before = copy(s_gpu_adapt.yn)

                @test !step!(s_cpu_adapt)
                @test !step!(s_gpu_adapt)
                # Rejection must not mutate the resident field. This is the
                # behavioral reason the pre-acceptance y5 lives in
                # `ystage_d` and is only swapped into `field_d` on accept.
                @test s_gpu_adapt.yn == field_before
                @test s_gpu_adapt.tn == t0
                @test s_gpu_adapt.err > 1
                @test isapprox(s_gpu_adapt.err, s_cpu_adapt.err; rtol=1e-11)
                @test isapprox(s_gpu_adapt.dtn, s_cpu_adapt.dtn; rtol=1e-11)

                # Retry with the controller-selected dt, then continue a
                # genuinely adaptive trajectory. This exercises accepted
                # trial-buffer swapping and deferred FSAL after a rejection.
                accepted_retry = false
                for _ in 1:5
                    cpu_ok = step!(s_cpu_adapt)
                    gpu_ok = step!(s_gpu_adapt)
                    @test gpu_ok == cpu_ok
                    @test isapprox(s_gpu_adapt.err, s_cpu_adapt.err; rtol=1e-10)
                    if cpu_ok
                        accepted_retry = true
                        break
                    end
                end
                @test accepted_retry
                @test isapprox(s_gpu_adapt.err, s_cpu_adapt.err; rtol=1e-10)
                @test norm(s_gpu_adapt.yn - s_cpu_adapt.yn) /
                      norm(s_cpu_adapt.yn) < 1e-12

                solve(s_cpu_adapt, flength)
                solve(s_gpu_adapt, flength)
                rel_adaptive = norm(s_gpu_adapt.yn - s_cpu_adapt.yn) /
                               norm(s_cpu_adapt.yn)
                println("GPU adaptive rejection/trajectory rel diff: ", rel_adaptive)
                @test rel_adaptive < 1e-6
            end

            @testset "locextrap=false matches CPU-native and Julia" begin
                false_kw = (; rtol=1e-6, atol=1e-10, locextrap=false,
                            max_dt=dt, min_dt=dt)
                s_jl_false = PreconStepper(transform, linop, copy(Eω), t0, dt;
                                           false_kw...)
                s_cpu_false = withenv("AMALTHEA_USE_RUST_CUDA_NATIVE" => "0",
                                      "AMALTHEA_NATIVE_GPU" => "off") do
                    RustNativeStepper(transform, linop, copy(Eω), t0, dt;
                                      false_kw...)
                end
                s_gpu_false = RustNativeStepper(transform, linop, copy(Eω), t0, dt;
                                                false_kw...)
                s_jl_true = PreconStepper(transform, linop, copy(Eω), t0, dt;
                                          rtol=1e-6, atol=1e-10,
                                          locextrap=true,
                                          max_dt=dt, min_dt=dt)

                @test step!(s_jl_false)
                @test step!(s_cpu_false)
                @test step!(s_gpu_false)
                @test step!(s_jl_true)
                @test norm(s_cpu_false.yn - s_jl_false.yn) /
                      norm(s_jl_false.yn) < 1e-13
                @test norm(s_gpu_false.yn - s_jl_false.yn) /
                      norm(s_jl_false.yn) < 1e-12
                @test isapprox(s_gpu_false.err, s_cpu_false.err; rtol=1e-10)
                @test norm(s_jl_false.yn - s_jl_true.yn) /
                      norm(s_jl_false.yn) > 1e-12

                # The false-mode trial buffer is also transactional on a
                # deliberate rejection; neither backend may expose it as the
                # resident field until the controller accepts.
                dt_reject = 0.1
                rejected_cpu = withenv("AMALTHEA_USE_RUST_CUDA_NATIVE" => "0",
                                       "AMALTHEA_NATIVE_GPU" => "off") do
                    RustNativeStepper(transform, linop, copy(Eω), t0, dt_reject;
                                      rtol=1e-6, atol=1e-10,
                                      locextrap=false)
                end
                rejected_gpu = RustNativeStepper(transform, linop, copy(Eω),
                                                 t0, dt_reject;
                                                 rtol=1e-6, atol=1e-10,
                                                 locextrap=false)
                @test !step!(rejected_cpu)
                @test !step!(rejected_gpu)
                @test rejected_cpu.yn == Eω
                @test rejected_gpu.yn == Eω
                @test isapprox(rejected_gpu.err, rejected_cpu.err; rtol=1e-10)
                @test isapprox(rejected_gpu.dtn, rejected_cpu.dtn; rtol=1e-10)

                # Continue the accepted fixed trajectory through several
                # deferred FSAL carries.
                for _ in 1:3
                    @test step!(s_jl_false)
                    @test step!(s_cpu_false)
                    @test step!(s_gpu_false)
                end
                @test norm(s_cpu_false.yn - s_jl_false.yn) /
                      norm(s_jl_false.yn) < 1e-13
                @test norm(s_gpu_false.yn - s_jl_false.yn) /
                      norm(s_jl_false.yn) < 1e-12
            end

            @testset "Full-solve equivalence (fixed step size)" begin
                solve(s_jl, flength)
                solve(s_ru, flength)

                rel_solve = norm(s_ru.yn - s_jl.yn) / norm(s_jl.yn)
                println("GPU-resident stepper full-solve rel_solve: ", rel_solve)
                # BACKLOG.md S3 item 0 fix: `set_mode_avg_params` now uploads
                # `pre`/`beta`/`sidx`/`owin`/`nlscale`/`sqrt_aeff` and the RHS
                # pipeline (`compute_rhs_mode_avg`, cuda_native.rs) ports the
                # CPU oracle's Steps 1/2/5/6/7 (oversampled crop+IFFT, input
                # scaling, forward-FFT crop+scale, norm_pre_beta, ωwin) that
                # were previously skipped entirely, and all Kerr buffers/plans
                # are now sized `n_time_over` (BACKLOG.md S3 item 6, folded in
                # here since the crop/pad kernels need it). Measured on real
                # hardware (RTX 5060 Ti, driver 610.43.02, CUDA 13.3):
                # rel_solve = 3.5e-16, i.e. GPU (cuFFT) now matches the Julia
                # oracle (FFTW) to within FP noise for this 16-step fixed-step
                # config — tighter than the reassociation tier (~1e-13) this
                # port would normally be held to (TESTING.md §2), not merely
                # the previous 1e-3. Per TESTING.md §2 ("pick the tightest
                # tier the math justifies"), 1e-12 pins that reassociation
                # tier directly (>1000x margin above the measured value, so
                # a different GPU/driver/cuFFT version has room to diverge
                # slightly without a false failure) while sitting >1e8x
                # *below* the nonlinear share (rel_nl≈4.5e-4) asserted above
                # — a regression back to (near-)zero nonlinearity fails this
                # test immediately and by a wide margin.
                @test rel_solve < 1e-12
                @test rel_nl > 100 * 1e-12
                @test s_ru.ok
                # The GPU now forms the true fifth-order trial before
                # acceptance and evaluates the same global weak norm as the
                # CPU/Julia oracle. Keep the fixed-step value diagnostic
                # here; the adaptive test above is the actual rejection gate.
                println("err diagnostic — s_ru.err=", s_ru.err,
                        " s_jl.err=", s_jl.err)
                @test isapprox(s_ru.err, s_jl.err; rtol=1e-10)
            end

            @testset "Luna.run / dense-output equivalence (adaptive stepping)" begin
                # Closes a real blind spot: every check above drives the
                # stepper via raw `solve()`/`step!()`, exactly like every
                # prior GPU test — the same class of gap that hid the Phase
                # 8 windowing bug and the dense-output order bug (CLAUDE.md's
                # Phase 8 gotchas; `cuda_native.rs::apply_prop`'s own
                # comment). `interpolate(s::RustNativeStepper, ti)`
                # (RK45.jl) computes a dense-output correction from
                # `get_ks_stage` — before this item's fix those `k` values
                # were ~3.5e-13 (below FP noise), so the correction term was
                # negligible regardless of whether `interpolate`'s formula
                # was even right; now they are ~1e3, so this is the first
                # time the interpolated *value* (not just "did it throw") is
                # actually exercised by a nonzero RHS on this backend. Uses
                # `prop_capillary` (adaptive stepping, `saveN=11`) rather
                # than the fixed-step raw stepper above — this is the actual
                # `Luna.run` path a user hits.
                kw_dense = merge(kw, (; saveN=11))
                out_julia = withenv("AMALTHEA_USE_RUST_NATIVE" => "0") do
                    with_logger(NullLogger()) do
                        prop_capillary(args...; kw_dense...)
                    end
                end
                out_gpu = withenv("AMALTHEA_USE_RUST_CUDA_NATIVE" => "1",
                                  "AMALTHEA_NATIVE_GPU" => "on") do
                    with_logger(NullLogger()) do
                        prop_capillary(args...; kw_dense...)
                    end
                end

                rel_final = norm(out_gpu["Eω"][:, end] - out_julia["Eω"][:, end]) /
                            norm(out_julia["Eω"][:, end])
                println("Luna.run rel diff at final save: ", rel_final)
                # ~1e-6 floor tier (TESTING.md §3): adaptive stepping means
                # the two integrators do not necessarily take an identical
                # step-size sequence (unlike the fixed-step tests above), so
                # this is held to the same tier as `test_native_phase1.jl`'s
                # own full-solve check, not the fixed-step reassociation
                # tier. Measured 1.25e-7 on real hardware.
                @test rel_final < 1e-6

                # The intermediate saves are the ones that actually exercise
                # `interpolate`'s dense-output formula (the final save is the
                # last *accepted* step's own state, not an interpolated
                # value) — this is the check GPU.md §8 and the Phase 8
                # gotcha say was never done for this backend. Measured
                # 1.8e-10 to 7.7e-8 across all 9 intermediate saves.
                for i in 2:size(out_julia["Eω"], 2)-1
                    rel_i = norm(out_gpu["Eω"][:, i] - out_julia["Eω"][:, i]) /
                            norm(out_julia["Eω"][:, i])
                    @test rel_i < 1e-6
                end
            end
        end

        if !gpu_available
            require_cuda && error("CUDA tests are required, but GPU setup failed: $gpu_error")
            @test_skip "CUDA GPU/toolkit not available on this machine: $gpu_error"
        end
    end
end

@testitem "Native-Rust GPU-resident stepper (CUDA, mode-avg ADK plasma)" tags=[:rust] begin
    import Test: @test, @test_skip, @testset
    using Amalthea
    using Amalthea.RK45: PreconStepper, RustNativeStepper, step!, solve
    import Logging: with_logger, NullLogger
    import LinearAlgebra: norm

    libpath = RK45._LIBAMALTHEA_RK45
    require_cuda = get(ENV, "AMALTHEA_REQUIRE_CUDA_TESTS", "0") == "1"
    if !isfile(libpath)
        require_cuda && error("CUDA tests are required, but the Rust library was not found")
        @test_skip "Rust library not found"
    else
        # This is deliberately the known non-vacuous ADK configuration from
        # test_native_adk_ionisation.jl: lower energy would put every sample
        # below ADK's threshold and make a no-op GPU rate appear correct.
        args = (125e-6, 0.02, :He, 1.0)
        kw = (; λ0=800e-9, λlims=(150e-9, 4e-6), trange=500e-15,
              raman=false, plasma=:ADK, kerr=true, shotnoise=false,
              energy=1.6e-3, τfwhm=30e-15)
        Eω, grid, linop, transform, _, _ = with_logger(NullLogger()) do
            Interface.prop_capillary_args(args...; kw...)
        end
        Eω_off, _, linop_off, transform_off, _, _ = with_logger(NullLogger()) do
            Interface.prop_capillary_args(args...; merge(kw, (; plasma=false))...)
        end
        t0, dt = 0.0, 0.005

        # The Julia control establishes that this exact ADK response changes
        # the physics by far more than every equivalence tolerance below.
        s_jl_on = PreconStepper(transform, linop, copy(Eω), t0, dt;
                                rtol=1e-6, atol=1e-10, max_dt=dt, min_dt=dt)
        s_jl_off = PreconStepper(transform_off, linop_off, copy(Eω_off), t0, dt;
                                 rtol=1e-6, atol=1e-10, max_dt=dt, min_dt=dt)
        solve(s_jl_on, args[2])
        solve(s_jl_off, args[2])
        adk_effect = norm(s_jl_on.yn - s_jl_off.yn) / norm(s_jl_on.yn)
        @test adk_effect > 100 * 1e-10

        gpu_available = true
        gpu_error = nothing
        withenv("AMALTHEA_USE_RUST_CUDA_NATIVE" => "1",
                "AMALTHEA_USE_RUST_IONISATION" => "1",
                "AMALTHEA_NATIVE_GPU" => "on") do
            local s_gpu
            try
                s_gpu = RustNativeStepper(transform, linop, copy(Eω), t0, dt;
                                           rtol=1e-6, atol=1e-10,
                                           max_dt=dt, min_dt=dt)
            catch e
                gpu_available = false
                gpu_error = e
                return
            end
            @test RK45._gpu_native_eligible(transform, linop, length(Eω))

            s_cpu = withenv("AMALTHEA_USE_RUST_CUDA_NATIVE" => "0",
                            "AMALTHEA_NATIVE_GPU" => "off") do
                RustNativeStepper(transform, linop, copy(Eω), t0, dt;
                                  rtol=1e-6, atol=1e-10, max_dt=dt, min_dt=dt)
            end
            getk(s, i) = begin
                k = zeros(ComplexF64, length(Eω))
                @test ccall((:get_ks_stage, libpath), Cint,
                            (Ptr{Cvoid}, Csize_t, Ptr{ComplexF64}, Csize_t),
                            s._handle.ptr, Csize_t(i), k, Csize_t(length(k))) == 0
                k
            end
            @testset "Nonzero ADK stage scale and CPU-native agreement" begin
                k_cpu, k_gpu = getk(s_cpu, 0), getk(s_gpu, 0)
                @test maximum(abs.(k_cpu)) > 100
                @test norm(k_gpu - k_cpu) / norm(k_cpu) < 1e-12
            end

            @testset "Fixed-step GPU, CPU, and Julia trajectories" begin
                solve(s_cpu, args[2])
                solve(s_gpu, args[2])
                @test norm(s_gpu.yn - s_cpu.yn) / norm(s_cpu.yn) < 1e-12
                @test norm(s_gpu.yn - s_jl_on.yn) / norm(s_jl_on.yn) < 1e-10
            end

            @testset "ADK rejection is bit-exact and retry remains adaptive" begin
                reject_dt = 0.1
                s_cpu_retry = withenv("AMALTHEA_USE_RUST_CUDA_NATIVE" => "0",
                                      "AMALTHEA_NATIVE_GPU" => "off") do
                    RustNativeStepper(transform, linop, copy(Eω), t0, reject_dt;
                                      rtol=1e-6, atol=1e-10, max_dt=0.1, min_dt=0.0)
                end
                s_gpu_retry = RustNativeStepper(transform, linop, copy(Eω), t0, reject_dt;
                                                rtol=1e-6, atol=1e-10,
                                                max_dt=0.1, min_dt=0.0)
                before = copy(s_gpu_retry.yn)
                @test !step!(s_cpu_retry)
                @test !step!(s_gpu_retry)
                @test s_gpu_retry.yn == before
                @test isapprox(s_gpu_retry.err, s_cpu_retry.err; rtol=1e-10)
                @test isapprox(s_gpu_retry.dtn, s_cpu_retry.dtn; rtol=1e-10)
                accepted = false
                for _ in 1:8
                    cpu_ok, gpu_ok = step!(s_cpu_retry), step!(s_gpu_retry)
                    @test gpu_ok == cpu_ok
                    if gpu_ok
                        accepted = true
                        break
                    end
                end
                @test accepted
                solve(s_cpu_retry, args[2])
                solve(s_gpu_retry, args[2])
                @test norm(s_gpu_retry.yn - s_cpu_retry.yn) / norm(s_cpu_retry.yn) < 1e-6
            end
        end
        if !gpu_available
            require_cuda && error("CUDA tests are required, but GPU setup failed: $gpu_error")
            @test_skip "CUDA GPU/toolkit not available on this machine: $gpu_error"
        end
    end
end

@testitem "Native-Rust GPU-resident stepper (CUDA, mode-avg Kerr+plasma)" tags=[:rust] begin
    import Test: @test, @test_skip, @testset
    using Amalthea
    using Amalthea.RK45: PreconStepper, RustNativeStepper, step!, solve
    import Logging: with_logger, NullLogger
    import LinearAlgebra: norm

    libpath = RK45._LIBAMALTHEA_RK45
    require_cuda = get(ENV, "AMALTHEA_REQUIRE_CUDA_TESTS", "0") == "1"
    if !isfile(libpath)
        require_cuda && error("CUDA tests are required, but the Rust library was not found")
        @test_skip "Rust library not found"
    else
        # docs/dev/BACKLOG.md S3: this sibling exercises the PPT rate branch;
        # ADK has its own explicit CUDA item above. Same mode-averaged
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

        # ── Non-vacuousness measurement (same rule as the Kerr-only sibling
        # test — TESTING.md §1 / AGENTS.md §3 step 4). This config's own
        # nonlinear (Kerr+plasma) share is much larger than the Kerr-only
        # config's, so it deserves its own measurement rather than reusing
        # the sibling's number.
        kw_off = merge(kw, (; kerr=false, plasma=false))
        Eω_off, _, linop_off, transform_off, _, _ = with_logger(NullLogger()) do
            Interface.prop_capillary_args(args...; kw_off...)
        end
        s_jl_nl = withenv("AMALTHEA_USE_RUST_IONISATION" => "1") do
            PreconStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10,
                          max_dt=dt, min_dt=dt)
        end
        s_jl_lin = PreconStepper(transform_off, linop_off, copy(Eω_off), t0, dt, rtol=1e-6, atol=1e-10,
                                  max_dt=dt, min_dt=dt)
        solve(s_jl_nl, flength)
        solve(s_jl_lin, flength)
        rel_nl = norm(s_jl_nl.yn - s_jl_lin.yn) / norm(s_jl_nl.yn)
        println("Kerr+plasma config's nonlinear share (Julia oracle, on vs off): ", rel_nl)
        # Measured ≈2.0e-2 — much larger than the Kerr-only sibling's 4.5e-4,
        # as expected (plasma's Keldysh-exponential sensitivity to field
        # amplitude).
        @test rel_nl > 1e-3

        s_jl = withenv("AMALTHEA_USE_RUST_IONISATION" => "1") do
            PreconStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10,
                          max_dt=dt, min_dt=dt)
        end

        local s_ru
        gpu_available = true
        gpu_error = nothing
        # `AMALTHEA_NATIVE_GPU=on` forces GPU independently of `:auto`'s
        # size threshold. This deliberately small case is for numerical
        # correctness; the parallel PPT scans and dispatch crossover are
        # benchmarked separately (BACKLOG.md S3 items 3 and 12).
        withenv("AMALTHEA_USE_RUST_CUDA_NATIVE" => "1", "AMALTHEA_USE_RUST_IONISATION" => "1",
                "AMALTHEA_NATIVE_GPU" => "on") do
            try
                s_ru = RustNativeStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10,
                                          max_dt=dt, min_dt=dt)
            catch e
                gpu_available = false
                gpu_error = e
                return
            end

            @testset "GPU handle actually used (not silently CPU)" begin
                @test RK45._gpu_native_eligible(transform, linop, length(Eω))
            end

            @testset "Adaptive Kerr+PPT rejection and retry" begin
                # At the fixed-step test's dt this plasma-sensitive config
                # has err≈1.82. With min_dt=0 it must reject, preserve the
                # field, and accept the controller-selected retry.
                s_cpu_adapt = withenv("AMALTHEA_USE_RUST_CUDA_NATIVE" => "0",
                                      "AMALTHEA_NATIVE_GPU" => "off") do
                    RustNativeStepper(transform, linop, copy(Eω), t0, dt,
                                      rtol=1e-6, atol=1e-10,
                                      max_dt=0.02, min_dt=0.0)
                end
                s_gpu_adapt = RustNativeStepper(transform, linop, copy(Eω), t0, dt,
                                                rtol=1e-6, atol=1e-10,
                                                max_dt=0.02, min_dt=0.0)
                field_before = copy(s_gpu_adapt.yn)

                @test !step!(s_cpu_adapt)
                @test !step!(s_gpu_adapt)
                @test s_gpu_adapt.yn == field_before
                @test s_gpu_adapt.err > 1
                @test isapprox(s_gpu_adapt.err, s_cpu_adapt.err; rtol=1e-11)
                @test isapprox(s_gpu_adapt.dtn, s_cpu_adapt.dtn; rtol=1e-11)

                @test step!(s_cpu_adapt)
                @test step!(s_gpu_adapt)
                @test norm(s_gpu_adapt.yn - s_cpu_adapt.yn) /
                      norm(s_cpu_adapt.yn) < 1e-12

                solve(s_cpu_adapt, flength)
                solve(s_gpu_adapt, flength)
                rel_adaptive = norm(s_gpu_adapt.yn - s_cpu_adapt.yn) /
                               norm(s_cpu_adapt.yn)
                println("GPU Kerr+PPT adaptive trajectory rel diff: ", rel_adaptive)
                @test rel_adaptive < 1e-6
                @test rel_nl > 100 * 1e-6
            end

            @testset "Full-solve equivalence (fixed step size)" begin
                solve(s_jl, flength)
                solve(s_ru, flength)

                rel_solve = norm(s_ru.yn - s_jl.yn) / norm(s_jl.yn)
                println("GPU-resident stepper (Kerr+plasma) full-solve rel_solve: ", rel_solve)
                # NOT the same tier as the sibling Kerr-only test's prior
                # history — this comment supersedes the pre-fix version's
                # ~2.0e-2/5e-2 rationale entirely. That rationale attributed
                # the gap to the `n_time`-vs-`n_time_over` Kerr/plasma buffer
                # sizing fidelity gap (BACKLOG.md S3 item 6); this item fixes
                # that gap directly (all Kerr/plasma buffers now sized
                # `n_time_over`, cuFFT plans likewise), so the old
                # measurement and its justification are stale, not just the
                # tolerance number. Re-measured on real hardware (RTX 5060
                # Ti, driver 610.43.02, CUDA 13.3): rel_solve = 1.8e-16 — GPU
                # (cuFFT + the same PPT ionisation LUT format) now matches
                # the Julia oracle to within FP noise for this 5-step
                # fixed-step config. Per TESTING.md §2 ("pick the tightest
                # tier the math justifies"), 1e-12 pins the reassociation
                # tier directly (>1000x margin above the measured value)
                # while sitting >1e10x below the nonlinear share
                # (rel_nl≈2.0e-2) asserted above.
                @test rel_solve < 1e-12
                @test rel_nl > 100 * 1e-12
                @test s_ru.ok
                println("err diagnostic — s_ru.err=", s_ru.err,
                        " s_jl.err=", s_jl.err)
                @test isapprox(s_ru.err, s_jl.err; rtol=1e-10)
            end
        end

        if !gpu_available
            require_cuda && error("CUDA tests are required, but GPU setup failed: $gpu_error")
            @test_skip "CUDA GPU/toolkit not available on this machine: $gpu_error"
        end
    end
end
