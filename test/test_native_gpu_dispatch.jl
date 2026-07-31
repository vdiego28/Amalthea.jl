using TestItems

@testitem "Native-Rust GPU dispatch policy (:off/:on/:auto)" tags=[:rust] begin
    import Test: @test, @test_skip, @testset
    using Amalthea
    using Amalthea.RK45: RustNativeStepper
    import Logging: with_logger, NullLogger

    # docs/dev/BACKLOG.md S3 item 3: `AMALTHEA_NATIVE_GPU=off/on/auto` layered on
    # top of `AMALTHEA_USE_RUST_CUDA_NATIVE`'s master opt-in. This test only
    # exercises `RK45._gpu_native_eligible`'s pure-Julia decision (no `ccall`,
    # no GPU hardware needed) — the sibling GPU-vs-CPU numerical-equivalence
    # tests in test_native_cuda.jl are the ones that need real CUDA hardware
    # and self-skip without it.
    libpath = RK45._LIBAMALTHEA_RK45
    if !isfile(libpath)
        @test_skip "Rust library not found"
    else
        radius = 125e-6
        flength = 0.15
        gas = :He
        pressure = 1.0
        λ0 = 800e-9
        λlims = (200e-9, 4e-6)
        args = (radius, flength, gas, pressure)

        # Below _GPU_KERR_ONLY_N_THRESHOLD (16384) — trange=1e-12 -> n=4097.
        kw_small = (; λ0, λlims, trange=1e-12, raman=false, plasma=false, kerr=true,
                      shotnoise=false, energy=1e-6, τfwhm=30e-15)
        # At/above the threshold — trange=4e-12 -> n=16385.
        kw_large = (; λ0, λlims, trange=4e-12, raman=false, plasma=false, kerr=true,
                      shotnoise=false, energy=1e-6, τfwhm=30e-15)
        # PPT uses its own measured threshold after the parallel prefix-scan
        # work: n=4097 is below it; n=8193 is above it.
        kw_plasma_small = (; λ0, λlims, trange=1e-12, raman=false, plasma=true,
                             kerr=true, shotnoise=false, energy=1e-6, τfwhm=30e-15)
        kw_plasma_large = (; λ0, λlims, trange=2e-12, raman=false, plasma=true,
                             kerr=true, shotnoise=false, energy=1e-6, τfwhm=30e-15)
        kw_adk_small = (; λ0, λlims, trange=1e-12, raman=false, plasma=:ADK,
                         kerr=true, shotnoise=false, energy=1.6e-3, τfwhm=30e-15)
        kw_adk_large = (; λ0, λlims, trange=2e-12, raman=false, plasma=:ADK,
                         kerr=true, shotnoise=false, energy=1.6e-3, τfwhm=30e-15)

        Eω_small, _, linop_small, transform_small, _, _ = with_logger(NullLogger()) do
            Interface.prop_capillary_args(args...; kw_small...)
        end
        Eω_large, _, linop_large, transform_large, _, _ = with_logger(NullLogger()) do
            Interface.prop_capillary_args(args...; kw_large...)
        end
        Eω_plasma_small, _, linop_plasma_small, transform_plasma_small, _, _ =
            with_logger(NullLogger()) do
                Interface.prop_capillary_args(args...; kw_plasma_small...)
            end
        Eω_plasma_large, _, linop_plasma_large, transform_plasma_large, _, _ =
            with_logger(NullLogger()) do
                Interface.prop_capillary_args(args...; kw_plasma_large...)
            end
        Eω_adk_small, _, linop_adk_small, transform_adk_small, _, _ =
            with_logger(NullLogger()) do
                Interface.prop_capillary_args(args...; kw_adk_small...)
            end
        Eω_adk_large, _, linop_adk_large, transform_adk_large, _, _ =
            with_logger(NullLogger()) do
                Interface.prop_capillary_args(args...; kw_adk_large...)
            end

        # `threshold=false` intentionally retains the Julia/CPU ADK formula's
        # zero-field behavior (which can yield NaN); CUDA's pointwise kernel
        # therefore must never be selected for it. Build this response by hand
        # rather than changing Interface's public default (`threshold=true`).
        adk_unthresholded = Ionisation.IonRateADK(
            PhysData.ionisation_potential(:He); threshold=false)
        plasma_unthresholded = Nonlinear.PlasmaCumtrapz(
            transform_adk_large.grid.to, transform_adk_large.grid.to,
            adk_unthresholded, PhysData.ionisation_potential(:He))
        transform_adk_unthresholded = NonlinearRHS.TransModeAvg(
            transform_adk_large.Pto, transform_adk_large.Eto,
            transform_adk_large.Eωo, transform_adk_large.Pωo,
            transform_adk_large.FT,
            (transform_adk_large.resp[1], plasma_unthresholded),
            transform_adk_large.grid, transform_adk_large.densityfun,
            transform_adk_large.norm!, transform_adk_large.aeff,
            transform_adk_large.Et_noise, transform_adk_large.Et_nl)

        @test length(Eω_plasma_small) < RK45._GPU_PPT_N_THRESHOLD
        @test length(Eω_plasma_large) >= RK45._GPU_PPT_N_THRESHOLD
        @test length(Eω_adk_small) < RK45._GPU_ADK_N_THRESHOLD
        @test length(Eω_adk_large) >= RK45._GPU_ADK_N_THRESHOLD
        @test RK45._GPU_ADK_N_THRESHOLD == 8193
        @test !transform_adk_unthresholded.resp[2].ratefunc.threshold

        @test length(Eω_small) < RK45._GPU_KERR_ONLY_N_THRESHOLD
        @test length(Eω_large) >= RK45._GPU_KERR_ONLY_N_THRESHOLD

        @testset "master switch off -> never eligible regardless of gpu_dispatch" begin
            withenv("AMALTHEA_USE_RUST_CUDA_NATIVE" => nothing, "AMALTHEA_NATIVE_GPU" => "on") do
                @test !RK45._gpu_native_eligible(transform_large, linop_large, length(Eω_large))
            end
        end

        withenv("AMALTHEA_USE_RUST_CUDA_NATIVE" => "1") do
            @testset "gpu_dispatch=off -> never eligible" begin
                withenv("AMALTHEA_NATIVE_GPU" => "off") do
                    @test !RK45._gpu_native_eligible(transform_small, linop_small, length(Eω_small))
                    @test !RK45._gpu_native_eligible(transform_large, linop_large, length(Eω_large))
                end
            end

            @testset "gpu_dispatch=on -> eligible regardless of size" begin
                withenv("AMALTHEA_NATIVE_GPU" => "on") do
                    @test RK45._gpu_native_eligible(transform_small, linop_small, length(Eω_small))
                    @test RK45._gpu_native_eligible(transform_large, linop_large, length(Eω_large))
                    @test RK45._gpu_native_eligible(transform_plasma_small,
                                                    linop_plasma_small,
                                                    length(Eω_plasma_small))
                    @test RK45._gpu_native_eligible(transform_adk_small,
                                                    linop_adk_small,
                                                    length(Eω_adk_small))
                    @test !RK45._gpu_kernel_supports(transform_adk_unthresholded,
                                                      linop_adk_large)
                    @test !RK45._gpu_native_eligible(transform_adk_unthresholded,
                                                       linop_adk_large,
                                                       length(Eω_adk_large))
                    # Construction succeeds through the CPU fallback even
                    # though GPU dispatch was explicitly forced on; the two
                    # false eligibility checks above are the observable
                    # dispatch decision (the opaque handle exposes no backend
                    # type query).
                    s_cpu_fallback = RustNativeStepper(
                        transform_adk_unthresholded, linop_adk_large,
                        copy(Eω_adk_large), 0.0, 0.005;
                        rtol=1e-6, atol=1e-10, max_dt=0.005, min_dt=0.005)
                    @test s_cpu_fallback isa RustNativeStepper
                end
            end

            @testset "gpu_dispatch=auto (default) -> size- and plasma-gated" begin
                withenv("AMALTHEA_NATIVE_GPU" => nothing) do
                    @test Amalthea.Config.backend_config().gpu_dispatch === :auto
                    @test !RK45._gpu_native_eligible(transform_small, linop_small, length(Eω_small))
                    @test RK45._gpu_native_eligible(transform_large, linop_large, length(Eω_large))
                    @test !RK45._gpu_native_eligible(transform_plasma_small,
                                                     linop_plasma_small,
                                                     length(Eω_plasma_small))
                    @test RK45._gpu_native_eligible(transform_plasma_large,
                                                    linop_plasma_large,
                                                    length(Eω_plasma_large))
                    @test !RK45._gpu_native_eligible(transform_adk_small,
                                                     linop_adk_small,
                                                     length(Eω_adk_small))
                    @test RK45._gpu_native_eligible(transform_adk_large,
                                                    linop_adk_large,
                                                    length(Eω_adk_large))
                    @test !RK45._gpu_native_eligible(transform_adk_unthresholded,
                                                     linop_adk_large,
                                                     length(Eω_adk_large))
                    # The benchmark measured only 8193, so 8192 must not be
                    # silently rounded into the first automatic GPU size.
                    @test !RK45._gpu_native_eligible(transform_adk_large,
                                                     linop_adk_large, 8192)
                    @test RK45._gpu_native_eligible(transform_adk_large,
                                                    linop_adk_large, 8193)
                end
                withenv("AMALTHEA_NATIVE_GPU" => "auto") do
                    @test !RK45._gpu_native_eligible(transform_small, linop_small, length(Eω_small))
                    @test RK45._gpu_native_eligible(transform_large, linop_large, length(Eω_large))
                end
            end
        end
    end
end
