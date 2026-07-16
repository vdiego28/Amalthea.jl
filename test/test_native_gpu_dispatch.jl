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
        # Plasma present, large enough to clear the size threshold anyway —
        # :auto must still reject it (no measured crossover with plasma).
        kw_plasma = (; λ0, λlims, trange=4e-12, raman=false, plasma=true, kerr=true,
                       shotnoise=false, energy=1e-6, τfwhm=30e-15)

        Eω_small, _, linop_small, transform_small, _, _ = with_logger(NullLogger()) do
            Interface.prop_capillary_args(args...; kw_small...)
        end
        Eω_large, _, linop_large, transform_large, _, _ = with_logger(NullLogger()) do
            Interface.prop_capillary_args(args...; kw_large...)
        end
        Eω_plasma, _, linop_plasma, transform_plasma, _, _ = with_logger(NullLogger()) do
            Interface.prop_capillary_args(args...; kw_plasma...)
        end

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
                end
            end

            @testset "gpu_dispatch=auto (default) -> size- and plasma-gated" begin
                withenv("AMALTHEA_NATIVE_GPU" => nothing) do
                    @test Amalthea.Config.backend_config().gpu_dispatch === :auto
                    @test !RK45._gpu_native_eligible(transform_small, linop_small, length(Eω_small))
                    @test RK45._gpu_native_eligible(transform_large, linop_large, length(Eω_large))
                    # plasma-bearing config: rejected under :auto even though
                    # it clears the size threshold (no measured GPU win with
                    # plasma present, at any tested size).
                    @test !RK45._gpu_native_eligible(transform_plasma, linop_plasma, length(Eω_plasma))
                end
                withenv("AMALTHEA_NATIVE_GPU" => "auto") do
                    @test !RK45._gpu_native_eligible(transform_small, linop_small, length(Eω_small))
                    @test RK45._gpu_native_eligible(transform_large, linop_large, length(Eω_large))
                end
            end
        end
    end
end
