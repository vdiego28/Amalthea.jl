using TestItems

@testitem "Native-Rust GPU dispatch policy (:off/:on/:auto)" tags=[:rust] begin
    import Test: @test, @test_skip, @testset
    using Amalthea
    using Amalthea.RK45: RustNativeStepper, step!
    import LinearAlgebra: norm
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

        # Interface intentionally has no envelope-plasma entry point, but the
        # low-level API can construct one. CudaNativeSim's EnvGrid RHS computes
        # Kerr and Raman only, so this exact shape must be rejected by the
        # pure config guard rather than selecting a backend that drops plasma.
        grid_env = Grid.EnvGrid(flength, λ0, (400e-9, 2000e-9), 0.5e-12)
        mode_env = Capillary.MarcatiliMode(radius, gas, pressure;
                                           kind=:HE, n=1, m=1)
        density_env = PhysData.density(gas, pressure)
        densityfun_env(z) = density_env
        linop_env, βfun_env!, _, _ =
            LinearOps.make_const_linop(grid_env, mode_env, grid_env.referenceλ)
        aeff_env = z -> Modes.Aeff(mode_env, z=z)
        input_env = Fields.GaussField(λ0=λ0, τfwhm=30e-15, energy=1e-6)
        Eω_env, transform_env, _ = with_logger(NullLogger()) do
            Amalthea.setup(grid_env, densityfun_env,
                           (Nonlinear.Kerr_env(PhysData.γ3_gas(gas)),),
                           input_env, βfun_env!, aeff_env)
        end

        function make_env_kerr_policy(trange)
            grid = Grid.EnvGrid(flength, λ0, (400e-9, 2000e-9), trange)
            mode = Capillary.MarcatiliMode(radius, gas, pressure;
                                           kind=:HE, n=1, m=1)
            density = PhysData.density(gas, pressure)
            densityfun(z) = density
            linop, βfun!, _, _ = LinearOps.make_const_linop(
                grid, mode, grid.referenceλ)
            aeff = z -> Modes.Aeff(mode, z=z)
            input = Fields.GaussField(λ0=λ0, τfwhm=30e-15, energy=1e-6)
            Eω, transform, _ = with_logger(NullLogger()) do
                Amalthea.setup(grid, densityfun,
                               (Nonlinear.Kerr_env(PhysData.γ3_gas(gas)),),
                               input, βfun!, aeff)
            end
            Eω, linop, transform
        end
        # The EnvGrid sweep retained 32768. Keep the 8192 point below-threshold
        # case in pure CPU-only coverage so this test never needs a CUDA device.
        Eω_env_auto_below, linop_env_auto_below, transform_env_auto_below =
            make_env_kerr_policy(8e-12)
        Eω_env_auto_at, linop_env_auto_at, transform_env_auto_at =
            make_env_kerr_policy(32e-12)
        adk_env = Ionisation.IonRateADK(
            PhysData.ionisation_potential(gas); threshold=true)
        plasma_env = Nonlinear.PlasmaCumtrapz(
            transform_env.grid.to, transform_env.grid.to,
            adk_env, PhysData.ionisation_potential(gas))
        transform_env_plasma = NonlinearRHS.TransModeAvg(
            transform_env.Pto, transform_env.Eto,
            transform_env.Eωo, transform_env.Pωo,
            transform_env.FT,
            (transform_env.resp[1], plasma_env),
            transform_env.grid, transform_env.densityfun,
            transform_env.norm!, transform_env.aeff,
            transform_env.Et_noise, transform_env.Et_nl)

        function make_raman_capacity_transform(n_osc)
            sdos = [Amalthea.Raman.RamanRespNormedSingleDampedOscillator(
                        1e-4, (1.0 + i) * 1e13, 1e-12) for i in 1:n_osc]
            rr = Amalthea.Raman.CombinedRamanResponse(transform_env.grid.to, sdos)
            NonlinearRHS.TransModeAvg(
                transform_env.Pto, transform_env.Eto,
                transform_env.Eωo, transform_env.Pωo,
                transform_env.FT,
                (transform_env.resp[1],
                 Nonlinear.RamanPolarEnv(transform_env.grid.to, rr)),
                transform_env.grid, transform_env.densityfun,
                transform_env.norm!, transform_env.aeff,
                transform_env.Et_noise, transform_env.Et_nl)
        end
        transform_raman_at_capacity =
            make_raman_capacity_transform(RK45._GPU_RAMAN_MAX_OSCILLATORS)
        transform_raman_over_capacity =
            make_raman_capacity_transform(RK45._GPU_RAMAN_MAX_OSCILLATORS + 1)

        @test length(Eω_plasma_small) < RK45._GPU_PPT_N_THRESHOLD
        @test length(Eω_plasma_large) >= RK45._GPU_PPT_N_THRESHOLD
        @test length(Eω_adk_small) < RK45._GPU_ADK_N_THRESHOLD
        @test length(Eω_adk_large) >= RK45._GPU_ADK_N_THRESHOLD
        @test RK45._GPU_ADK_N_THRESHOLD == 8193
        @test !transform_adk_unthresholded.resp[2].ratefunc.threshold

        @test length(Eω_small) < RK45._GPU_KERR_ONLY_N_THRESHOLD
        @test length(Eω_large) >= RK45._GPU_KERR_ONLY_N_THRESHOLD

        @testset "EnvGrid Kerr has a separate measured :auto threshold" begin
            @test length(Eω_env_auto_below) == 8192
            @test length(Eω_env_auto_at) == RK45._GPU_ENV_KERR_N_THRESHOLD == 32768
            @test RK45._gpu_kernel_supports(
                transform_env_auto_below, linop_env_auto_below)
            @test RK45._gpu_kernel_supports(
                transform_env_auto_at, linop_env_auto_at)
            withenv("AMALTHEA_USE_RUST_CUDA_NATIVE" => "1",
                    "AMALTHEA_NATIVE_GPU" => "auto") do
                @test !RK45._gpu_native_eligible(
                    transform_env_auto_below, linop_env_auto_below,
                    length(Eω_env_auto_below))
                @test RK45._gpu_native_eligible(
                    transform_env_auto_at, linop_env_auto_at,
                    length(Eω_env_auto_at))
                s_env_auto_below = RustNativeStepper(
                    transform_env_auto_below, linop_env_auto_below,
                    copy(Eω_env_auto_below), 0.0, 0.005;
                    rtol=1e-6, atol=1e-10, max_dt=0.005, min_dt=0.005)
                @test RK45._native_backend(s_env_auto_below) === :cpu
            end
        end

        @testset "backend kind is observable on CPU-only constructor paths" begin
            # A supported small config proves the three policy decisions without
            # attempting CUDA construction on a host that may not have a driver:
            # :off and below-threshold :auto construct CPU, while :on is checked
            # only through the pure eligibility predicate.
            withenv("AMALTHEA_USE_RUST_CUDA_NATIVE" => "1",
                    "AMALTHEA_NATIVE_GPU" => "off") do
                s = RustNativeStepper(transform_small, linop_small,
                                      copy(Eω_small), 0.0, 0.005;
                                      rtol=1e-6, atol=1e-10,
                                      max_dt=0.005, min_dt=0.005)
                @test RK45._native_backend(s) === :cpu
            end
            withenv("AMALTHEA_USE_RUST_CUDA_NATIVE" => "1",
                    "AMALTHEA_NATIVE_GPU" => nothing) do
                @test Amalthea.Config.backend_config().gpu_dispatch === :auto
                s = RustNativeStepper(transform_small, linop_small,
                                      copy(Eω_small), 0.0, 0.005;
                                      rtol=1e-6, atol=1e-10,
                                      max_dt=0.005, min_dt=0.005)
                @test RK45._native_backend(s) === :cpu
            end
            withenv("AMALTHEA_USE_RUST_CUDA_NATIVE" => "1",
                    "AMALTHEA_NATIVE_GPU" => "on") do
                @test RK45._gpu_native_eligible(
                    transform_small, linop_small, length(Eω_small))
            end

            # The unsupported ADK threshold=false response must stay CPU even
            # when :on is requested; this is the observability regression that
            # replaces the old type-only assertion.
            withenv("AMALTHEA_USE_RUST_CUDA_NATIVE" => "1",
                    "AMALTHEA_NATIVE_GPU" => "on") do
                @test !RK45._gpu_native_eligible(
                    transform_adk_unthresholded, linop_adk_large,
                    length(Eω_adk_large))
                s = RustNativeStepper(transform_adk_unthresholded,
                                      linop_adk_large, copy(Eω_adk_large),
                                      0.0, 0.005;
                                      rtol=1e-6, atol=1e-10,
                                      max_dt=0.005, min_dt=0.005)
                @test RK45._native_backend(s) === :cpu
            end
        end

        @testset "Raman oscillator capacity boundary" begin
            @test length(Amalthea.Raman.flatten_sdo_oscillators(
                transform_raman_at_capacity.resp[2].r)) ==
                RK45._GPU_RAMAN_MAX_OSCILLATORS
            @test length(Amalthea.Raman.flatten_sdo_oscillators(
                transform_raman_over_capacity.resp[2].r)) ==
                RK45._GPU_RAMAN_MAX_OSCILLATORS + 1
            @test RK45._gpu_kernel_supports(transform_raman_at_capacity, linop_env)
            @test !RK45._gpu_kernel_supports(transform_raman_over_capacity, linop_env)
            withenv("AMALTHEA_USE_RUST_CUDA_NATIVE" => "1",
                    "AMALTHEA_NATIVE_GPU" => "on") do
                @test RK45._gpu_native_eligible(
                    transform_raman_at_capacity, linop_env, length(Eω_env))
                @test !RK45._gpu_native_eligible(
                    transform_raman_over_capacity, linop_env, length(Eω_env))
            end
        end

        @testset "Raman :auto remains an explicit/manual policy" begin
            # Plan 05 measured every supported Raman pipeline below the 1.4x
            # retention bar. Keep the policy names explicit so Raman cannot
            # accidentally inherit either Kerr threshold in a future edit.
            @test isnothing(RK45._GPU_RAMAN_REAL_THG_TRUE_N_THRESHOLD)
            @test isnothing(RK45._GPU_RAMAN_REAL_THG_FALSE_N_THRESHOLD)
            @test isnothing(RK45._GPU_RAMAN_ENV_N_THRESHOLD)
            @test isnothing(RK45._GPU_RAMAN_ROTATIONAL_N_THRESHOLD)
            withenv("AMALTHEA_USE_RUST_CUDA_NATIVE" => "1",
                    "AMALTHEA_NATIVE_GPU" => "auto") do
                @test !RK45._gpu_native_eligible(
                    transform_raman_at_capacity, linop_env, length(Eω_env))
                s_raman_auto = RustNativeStepper(
                    transform_raman_at_capacity, linop_env, copy(Eω_env),
                    0.0, 0.005; rtol=1e-6, atol=1e-10,
                    max_dt=0.005, min_dt=0.005)
                @test RK45._native_backend(s_raman_auto) === :cpu
                @test !RK45._gpu_native_eligible(
                    transform_raman_over_capacity, linop_env, length(Eω_env))
            end
        end

        @testset "EnvGrid plasma is an explicit CPU fallback" begin
            @test transform_env_plasma.grid isa Grid.EnvGrid
            @test !RK45._gpu_kernel_supports(transform_env_plasma, linop_env)
            withenv("AMALTHEA_USE_RUST_CUDA_NATIVE" => "1",
                    "AMALTHEA_NATIVE_GPU" => "on") do
                @test !RK45._gpu_native_eligible(
                    transform_env_plasma, linop_env, length(Eω_env))
            end

            dt_env = 0.005
            s_fallback = withenv("AMALTHEA_USE_RUST_CUDA_NATIVE" => "1",
                                 "AMALTHEA_NATIVE_GPU" => "on") do
                RustNativeStepper(transform_env_plasma, linop_env,
                                  copy(Eω_env), 0.0, dt_env;
                                  rtol=1e-6, atol=1e-10,
                                  max_dt=dt_env, min_dt=dt_env)
            end
            s_cpu = withenv("AMALTHEA_USE_RUST_CUDA_NATIVE" => "1",
                            "AMALTHEA_NATIVE_GPU" => "off") do
                RustNativeStepper(transform_env_plasma, linop_env,
                                  copy(Eω_env), 0.0, dt_env;
                                  rtol=1e-6, atol=1e-10,
                                  max_dt=dt_env, min_dt=dt_env)
            end
            step!(s_fallback)
            step!(s_cpu)
            @test RK45._native_backend(s_fallback) === :cpu
            @test RK45._native_backend(s_cpu) === :cpu
            @test all(isfinite, s_fallback.yn)
            rel_env_fallback = norm(s_fallback.yn - s_cpu.yn) / norm(s_cpu.yn)
            println("EnvGrid plasma forced-on CPU-fallback rel: ", rel_env_fallback)
            @test rel_env_fallback < 1e-13
        end

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
                    @test RK45._native_backend(s_cpu_fallback) === :cpu
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
