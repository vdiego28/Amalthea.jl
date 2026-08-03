using TestItems

@testitem "Native-Rust CUDA mode-averaged SDO Raman" tags=[:rust] begin
    import Test: @test, @test_skip, @testset
    using Amalthea
    import Amalthea: Grid, NonlinearRHS, Fields, LinearOps, PhysData, Nonlinear,
                     Capillary, Modes, Raman
    using Amalthea.RK45: PreconStepper, RustNativeStepper, step!, solve
    import LinearAlgebra: norm
    import Logging: with_logger, NullLogger

    libpath = RK45._LIBAMALTHEA_RK45
    require_cuda = get(ENV, "AMALTHEA_REQUIRE_CUDA_TESTS", "0") == "1"
    if !isfile(libpath)
        require_cuda && error("CUDA Raman tests are required, but the Rust library was not found")
        @test_skip "Rust library not found"
    else
        gas = :N2
        pressure = 1.0
        λ0 = 800e-9
        radius = 125e-6
        flength = 0.05
        τfwhm = 20e-15
        energy = 5e-6
        dt = 0.01

        function make_mode_avg(grid_kind::Symbol, thg::Bool=true; gas=:N2,
                               rotation=false, vibration=true)
            grid = grid_kind === :real ?
                Grid.RealGrid(flength, λ0, (400e-9, 2000e-9), 0.5e-12) :
                Grid.EnvGrid(flength, λ0, (400e-9, 2000e-9), 0.5e-12)
            mode = Capillary.MarcatiliMode(radius, gas, pressure; kind=:HE, n=1, m=1)
            density = PhysData.density(gas, pressure)
            densityfun(z) = density
            rr = Raman.raman_response(grid.to, gas;
                                      rotation=rotation, vibration=vibration)
            responses = if grid_kind === :real
                (Nonlinear.Kerr_field(PhysData.γ3_gas(gas)),
                 Nonlinear.RamanPolarField(grid.to, rr; thg=thg))
            else
                (Nonlinear.Kerr_env(PhysData.γ3_gas(gas)),
                 Nonlinear.RamanPolarEnv(grid.to, rr))
            end
            linop, βfun!, _, _ = LinearOps.make_const_linop(grid, mode, grid.referenceλ)
            aeff = z -> Modes.Aeff(mode, z=z)
            input = Fields.GaussField(λ0=λ0, τfwhm=τfwhm, energy=energy)
            Eω, transform, FT = with_logger(NullLogger()) do
                Amalthea.setup(grid, densityfun, responses, input, βfun!, aeff)
            end
            @assert transform isa NonlinearRHS.TransModeAvg
            Eω, grid, linop, transform, FT, densityfun, aeff
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
            input = Fields.GaussField(λ0=λ0, τfwhm=τfwhm, energy=energy)
            Eω, transform, _ = with_logger(NullLogger()) do
                Amalthea.setup(grid, densityfun,
                               (Nonlinear.Kerr_env(PhysData.γ3_gas(gas)),),
                               input, βfun!, aeff)
            end
            Eω, linop, transform
        end

        function get_ks(handle_ptr, idx, n)
            out = zeros(ComplexF64, n)
            rc = ccall((:get_ks_stage, libpath), Cint,
                       (Ptr{Cvoid}, Csize_t, Ptr{ComplexF64}, Csize_t),
                       handle_ptr, Csize_t(idx), out, Csize_t(n))
            rc == 0 || error("get_ks_stage failed with rc=$rc")
            out
        end

        function make_stepper(transform, linop, Eω; gpu::Bool, dt=0.01,
                              max_dt=dt, min_dt=dt)
            withenv("AMALTHEA_USE_RUST_CUDA_NATIVE" => gpu ? "1" : "0",
                    "AMALTHEA_NATIVE_GPU" => gpu ? "on" : "off") do
                RustNativeStepper(transform, linop, copy(Eω), 0.0, dt;
                                  rtol=1e-6, atol=1e-10, max_dt, min_dt)
            end
        end

        @testset "Pure Raman eligibility and CPU fallback" begin
            # These checks deliberately precede CUDA initialization. They must
            # count on a CPU-only host and prove the dispatch decision without
            # inferring it from the opaque stepper type.
            Eω_small, _, linop_small, transform_small, _, _, _ =
                make_mode_avg(:real, true; rotation=false, vibration=true)
            @test RK45._gpu_kernel_supports(transform_small, linop_small)
            withenv("AMALTHEA_USE_RUST_CUDA_NATIVE" => "1",
                    "AMALTHEA_NATIVE_GPU" => "on") do
                @test RK45._gpu_native_eligible(
                    transform_small, linop_small, length(Eω_small))
            end
            withenv("AMALTHEA_USE_RUST_CUDA_NATIVE" => "1",
                    "AMALTHEA_NATIVE_GPU" => "off") do
                s_cpu = RustNativeStepper(transform_small, linop_small,
                                           copy(Eω_small), 0.0, dt;
                                           rtol=1e-6, atol=1e-10,
                                           max_dt=dt, min_dt=dt)
                @test RK45._native_backend(s_cpu) === :cpu
            end
            withenv("AMALTHEA_USE_RUST_CUDA_NATIVE" => "1",
                    "AMALTHEA_NATIVE_GPU" => "auto") do
                @test !RK45._gpu_native_eligible(
                    transform_small, linop_small, length(Eω_small))
                s_auto = RustNativeStepper(transform_small, linop_small,
                                           copy(Eω_small), 0.0, dt;
                                           rtol=1e-6, atol=1e-10,
                                           max_dt=dt, min_dt=dt)
                @test RK45._native_backend(s_auto) === :cpu
            end

            Eω_small_no_thg, _, linop_small_no_thg, transform_small_no_thg, _, _, _ =
                make_mode_avg(:real, false; rotation=false, vibration=true)
            @test RK45._gpu_kernel_supports(transform_small_no_thg,
                                             linop_small_no_thg)
            withenv("AMALTHEA_USE_RUST_CUDA_NATIVE" => "1",
                    "AMALTHEA_NATIVE_GPU" => "auto") do
                @test !RK45._gpu_native_eligible(
                    transform_small_no_thg, linop_small_no_thg,
                    length(Eω_small_no_thg))
                s_auto_no_thg = RustNativeStepper(
                    transform_small_no_thg, linop_small_no_thg,
                    copy(Eω_small_no_thg), 0.0, dt;
                    rtol=1e-6, atol=1e-10, max_dt=dt, min_dt=dt)
                @test RK45._native_backend(s_auto_no_thg) === :cpu
            end

            for (rotation, vibration, expected) in ((true, false, 49),
                                                       (true, true, 50))
                Eω, _, linop, transform, _, _, _ =
                    make_mode_avg(:real, true;
                                  rotation=rotation, vibration=vibration)
                Rs = Raman.flatten_sdo_oscillators(transform.resp[2].r)
                @test length(Rs) == expected
                @test RK45._gpu_kernel_supports(transform, linop)
                withenv("AMALTHEA_USE_RUST_CUDA_NATIVE" => "1",
                        "AMALTHEA_NATIVE_GPU" => "on") do
                    @test RK45._gpu_native_eligible(transform, linop, length(Eω))
                end
                s_cpu = make_stepper(transform, linop, Eω; gpu=false)
                @test RK45._native_backend(s_cpu) === :cpu
                withenv("AMALTHEA_USE_RUST_CUDA_NATIVE" => "1",
                        "AMALTHEA_NATIVE_GPU" => "auto") do
                    @test !RK45._gpu_native_eligible(transform, linop, length(Eω))
                    s_auto = RustNativeStepper(transform, linop, copy(Eω), 0.0, dt;
                                               rtol=1e-6, atol=1e-10,
                                               max_dt=dt, min_dt=dt)
                    @test RK45._native_backend(s_auto) === :cpu
                end
            end

            Eω_env, _, linop_env, transform_env, _, _, _ = make_mode_avg(:env)
            @test RK45._gpu_kernel_supports(transform_env, linop_env)
            withenv("AMALTHEA_USE_RUST_CUDA_NATIVE" => "1",
                    "AMALTHEA_NATIVE_GPU" => "on") do
                @test RK45._gpu_native_eligible(transform_env, linop_env,
                                                 length(Eω_env))
            end
            s_env_cpu = make_stepper(transform_env, linop_env, Eω_env; gpu=false)
            @test RK45._native_backend(s_env_cpu) === :cpu
            withenv("AMALTHEA_USE_RUST_CUDA_NATIVE" => "1",
                    "AMALTHEA_NATIVE_GPU" => "auto") do
                @test !RK45._gpu_native_eligible(
                    transform_env, linop_env, length(Eω_env))
                s_env_auto = RustNativeStepper(transform_env, linop_env,
                                               copy(Eω_env), 0.0, dt;
                                               rtol=1e-6, atol=1e-10,
                                               max_dt=dt, min_dt=dt)
                @test RK45._native_backend(s_env_auto) === :cpu
            end

            # :SiO2 uses the intermediate-broadening response and is supported
            # only by the EnvGrid resident FFT-convolution CUDA path. It is
            # eligible under explicit `:on`, while the measured Raman `:auto`
            # policy remains deliberately disabled.
            grid = Grid.EnvGrid(flength, λ0, (400e-9, 2000e-9), 0.5e-12)
            mode = Capillary.MarcatiliMode(radius, gas, pressure;
                                           kind=:HE, n=1, m=1)
            density = PhysData.density(gas, pressure)
            densityfun(z) = density
            rr = Raman.raman_response(
                grid.to, :SiO2,
                2 * 0.18 * PhysData.ε_0 * PhysData.γ3_gas(gas))
            responses = (Nonlinear.Kerr_env(PhysData.γ3_gas(gas)),
                         Nonlinear.RamanPolarEnv(grid.to, rr))
            linop, βfun!, _, _ = LinearOps.make_const_linop(grid, mode, grid.referenceλ)
            aeff = z -> Modes.Aeff(mode, z=z)
            input = Fields.GaussField(λ0=λ0, τfwhm=τfwhm, energy=energy)
            Eω, transform, _ = with_logger(NullLogger()) do
                Amalthea.setup(grid, densityfun, responses, input, βfun!, aeff)
            end
            @test RK45._gpu_kernel_supports(transform, linop)
            withenv("AMALTHEA_USE_RUST_CUDA_NATIVE" => "1",
                    "AMALTHEA_NATIVE_GPU" => "on") do
                @test RK45._gpu_native_eligible(transform, linop, length(Eω))
            end
            withenv("AMALTHEA_USE_RUST_CUDA_NATIVE" => "1",
                    "AMALTHEA_NATIVE_GPU" => "auto") do
                @test !RK45._gpu_native_eligible(transform, linop, length(Eω))
                s_auto = RustNativeStepper(transform, linop, copy(Eω), 0.0, dt;
                                           rtol=1e-6, atol=1e-10,
                                           max_dt=dt, min_dt=dt)
                @test RK45._native_backend(s_auto) === :cpu
            end
            withenv("AMALTHEA_USE_RUST_CUDA_NATIVE" => "0",
                    "AMALTHEA_NATIVE_GPU" => "on") do
                s_cpu = RustNativeStepper(transform, linop, copy(Eω), 0.0, dt;
                                           rtol=1e-6, atol=1e-10,
                                           max_dt=dt, min_dt=dt)
                @test RK45._native_backend(s_cpu) === :cpu
            end
        end

        # Construction is the hardware gate. The test suite runs the dispatch
        # checks above everywhere, but only a machine with a live CUDA driver
        # can exercise the resident kernels.
        Eω_probe, _, linop_probe, transform_probe, _, _, _ =
            make_mode_avg(:real, true)
        gpu_error = try
            make_stepper(transform_probe, linop_probe, Eω_probe; gpu=true)
            nothing
        catch err
            err
        end
        if gpu_error !== nothing
            require_cuda && throw(gpu_error)
            @test_skip "CUDA device unavailable: $(sprint(showerror, gpu_error))"
        else
            @test RK45._native_backend(make_stepper(
                transform_probe, linop_probe, Eω_probe; gpu=true)) === :cuda
            @testset "EnvGrid Kerr :auto at measured threshold" begin
                Eω_env, linop_env, transform_env = make_env_kerr_policy(32e-12)
                @test length(Eω_env) == RK45._GPU_ENV_KERR_N_THRESHOLD
                @test RK45._gpu_kernel_supports(transform_env, linop_env)
                withenv("AMALTHEA_USE_RUST_CUDA_NATIVE" => "1",
                        "AMALTHEA_NATIVE_GPU" => "auto") do
                    @test RK45._gpu_native_eligible(
                        transform_env, linop_env, length(Eω_env))
                    s_env_auto = RustNativeStepper(
                        transform_env, linop_env, copy(Eω_env), 0.0, dt;
                        rtol=1e-6, atol=1e-10, max_dt=dt, min_dt=dt)
                    @test RK45._native_backend(s_env_auto) === :cuda
                end
            end
            @testset "RealGrid RamanPolarField (thg=true/false)" begin
                for thg in (true, false)
                    @testset "thg=$thg" begin
                        Eω, _, linop, transform, _, _, _ = make_mode_avg(:real, thg)
                        @test RK45._gpu_kernel_supports(transform, linop)
                        withenv("AMALTHEA_USE_RUST_CUDA_NATIVE" => "1",
                                "AMALTHEA_NATIVE_GPU" => "on") do
                            @test RK45._gpu_native_eligible(transform, linop, length(Eω))
                        end

                        s_cpu = make_stepper(transform, linop, Eω; gpu=false)
                        s_gpu = make_stepper(transform, linop, Eω; gpu=true)
                        @test RK45._native_backend(s_cpu) === :cpu
                        @test RK45._native_backend(s_gpu) === :cuda
                        n = length(Eω)
                        k_cpu0 = get_ks(s_cpu._handle.ptr, 0, n)
                        k_gpu0 = get_ks(s_gpu._handle.ptr, 0, n)
                        @test maximum(abs.(k_cpu0)) > 1e-8
                        @test norm(k_gpu0 - k_cpu0) / norm(k_cpu0) < 1e-9

                        @test step!(s_cpu)
                        @test step!(s_gpu)
                        for idx in 0:6
                            kc = get_ks(s_cpu._handle.ptr, idx, n)
                            kg = get_ks(s_gpu._handle.ptr, idx, n)
                            @test norm(kg - kc) / norm(kc) < 1e-9
                        end

                        # A Julia-on/Juila-off comparison proves that Raman is
                        # materially present; CPU-vs-GPU agreement alone could
                        # otherwise compare two equally empty backends.
                        _, grid, _, _, FT, densityfun, aeff = make_mode_avg(:real, thg)
                        resp_off = (Nonlinear.Kerr_field(PhysData.γ3_gas(gas)),)
                        transform_off = NonlinearRHS.TransModeAvg(
                            grid, transform.FT, resp_off, densityfun, transform.norm!, aeff)
                        s_on_jl = PreconStepper(transform, linop, copy(Eω), 0.0, dt;
                                                rtol=1e-6, atol=1e-10, max_dt=dt, min_dt=dt)
                        s_off_jl = PreconStepper(transform_off, linop, copy(Eω), 0.0, dt;
                                                 rtol=1e-6, atol=1e-10, max_dt=dt, min_dt=dt)
                        solve(s_on_jl, flength)
                        solve(s_off_jl, flength)
                        rel_raman = norm(s_on_jl.yn - s_off_jl.yn) / norm(s_on_jl.yn)
                        println("RealGrid thg=$thg Raman-on/off rel: ", rel_raman)
                        @test rel_raman > 1e-8

                        s_cpu_fixed = make_stepper(transform, linop, Eω; gpu=false)
                        s_gpu_fixed = make_stepper(transform, linop, Eω; gpu=true)
                        solve(s_cpu_fixed, flength)
                        solve(s_gpu_fixed, flength)
                        rel_fixed = norm(s_gpu_fixed.yn - s_cpu_fixed.yn) /
                                    norm(s_cpu_fixed.yn)
                        println("RealGrid thg=$thg GPU/CPU fixed-solve rel: ", rel_fixed)
                        @test rel_fixed < 1e-7
                    end
                end
            end

            @testset "N2 rotational Raman capacity (49/50 oscillators)" begin
                for (rotation, vibration, expected) in ((true, false, 49),
                                                           (true, true, 50))
                    @testset "rotation=$rotation vibration=$vibration" begin
                        Eω, _, linop, transform, _, _, _ =
                            make_mode_avg(:real, true;
                                          rotation=rotation, vibration=vibration)
                        Rs = Raman.flatten_sdo_oscillators(transform.resp[2].r)
                        @test length(Rs) == expected
                        @test length(Rs) <= RK45._GPU_RAMAN_MAX_OSCILLATORS
                        @test RK45._gpu_kernel_supports(transform, linop)

                        s_cpu = make_stepper(transform, linop, Eω; gpu=false)
                        s_gpu = make_stepper(transform, linop, Eω; gpu=true)
                        @test RK45._native_backend(s_cpu) === :cpu
                        @test RK45._native_backend(s_gpu) === :cuda
                        n = length(Eω)
                        k_cpu0 = get_ks(s_cpu._handle.ptr, 0, n)
                        k_gpu0 = get_ks(s_gpu._handle.ptr, 0, n)
                        @test maximum(abs.(k_cpu0)) > 1e-8
                        @test norm(k_gpu0 - k_cpu0) / norm(k_cpu0) < 1e-9

                        s_cpu_fixed = make_stepper(transform, linop, Eω; gpu=false)
                        s_gpu_fixed = make_stepper(transform, linop, Eω; gpu=true)
                        solve(s_cpu_fixed, flength)
                        solve(s_gpu_fixed, flength)
                        rel_fixed = norm(s_gpu_fixed.yn - s_cpu_fixed.yn) /
                                    norm(s_cpu_fixed.yn)
                        println("N2 rotation=$rotation vibration=$vibration " *
                                " GPU/CPU fixed-solve rel: ", rel_fixed)
                        @test rel_fixed < 1e-7

                        transform_off = NonlinearRHS.TransModeAvg(
                            transform.grid, transform.FT,
                            (Nonlinear.Kerr_field(PhysData.γ3_gas(gas)),),
                            transform.densityfun, transform.norm!, transform.aeff)
                        s_on_jl = PreconStepper(transform, linop, copy(Eω), 0.0, dt;
                                                rtol=1e-6, atol=1e-10,
                                                max_dt=dt, min_dt=dt)
                        s_off_jl = PreconStepper(transform_off, linop, copy(Eω), 0.0, dt;
                                                 rtol=1e-6, atol=1e-10,
                                                 max_dt=dt, min_dt=dt)
                        solve(s_on_jl, flength)
                        solve(s_off_jl, flength)
                        rel_raman = norm(s_on_jl.yn - s_off_jl.yn) /
                                    norm(s_on_jl.yn)
                        println("N2 rotation=$rotation vibration=$vibration " *
                                " Raman-on/off rel: ", rel_raman)
                        @test rel_raman > 1e-5
                    end
                end
            end

            @testset "RealGrid adaptive rejection and retry" begin
                Eω, _, linop, transform, _, _, _ =
                    make_mode_avg(:real, true; rotation=true, vibration=true)
                dt_reject = 1.0
                s_cpu = make_stepper(transform, linop, Eω; gpu=false,
                                     dt=dt_reject, max_dt=2.0, min_dt=0.0)
                s_gpu = make_stepper(transform, linop, Eω; gpu=true,
                                     dt=dt_reject, max_dt=2.0, min_dt=0.0)
                @test RK45._native_backend(s_cpu) === :cpu
                @test RK45._native_backend(s_gpu) === :cuda
                field_before = copy(s_gpu.yn)
                @test !step!(s_cpu)
                @test !step!(s_gpu)
                @test s_gpu.yn == field_before
                @test s_gpu.tn == 0.0
                @test s_gpu.err > 1
                @test isapprox(s_gpu.err, s_cpu.err; rtol=1e-8)
                @test isapprox(s_gpu.dtn, s_cpu.dtn; rtol=1e-8)

                accepted = false
                for _ in 1:8
                    cpu_ok = step!(s_cpu)
                    gpu_ok = step!(s_gpu)
                    @test cpu_ok == gpu_ok
                    @test isapprox(s_gpu.err, s_cpu.err; rtol=1e-7)
                    if cpu_ok
                        accepted = true
                        break
                    end
                end
                @test accepted
                @test norm(s_gpu.yn - s_cpu.yn) / norm(s_cpu.yn) < 1e-7
                solve(s_cpu, flength)
                solve(s_gpu, flength)
                @test norm(s_gpu.yn - s_cpu.yn) / norm(s_cpu.yn) < 1e-6
            end

            @testset "EnvGrid RamanPolarEnv" begin
                Eω, grid, linop, transform, FT, densityfun, aeff = make_mode_avg(:env)
                @test RK45._gpu_kernel_supports(transform, linop)
                withenv("AMALTHEA_USE_RUST_CUDA_NATIVE" => "1",
                        "AMALTHEA_NATIVE_GPU" => "on") do
                    @test RK45._gpu_native_eligible(transform, linop, length(Eω))
                end
                s_cpu = make_stepper(transform, linop, Eω; gpu=false)
                s_gpu = make_stepper(transform, linop, Eω; gpu=true)
                @test RK45._native_backend(s_cpu) === :cpu
                @test RK45._native_backend(s_gpu) === :cuda
                n = length(Eω)
                k_cpu0 = get_ks(s_cpu._handle.ptr, 0, n)
                k_gpu0 = get_ks(s_gpu._handle.ptr, 0, n)
                @test maximum(abs.(k_cpu0)) > 1e-8
                @test norm(k_gpu0 - k_cpu0) / norm(k_cpu0) < 1e-9
                @test step!(s_cpu)
                @test step!(s_gpu)
                @test norm(s_gpu.yn - s_cpu.yn) / norm(s_cpu.yn) < 1e-8

                resp_off = (Nonlinear.Kerr_env(PhysData.γ3_gas(gas)),)
                transform_off = NonlinearRHS.TransModeAvg(
                    grid, transform.FT, resp_off, densityfun, transform.norm!, aeff)
                s_on_jl = PreconStepper(transform, linop, copy(Eω), 0.0, dt;
                                        rtol=1e-6, atol=1e-10, max_dt=dt, min_dt=dt)
                s_off_jl = PreconStepper(transform_off, linop, copy(Eω), 0.0, dt;
                                         rtol=1e-6, atol=1e-10, max_dt=dt, min_dt=dt)
                solve(s_on_jl, flength)
                solve(s_off_jl, flength)
                rel_raman = norm(s_on_jl.yn - s_off_jl.yn) / norm(s_on_jl.yn)
                @test rel_raman > 1e-8

                s_cpu_fixed = make_stepper(transform, linop, Eω; gpu=false)
                s_gpu_fixed = make_stepper(transform, linop, Eω; gpu=true)
                solve(s_cpu_fixed, flength)
                solve(s_gpu_fixed, flength)
                rel_fixed = norm(s_gpu_fixed.yn - s_cpu_fixed.yn) /
                            norm(s_cpu_fixed.yn)
                println("EnvGrid Raman GPU/CPU fixed-solve rel: ", rel_fixed)
                @test rel_fixed < 1e-7
            end

            @testset "EnvGrid :SiO2 Raman FFT convolution" begin
                Eω, grid, linop, transform, _, densityfun, aeff = begin
                    grid = Grid.EnvGrid(flength, λ0, (400e-9, 2000e-9), 0.5e-12)
                    mode = Capillary.MarcatiliMode(radius, gas, pressure;
                                                    kind=:HE, n=1, m=1)
                    density = PhysData.density(gas, pressure)
                    densityfun(z) = density
                    # Match the physical per-molecule scaling used by the
                    # capillary Kerr response; density is supplied separately
                    # to the nonlinear transform.
                    rr = Raman.raman_response(
                        grid.to, :SiO2,
                        2 * 0.18 * PhysData.ε_0 * PhysData.γ3_gas(gas))
                    responses = (Nonlinear.Kerr_env(PhysData.γ3_gas(gas)),
                                 Nonlinear.RamanPolarEnv(grid.to, rr))
                    linop, βfun!, _, _ = LinearOps.make_const_linop(
                        grid, mode, grid.referenceλ)
                    aeff = z -> Modes.Aeff(mode, z=z)
                    input = Fields.GaussField(λ0=λ0, τfwhm=τfwhm, energy=energy)
                    Eω, transform, _ = with_logger(NullLogger()) do
                        Amalthea.setup(grid, densityfun, responses, input,
                                       βfun!, aeff)
                    end
                    Eω, grid, linop, transform, nothing, densityfun, aeff
                end
                # Keep the hardware trajectory in the stable regime while
                # retaining the production-sized spectral grid and response.
                # The dedicated CPU triangulation test exercises the full
                # high-power prop_gnlse workload.
                Eω .*= 1e-2
                @test grid isa Grid.EnvGrid
                @test transform.resp[2].r isa Raman.RamanRespIntermediateBroadening
                @test RK45._gpu_kernel_supports(transform, linop)
                withenv("AMALTHEA_USE_RUST_CUDA_NATIVE" => "1",
                        "AMALTHEA_NATIVE_GPU" => "on") do
                    @test RK45._gpu_native_eligible(transform, linop, length(Eω))
                end

                s_cpu = make_stepper(transform, linop, Eω; gpu=false)
                s_gpu = make_stepper(transform, linop, Eω; gpu=true)
                @test RK45._native_backend(s_cpu) === :cpu
                @test RK45._native_backend(s_gpu) === :cuda
                n = length(Eω)
                k_cpu0 = get_ks(s_cpu._handle.ptr, 0, n)
                k_gpu0 = get_ks(s_gpu._handle.ptr, 0, n)
                @test maximum(abs.(k_cpu0)) > 1e-8
                rel_stage = norm(k_gpu0 - k_cpu0) / norm(k_cpu0)
                println("EnvGrid :SiO2 Raman GPU/CPU stage rel: ", rel_stage)
                @test rel_stage < 1e-9

                dt_fixed = dt
                s_cpu_fixed = make_stepper(transform, linop, Eω; gpu=false,
                                           dt=dt_fixed)
                s_gpu_fixed = make_stepper(transform, linop, Eω; gpu=true,
                                           dt=dt_fixed)
                for _ in 1:6
                    cpu_ok = step!(s_cpu_fixed)
                    gpu_ok = step!(s_gpu_fixed)
                    @test cpu_ok == gpu_ok
                    @test cpu_ok
                    @test all(isfinite, s_cpu_fixed.yn)
                    @test all(isfinite, s_gpu_fixed.yn)
                end
                rel_fixed = norm(s_gpu_fixed.yn - s_cpu_fixed.yn) /
                            norm(s_cpu_fixed.yn)
                println("EnvGrid :SiO2 Raman GPU/CPU fixed-solve rel: ", rel_fixed,
                        " cpu_finite=", all(isfinite, s_cpu_fixed.yn),
                        " gpu_finite=", all(isfinite, s_gpu_fixed.yn),
                        " cpu_norm=", norm(s_cpu_fixed.yn),
                        " gpu_norm=", norm(s_gpu_fixed.yn))
                @test rel_fixed < 1e-7

                dt_reject = 100.0
                s_cpu_adapt = make_stepper(transform, linop, Eω; gpu=false,
                                           dt=dt_reject, max_dt=200.0, min_dt=0.0)
                s_gpu_adapt = make_stepper(transform, linop, Eω; gpu=true,
                                           dt=dt_reject, max_dt=200.0, min_dt=0.0)
                field_before = copy(s_gpu_adapt.yn)
                @test !step!(s_cpu_adapt)
                @test !step!(s_gpu_adapt)
                @test s_gpu_adapt.yn == field_before
                @test s_gpu_adapt.tn == 0.0
                @test s_gpu_adapt.err > 1
                @test isapprox(s_gpu_adapt.err, s_cpu_adapt.err; rtol=1e-8)
                @test isapprox(s_gpu_adapt.dtn, s_cpu_adapt.dtn; rtol=1e-8)
                accepted = false
                for _ in 1:8
                    cpu_ok = step!(s_cpu_adapt)
                    gpu_ok = step!(s_gpu_adapt)
                    @test gpu_ok == cpu_ok
                    @test isapprox(s_gpu_adapt.err, s_cpu_adapt.err; rtol=1e-7)
                    if cpu_ok
                        accepted = true
                        break
                    end
                end
                @test accepted
                @test norm(s_gpu_adapt.yn - s_cpu_adapt.yn) /
                      norm(s_cpu_adapt.yn) < 1e-7
                solve(s_cpu_adapt, flength)
                solve(s_gpu_adapt, flength)
                @test norm(s_gpu_adapt.yn - s_cpu_adapt.yn) /
                      norm(s_cpu_adapt.yn) < 1e-6
            end

        end
    end
end
