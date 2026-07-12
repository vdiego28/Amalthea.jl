using TestItems

@testitem "Native-Rust Phase I item 1 (mode-averaged shot noise, EnvGrid)" tags=[:rust] begin
    import Test: @test, @test_skip, @testset
    using Luna
    using Luna.RK45: PreconStepper, RustNativeStepper, step!, solve
    import Logging: with_logger, NullLogger
    import LinearAlgebra: norm
    import Random: MersenneTwister

    libpath = RK45._LIBLUNA_RUST_RK45
    if !isfile(libpath)
        @test_skip "Rust library not found"
    else
        # docs/dev/BACKLOG.md Phase I item 1: the ComplexF64 (EnvGrid) counterpart of
        # test_native_shotnoise.jl. This is also the config that determines
        # whether the DEFAULT prop_gnlse workload (envelope, mode-averaged,
        # shotnoise=true) actually runs natively — see docs/dev/BACKLOG.md's note on
        # this item.
        radius = 125e-6
        flength = 0.15
        gas = :He
        pressure = 1.0
        λ0 = 800e-9
        λlims = (200e-9, 4e-6)
        trange = 1e-12

        args = (radius, flength, gas, pressure)
        kw = (; λ0, λlims, trange, raman=false, plasma=false, kerr=true,
              shotnoise=true, energy=1e-6, τfwhm=30e-15, envelope=true)

        Eω, grid, linop, transform_jl, FT, output = with_logger(NullLogger()) do
            Interface.prop_capillary_args(args...; kw..., rng=MersenneTwister(42))
        end
        Eω2, grid2, linop2, transform_ru, FT2, output2 = with_logger(NullLogger()) do
            Interface.prop_capillary_args(args...; kw..., rng=MersenneTwister(42))
        end

        @assert grid isa Luna.Grid.EnvGrid "Expected EnvGrid for envelope=true"
        @test transform_jl.Et_noise == transform_ru.Et_noise
        @test !isnothing(transform_jl.Et_noise)

        t0 = 0.0
        dt = 0.01

        @testset "Single-step equivalence with noise (EnvGrid, ~1e-10)" begin
            s_jl = PreconStepper(transform_jl, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10)
            s_ru = RustNativeStepper(transform_ru, linop2, copy(Eω2), t0, dt, rtol=1e-6, atol=1e-10)

            ok_jl = step!(s_jl)
            ok_ru = step!(s_ru)

            @test ok_jl == ok_ru
            rel_step = norm(s_ru.yn - s_jl.yn) / norm(s_jl.yn)
            @test rel_step < 1e-10
        end

        @testset "Noise actually changes the native result (regression guard)" begin
            kw_nonoise = (; λ0, λlims, trange, raman=false, plasma=false, kerr=true,
                          shotnoise=false, energy=1e-6, τfwhm=30e-15, envelope=true)
            Eω3, grid3, linop3, transform_nonoise, FT3, output3 = with_logger(NullLogger()) do
                Interface.prop_capillary_args(args...; kw_nonoise...)
            end
            s_ru_noisy = RustNativeStepper(transform_ru, linop2, copy(Eω2), t0, dt, rtol=1e-6, atol=1e-10)
            s_ru_clean = RustNativeStepper(transform_nonoise, linop3, copy(Eω3), t0, dt, rtol=1e-6, atol=1e-10)
            step!(s_ru_noisy)
            step!(s_ru_clean)
            rel_diff = norm(s_ru_noisy.yn - s_ru_clean.yn) / norm(s_ru_clean.yn)
            println("EnvGrid mode-avg noisy-vs-clean native single-step rel_diff: ", rel_diff)
            @test rel_diff > 0.0
        end
    end
end

@testitem "Native-Rust Phase I item 1 (radial shot noise, EnvGrid)" tags=[:rust] begin
    import Test: @test, @test_skip, @testset
    using Luna
    import Luna: Grid, NonlinearRHS, Fields, LinearOps, PhysData, Nonlinear
    using Luna.RK45: PreconStepper, RustNativeStepper, step!, solve
    import Hankel
    import Logging: with_logger, NullLogger
    import LinearAlgebra: norm
    import Random: MersenneTwister

    libpath = RK45._LIBLUNA_RUST_RK45
    if !isfile(libpath)
        @test_skip "Rust library not found"
    else
        gas = :Ar; pres = 1.2; τ = 20e-15; λ0 = 800e-9
        w0 = 40e-6; energy = 1e-12; L = 0.05; R = 4e-3; N = 32

        grid = Grid.EnvGrid(L, λ0, (400e-9, 2000e-9), 0.2e-12)
        q    = Hankel.QDHT(R, N, dim=2)

        dens0 = PhysData.density(gas, pres)
        densityfun(z) = dens0
        responses = (Nonlinear.Kerr_env(PhysData.γ3_gas(gas)),)
        linop   = LinearOps.make_const_linop(grid, q, PhysData.ref_index_fun(gas, pres))
        normfun = NonlinearRHS.const_norm_radial(grid, q, PhysData.ref_index_fun(gas, pres))
        inputs  = Fields.GaussGaussField(λ0=λ0, τfwhm=τ, energy=energy, w0=w0, propz=-0.15)

        noise_field = Fields.generate_noise_field(grid; rng=MersenneTwister(7), nmodes=N)

        Eω, transform, FT = with_logger(NullLogger()) do
            Luna.setup(grid, q, densityfun, normfun, responses, inputs; noise_field)
        end

        @assert transform isa Luna.NonlinearRHS.TransRadial "Expected TransRadial"
        @test !isnothing(transform.Et_noise)

        t0 = 0.0
        dt = 0.001

        @testset "Single-step equivalence with noise (radial EnvGrid, ~1e-10)" begin
            s_jl = PreconStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10)
            s_ru = RustNativeStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10)

            step!(s_jl)
            step!(s_ru)

            rel_step = norm(s_ru.yn - s_jl.yn) / norm(s_jl.yn)
            println("Radial EnvGrid noisy single-step rel: ", rel_step)
            @test rel_step < 1e-10
        end

        @testset "Noise actually changes the native result (regression guard)" begin
            Eω_clean, transform_clean, FT_clean = with_logger(NullLogger()) do
                Luna.setup(grid, q, densityfun, normfun, responses, inputs)
            end
            s_ru_noisy = RustNativeStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10)
            s_ru_clean = RustNativeStepper(transform_clean, linop, copy(Eω_clean), t0, dt, rtol=1e-6, atol=1e-10)
            step!(s_ru_noisy)
            step!(s_ru_clean)
            rel_diff = norm(s_ru_noisy.yn - s_ru_clean.yn) / norm(s_ru_clean.yn)
            println("Radial EnvGrid noisy-vs-clean native single-step rel_diff: ", rel_diff)
            @test rel_diff > 0.0
        end
    end
end
