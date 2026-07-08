using TestItems

@testitem "Native-Rust Phase I item 1 (radial shot noise, RealGrid)" tags=[:rust] begin
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

        grid = Grid.RealGrid(L, λ0, (400e-9, 2000e-9), 0.2e-12)
        q    = Hankel.QDHT(R, N, dim=2)

        dens0 = PhysData.density(gas, pres)
        densityfun(z) = dens0
        responses = (Nonlinear.Kerr_field(PhysData.γ3_gas(gas)),)
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

        @testset "Single-step equivalence with noise (radial, ~1e-10)" begin
            s_jl = PreconStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10)
            s_ru = RustNativeStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10)

            step!(s_jl)
            step!(s_ru)

            rel_step = norm(s_ru.yn - s_jl.yn) / norm(s_jl.yn)
            println("Radial noisy single-step rel: ", rel_step)
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
            println("Radial noisy-vs-clean native single-step rel_diff: ", rel_diff)
            @test rel_diff > 0.0
        end
    end
end
