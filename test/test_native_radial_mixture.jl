using TestItems

@testitem "Native-Rust Phase I item 4 (radial gas mixture, RealGrid)" tags=[:rust] begin
    import Test: @test, @test_skip, @testset
    using Amalthea
    import Amalthea: Grid, NonlinearRHS, Fields, LinearOps, PhysData, Nonlinear
    using Amalthea.RK45: PreconStepper, RustNativeStepper, step!, solve
    import Hankel
    import Logging: with_logger, NullLogger
    import LinearAlgebra: norm

    libpath = RK45._LIBLUNA_RUST_RK45
    if !isfile(libpath)
        @test_skip "Rust library not found"
    else
        # docs/dev/BACKLOG.md Phase I item 4: gas mixtures outside mode-averaged
        # geometry. Kerr is linear in density·γ3, so it collapses to one
        # scalar kerr_fac regardless of geometry — same reasoning as Phase
        # F.4's mode-averaged mixture support, just applied to radial here.
        gas = :Ar; pres = 1.2; τ = 20e-15; λ0 = 800e-9
        w0 = 40e-6; energy = 1e-12; L = 0.05; R = 4e-3; N = 32
        halfpres = PhysData.pressure(gas, PhysData.density(gas, pres)/2)

        grid = Grid.RealGrid(L, λ0, (400e-9, 2000e-9), 0.2e-12)
        q    = Hankel.QDHT(R, N, dim=2)

        dens0 = PhysData.density(gas, halfpres)
        densityfun(z) = [dens0, dens0]
        responses = (
            (Nonlinear.Kerr_field(PhysData.γ3_gas(gas)),),
            (Nonlinear.Kerr_field(PhysData.γ3_gas(gas)),),
        )
        refidx = PhysData.ref_index_fun((gas, gas), (halfpres, halfpres))
        linop   = LinearOps.make_const_linop(grid, q, refidx)
        normfun = NonlinearRHS.const_norm_radial(grid, q, refidx)
        inputs  = Fields.GaussGaussField(λ0=λ0, τfwhm=τ, energy=energy, w0=w0, propz=-0.15)

        Eω, transform, FT = with_logger(NullLogger()) do
            Amalthea.setup(grid, q, densityfun, normfun, responses, inputs)
        end

        @assert transform isa Amalthea.NonlinearRHS.TransRadial "Expected TransRadial"

        t0 = 0.0
        dt = 0.001

        @testset "Not silently ineligible: native stepper actually builds" begin
            s_ru = RustNativeStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10)
            @test s_ru isa RustNativeStepper
        end

        @testset "Single-step equivalence (radial mixture, ~1e-13)" begin
            s_jl = PreconStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10)
            s_ru = RustNativeStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10)

            step!(s_jl)
            step!(s_ru)

            rel_step = norm(s_ru.yn - s_jl.yn) / norm(s_jl.yn)
            println("Radial mixture single-step rel: ", rel_step)
            @test rel_step < 1e-13
        end

        @testset "Full-solve equivalence (radial mixture, fixed dt)" begin
            s_jl = PreconStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10,
                                  max_dt=dt, min_dt=dt)
            s_ru = RustNativeStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10,
                                  max_dt=dt, min_dt=dt)

            solve(s_jl, L)
            solve(s_ru, L)

            rel_solve = norm(s_ru.yn - s_jl.yn) / norm(s_jl.yn)
            println("Radial mixture full-solve rel: ", rel_solve)
            @test rel_solve < 1e-6
        end
    end
end
