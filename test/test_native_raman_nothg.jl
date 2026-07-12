using TestItems

@testitem "Native-Rust Phase F.1 (Raman thg=false, mode-avg/radial/modal)" tags=[:rust] begin
    import Test: @test, @test_skip, @testset
    using Luna
    import Luna: Grid, NonlinearRHS, Fields, LinearOps, PhysData, Nonlinear, Capillary, Modes, Raman
    using Luna.RK45: PreconStepper, RustNativeStepper, step!, solve
    import Hankel
    import Logging: with_logger, NullLogger
    import LinearAlgebra: norm

    libpath = RK45._LIBLUNA_RUST_RK45
    if !isfile(libpath)
        @test_skip "Rust library not found"
    else
        # `thg=false` Raman (`sqr!`'s `1/2·|hilbert(E)|²` branch,
        # Nonlinear.jl:342-349) needs its own resident c2c FFT plan since
        # RealGrid otherwise has none — docs/dev/BACKLOG.md Phase F item 1. Wired into
        # all three geometries that already carry `thg=true` Raman
        # (mode-averaged, radial, modal — Phase D.4). `Interface.makeresponse`
        # couples Kerr's and Raman's `thg` to the same flag, so a
        # Kerr_field (thg=true, native-eligible) + RamanPolarField(thg=false)
        # combination can only be built manually, not via `prop_capillary`.
        gas = :N2; pres = 1.0; τ = 20e-15; λ0 = 800e-9
        γ3 = PhysData.γ3_gas(gas)

        @testset "Mode-averaged" begin
            a = 125e-6; energy = 5e-6; L = 0.05
            grid = Grid.RealGrid(L, λ0, (400e-9, 2000e-9), 0.5e-12)
            mode = Capillary.MarcatiliMode(a, gas, pres; kind=:HE, n=1, m=1)
            dens0 = PhysData.density(gas, pres)
            densityfun(z) = dens0
            rr = Raman.raman_response(grid.to, gas; rotation=false, vibration=true)
            raman_nothg = Nonlinear.RamanPolarField(grid.to, rr; thg=false)
            responses = (Nonlinear.Kerr_field(γ3), raman_nothg)
            input = Fields.GaussField(λ0=λ0, τfwhm=τ, energy=energy)

            linop, βfun!, _, _ = LinearOps.make_const_linop(grid, mode, grid.referenceλ)
            aeff = z -> Modes.Aeff(mode, z=z)

            Eω, transform, FT = with_logger(NullLogger()) do
                Luna.setup(grid, densityfun, responses, input, βfun!, aeff)
            end
            @assert transform isa NonlinearRHS.TransModeAvg "Expected TransModeAvg"

            t0 = 0.0
            dt = 0.01

            s_jl = PreconStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10,
                                  max_dt=dt, min_dt=dt)
            s_ru = RustNativeStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10,
                                      max_dt=dt, min_dt=dt)
            solve(s_jl, L)
            solve(s_ru, L)
            rel_solve = norm(s_ru.yn - s_jl.yn) / norm(s_jl.yn)
            println("Mode-avg Raman thg=false full-solve rel: ", rel_solve)
            @test rel_solve < 1e-6

            resp_thg = (Nonlinear.Kerr_field(γ3), Nonlinear.RamanPolarField(grid.to, rr))
            transform_thg = NonlinearRHS.TransModeAvg(grid, transform.FT, resp_thg, densityfun,
                                                        transform.norm!, aeff)
            s_jl_thg = PreconStepper(transform_thg, linop, copy(Eω), t0, dt,
                                      rtol=1e-6, atol=1e-10, max_dt=dt, min_dt=dt)
            solve(s_jl_thg, L)
            rel_thg_matters = norm(s_jl.yn - s_jl_thg.yn) / norm(s_jl.yn)
            println("Mode-avg Raman thg=false vs thg=true rel (Julia, non-vacuousness): ", rel_thg_matters)
            @test rel_thg_matters > 1e-8
        end

        @testset "Radial" begin
            pres_r = 1.5; τr = 15e-15; w0 = 150e-6; energy = 6e-5; L = 0.02; R = 4e-3; N = 32
            grid = Grid.RealGrid(L, λ0, (150e-9, 2000e-9), 0.5e-12)
            q = Hankel.QDHT(R, N, dim=2)
            dens0 = PhysData.density(gas, pres_r)
            densityfun(z) = dens0
            rr = Raman.raman_response(grid.to, gas; rotation=false, vibration=true)
            responses = (Nonlinear.Kerr_field(PhysData.γ3_gas(gas)),
                         Nonlinear.RamanPolarField(grid.to, rr; thg=false))
            linop = LinearOps.make_const_linop(grid, q, PhysData.ref_index_fun(gas, pres_r))
            normfun = NonlinearRHS.const_norm_radial(grid, q, PhysData.ref_index_fun(gas, pres_r))
            inputs = Fields.GaussGaussField(λ0=λ0, τfwhm=τr, energy=energy, w0=w0, propz=-0.1)

            Eω, transform, FT = with_logger(NullLogger()) do
                Luna.setup(grid, q, densityfun, normfun, responses, inputs)
            end
            @assert transform isa NonlinearRHS.TransRadial "Expected TransRadial"

            t0 = 0.0
            dt = 0.0005
            s_jl = PreconStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10,
                                  max_dt=dt, min_dt=dt)
            s_ru = RustNativeStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10,
                                      max_dt=dt, min_dt=dt)
            solve(s_jl, L)
            solve(s_ru, L)
            rel_solve = norm(s_ru.yn - s_jl.yn) / norm(s_jl.yn)
            println("Radial Raman thg=false full-solve rel: ", rel_solve)
            @test rel_solve < 1e-6

            transform_thg = NonlinearRHS.TransRadial(grid, q, transform.FT,
                (Nonlinear.Kerr_field(PhysData.γ3_gas(gas)), Nonlinear.RamanPolarField(grid.to, rr)),
                densityfun, normfun)
            s_jl_thg = PreconStepper(transform_thg, linop, copy(Eω), t0, dt,
                                      rtol=1e-6, atol=1e-10, max_dt=dt, min_dt=dt)
            solve(s_jl_thg, L)
            rel_thg_matters = norm(s_jl.yn - s_jl_thg.yn) / norm(s_jl.yn)
            println("Radial Raman thg=false vs thg=true rel (Julia, non-vacuousness): ", rel_thg_matters)
            @test rel_thg_matters > 1e-8
        end

        @testset "Modal" begin
            a = 125e-6; energy = 5e-6; L = 0.05
            grid = Grid.RealGrid(L, λ0, (400e-9, 2000e-9), 0.5e-12)
            modes = (Capillary.MarcatiliMode(a, gas, pres; m=1),)
            dens0 = PhysData.density(gas, pres)
            densityfun(z) = dens0
            rr = Raman.raman_response(grid.to, gas; rotation=false, vibration=true)
            responses = (Nonlinear.Kerr_field(γ3), Nonlinear.RamanPolarField(grid.to, rr; thg=false))
            linop = LinearOps.make_const_linop(grid, modes, grid.referenceλ)
            input = Fields.GaussField(λ0=λ0, τfwhm=τ, energy=energy)

            Eω, transform, FT = with_logger(NullLogger()) do
                Luna.setup(grid, densityfun, responses, input, modes, :y)
            end
            @assert transform isa NonlinearRHS.TransModal "Expected TransModal"

            t0 = 0.0
            dt = 0.001
            s_jl = PreconStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10,
                                  max_dt=dt, min_dt=dt)
            s_ru = RustNativeStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10,
                                      max_dt=dt, min_dt=dt)
            solve(s_jl, L)
            solve(s_ru, L)
            rel_solve = norm(s_ru.yn - s_jl.yn) / norm(s_jl.yn)
            println("Modal Raman thg=false full-solve rel: ", rel_solve)
            @test rel_solve < 1.5e-6

            responses_thg = (Nonlinear.Kerr_field(γ3), Nonlinear.RamanPolarField(grid.to, rr))
            Eω_thg, transform_thg, _ = with_logger(NullLogger()) do
                Luna.setup(grid, densityfun, responses_thg, input, modes, :y)
            end
            s_jl_thg = PreconStepper(transform_thg, linop, copy(Eω_thg), t0, dt,
                                      rtol=1e-6, atol=1e-10, max_dt=dt, min_dt=dt)
            solve(s_jl_thg, L)
            rel_thg_matters = norm(s_jl.yn - s_jl_thg.yn) / norm(s_jl.yn)
            println("Modal Raman thg=false vs thg=true rel (Julia, non-vacuousness): ", rel_thg_matters)
            @test rel_thg_matters > 1e-8
        end
    end
end
