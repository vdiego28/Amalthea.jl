using TestItems

@testitem "Native-Rust Phase F.4 (gas mixtures, mode-avg Kerr)" tags=[:rust] begin
    import Test: @test, @test_skip, @testset
    using Luna
    using Luna.RK45: PreconStepper, RustNativeStepper, step!, solve
    import Logging: with_logger, NullLogger
    import LinearAlgebra: norm

    libpath = RK45._LIBLUNA_RUST_RK45
    if !isfile(libpath)
        @test_skip "Rust library not found"
    else
        # ── Test Geometry & Setup ─────────────────────────────────────────────
        # A 2-species mixture of the same gas, each at half the pressure of
        # the single-gas reference — physically equivalent to the single-gas
        # case (test_mixtures.jl's own "propagation" testset), but exercises
        # `densityfun(z)::Vector` + a nested tuple-of-tuples `resp`, the
        # shape Phase F item 4 (docs/dev/BACKLOG.md) adds native support for.
        a = 125e-6
        gas = :Ar
        pres = 5.0
        flength = 0.15
        λ0 = 800e-9
        τfwhm = 30e-15

        halfpres = PhysData.pressure(gas, PhysData.density(gas, pres) / 2)

        grid = Grid.RealGrid(flength, λ0, (160e-9, 3000e-9), 1e-12)
        inputs = Fields.GaussField(λ0=λ0, τfwhm=τfwhm, energy=1e-6)

        m = Capillary.MarcatiliMode(a, (gas, gas), (halfpres, halfpres); loss=false)
        aeff(z) = Modes.Aeff(m; z=z)
        densityfun = let dens0 = PhysData.density(gas, halfpres)
            z -> [dens0, dens0]
        end
        responses = (
            (Nonlinear.Kerr_field(PhysData.γ3_gas(gas)),),
            (Nonlinear.Kerr_field(PhysData.γ3_gas(gas)),)
        )
        linop, βfun!, _, _ = LinearOps.make_const_linop(grid, m, λ0)
        Eω, transform, FT = Luna.setup(grid, densityfun, responses, inputs, βfun!, aeff)

        t0 = 0.0
        dt = 0.01

        @testset "Single-step equivalence (~1e-13)" begin
            s_jl = PreconStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10)
            s_ru = RustNativeStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10)

            ok_jl = step!(s_jl)
            ok_ru = step!(s_ru)

            @test ok_jl == ok_ru
            @test s_jl.tn ≈ s_ru.tn
            @test s_jl.dtn ≈ s_ru.dtn rtol=1e-11
            @test s_jl.err ≈ s_ru.err rtol=1e-11
            @test s_jl.errlast ≈ s_ru.errlast rtol=1e-11

            rel_step = norm(s_ru.yn - s_jl.yn) / norm(s_jl.yn)
            @test rel_step < 1e-13
        end

        @testset "Full-solve equivalence (~1e-6)" begin
            s_jl = PreconStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10,
                                  max_dt=dt, min_dt=dt)
            s_ru = RustNativeStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10,
                                  max_dt=dt, min_dt=dt)

            solve(s_jl, flength)
            solve(s_ru, flength)

            rel_solve = norm(s_ru.yn - s_jl.yn) / norm(s_jl.yn)
            println("Phase F.4 mixture full-solve rel_solve: ", rel_solve)
            @test rel_solve < 1e-6
        end

        @testset "Rejections (out-of-scope mixture configs)" begin
            # Plasma on one species: not a simple Kerr sum, must fall back.
            responses_bad = (
                (Nonlinear.Kerr_field(PhysData.γ3_gas(gas)),),
                (Nonlinear.Kerr_field(PhysData.γ3_gas(gas)), Nonlinear.Kerr_field(PhysData.γ3_gas(gas)))
            )
            Eω_bad, transform_bad, _ = Luna.setup(grid, densityfun, responses_bad, inputs, βfun!, aeff)
            @test_throws RK45.NativeIneligible RustNativeStepper(
                transform_bad, linop, copy(Eω_bad), t0, dt, rtol=1e-6, atol=1e-10)
        end
    end
end
