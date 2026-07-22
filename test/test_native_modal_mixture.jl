using TestItems

@testitem "Native-Rust Phase I item 4 (modal gas mixture, RealGrid)" tags=[:rust] begin
    import Test: @test, @test_skip, @testset
    using Amalthea
    import Amalthea: Grid, NonlinearRHS, Fields, LinearOps, PhysData, Nonlinear, Capillary, Modes
    using Amalthea.RK45: PreconStepper, RustNativeStepper, step!, solve
    import Logging: with_logger, NullLogger
    import LinearAlgebra: norm

    libpath = RK45._LIBAMALTHEA_RK45
    if !isfile(libpath)
        @test_skip "Rust library not found"
    else
        # docs/dev/BACKLOG.md Phase I item 4 (modal): gas mixtures outside
        # mode-averaged geometry. Discriminating check found the same
        # collapse as mode-averaged/radial: `Erω_to_Prω!` calls the SAME
        # generic `Et_to_Pt!(t.Pr, t.Er, t.resp, t.density)` at every
        # quadrature node (NonlinearRHS.jl), which already dispatches
        # per-species for `density::AbstractVector`; the modal-specific
        # normalization quantities (`invsqrtn = 1/√N(m,z=0)`, `nlfac` from
        # `norm_modal`) are both pure functions of mode geometry / grid.ω —
        # no density argument at all. So Kerr's linearity in density·γ3
        # collapses a per-species mixture to one scalar kerr_fac here too.
        gas = :Ar; pres = 1.0; τ = 20e-15; λ0 = 800e-9
        a = 125e-6; energy = 5e-6; L = 0.1
        halfpres = PhysData.pressure(gas, PhysData.density(gas, pres)/2)

        grid = Grid.RealGrid(L, λ0, (400e-9, 2000e-9), 0.5e-12)

        # Single HE1m mode; the *linear* index (used for the linop) is built
        # from a genuine two-species mixture at half the single-gas pressure
        # each, so the linop/dispersion match the single-gas case (same
        # reasoning as test_mixtures.jl / test_native_radial_mixture.jl).
        modes = (Capillary.MarcatiliMode(a, (gas, gas), (halfpres, halfpres); m=1),)

        dens0 = PhysData.density(gas, halfpres)
        densityfun(z) = [dens0, dens0]
        responses = (
            (Nonlinear.Kerr_field(PhysData.γ3_gas(gas)),),
            (Nonlinear.Kerr_field(PhysData.γ3_gas(gas)),),
        )
        linop = LinearOps.make_const_linop(grid, modes, grid.referenceλ)
        input = Fields.GaussField(λ0=λ0, τfwhm=τ, energy=energy)

        Eω, transform, FT = with_logger(NullLogger()) do
            Amalthea.setup(grid, densityfun, responses, input, modes, :y)
        end

        @assert transform isa Amalthea.NonlinearRHS.TransModal "Expected TransModal"
        @assert !transform.full "Expected full=false (radial modal integral)"

        t0 = 0.0
        dt = 0.001

        @testset "Not silently ineligible: native stepper actually builds" begin
            s_ru = RustNativeStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10)
            @test s_ru isa RustNativeStepper
        end

        @testset "Single-step equivalence (modal mixture, ~1e-10)" begin
            s_jl = PreconStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10)
            s_ru = RustNativeStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10)

            step!(s_jl)
            step!(s_ru)

            rel_step = norm(s_ru.yn - s_jl.yn) / norm(s_jl.yn)
            println("Modal mixture single-step rel: ", rel_step)
            # Same tolerance tier as the plain modal test (Phase 5): the
            # mixture collapse changes nothing about the mode-field
            # synthesis / cubature that sets that tier.
            @test rel_step < 1e-10
        end

        @testset "Full-solve equivalence (modal mixture, fixed dt)" begin
            s_jl = PreconStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10,
                                  max_dt=dt, min_dt=dt)
            s_ru = RustNativeStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10,
                                  max_dt=dt, min_dt=dt)

            solve(s_jl, L)
            solve(s_ru, L)

            rel_solve = norm(s_ru.yn - s_jl.yn) / norm(s_jl.yn)
            println("Modal mixture full-solve rel: ", rel_solve)
            @test rel_solve < 1e-6
        end

        @testset "Non-vacuousness: mixture sums BOTH species (not silently one)" begin
            # Guards against a "mixture silently ignored" regression: if the
            # native path only picked up the first species (dropping the
            # second term of the kerr_fac sum, or misreading `density` as a
            # bare Number equal to just one entry), the effective Kerr
            # strength would match a *single*-species run at the SAME HALF
            # pressure as one mixture component, instead of the sum over
            # both. Build that single-species/half-density comparison run
            # (deliberately the buggy-looking one, not the total-density
            # match used above) and confirm the two-species mixture result
            # is NOT vacuously identical to it.
            modes_half = (Capillary.MarcatiliMode(a, gas, halfpres; m=1),)
            dens0_half = PhysData.density(gas, halfpres)
            densityfun_half(z) = dens0_half
            responses_half = (Nonlinear.Kerr_field(PhysData.γ3_gas(gas)),)
            linop_half = LinearOps.make_const_linop(grid, modes_half, grid.referenceλ)
            Eω_half, transform_half, FT_half = with_logger(NullLogger()) do
                Amalthea.setup(grid, densityfun_half, responses_half, input, modes_half, :y)
            end
            s_half = RustNativeStepper(transform_half, linop_half, copy(Eω_half), t0, dt,
                                        rtol=1e-6, atol=1e-10, max_dt=dt, min_dt=dt)
            s_mix = RustNativeStepper(transform, linop, copy(Eω), t0, dt,
                                       rtol=1e-6, atol=1e-10, max_dt=dt, min_dt=dt)
            solve(s_half, L)
            solve(s_mix, L)
            rel_diff = norm(s_mix.yn - s_half.yn) / norm(s_half.yn)
            println("Modal mixture vs single-species(one component's pressure) rel diff: ", rel_diff)
            @test rel_diff > 1e-3
        end
    end
end
