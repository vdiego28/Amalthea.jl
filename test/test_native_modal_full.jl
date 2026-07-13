using TestItems

@testitem "Native-Rust Phase E.3 (modal, full=true 2-D integral)" tags=[:rust] begin
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
        # docs/dev/BACKLOG.md Phase E.3: `full=true` — a genuine 2-D `(r,θ)` cubature
        # (`Cubature.hcubature_v`, the same h-adaptive routine Julia calls;
        # native.rs binds it directly, so node placement is bit-identical,
        # not just close) instead of the `full=false` radial-only integral
        # at a fixed `θ=0` (E.1/E.2's scope). Node placement being identical
        # AND the mode-field formula (`mode_angle_xy`) being exact (no
        # BigFloat-derivative approximation, unlike E.2's tapered-radius
        # β1(z)) means this reaches machine precision, tighter than E.1's
        # own `full=false` gate.
        gas = :Ar; pres = 1.0; τ = 20e-15; λ0 = 800e-9
        a = 125e-6; energy = 5e-6; L = 0.05

        grid = Grid.RealGrid(L, λ0, (400e-9, 2000e-9), 0.5e-12)
        t0 = 0.0
        dt = 0.001

        for (label, modes) in (
            ("HE n=1", (Capillary.MarcatiliMode(a, gas, pres; kind=:HE, n=1, m=1),)),
            ("HE n=2", (Capillary.MarcatiliMode(a, gas, pres; kind=:HE, n=2, m=1),)),
            ("TE", (Capillary.MarcatiliMode(a, gas, pres; kind=:TE, n=0, m=1),)),
            ("TM", (Capillary.MarcatiliMode(a, gas, pres; kind=:TM, n=0, m=1),)),
        )
            @testset "$label" begin
                dens0 = PhysData.density(gas, pres)
                densityfun(z) = dens0
                responses = (Nonlinear.Kerr_field(PhysData.γ3_gas(gas)),)
                linop = LinearOps.make_const_linop(grid, modes, grid.referenceλ)
                input = Fields.GaussField(λ0=λ0, τfwhm=τ, energy=energy)

                Eω, transform, FT = with_logger(NullLogger()) do
                    Amalthea.setup(grid, densityfun, responses, input, modes, :y; full=true, mfcn=2000)
                end
                @assert transform.full "Expected full=true"

                s_jl = PreconStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10,
                                      max_dt=dt, min_dt=dt)
                s_ru = RustNativeStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10,
                                          max_dt=dt, min_dt=dt)

                solve(s_jl, L)
                solve(s_ru, L)

                rel = norm(s_ru.yn - s_jl.yn) / norm(s_jl.yn)
                println("Modal full=true ($label) full-solve rel: ", rel)
                @test rel < 1e-10
            end
        end

        @testset "Non-vacuousness: full=true agrees with full=false for an axisymmetric mode" begin
            # HE,n=1 is azimuthally simple enough that `full=false`'s
            # analytic-θ-integral shortcut and `full=true`'s genuine 2-D
            # integral should agree closely — a useful cross-check that
            # `full=true`'s new code path isn't silently computing something
            # unrelated (both compared against the same Julia oracle
            # already above, but this checks the two Rust code paths against
            # each other too).
            modes = (Capillary.MarcatiliMode(a, gas, pres; kind=:HE, n=1, m=1),)
            dens0 = PhysData.density(gas, pres)
            densityfun(z) = dens0
            responses = (Nonlinear.Kerr_field(PhysData.γ3_gas(gas)),)
            linop = LinearOps.make_const_linop(grid, modes, grid.referenceλ)
            input = Fields.GaussField(λ0=λ0, τfwhm=τ, energy=energy)

            Eω1, transform_full, _ = with_logger(NullLogger()) do
                Amalthea.setup(grid, densityfun, responses, input, modes, :y; full=true, mfcn=2000)
            end
            Eω2, transform_radial, _ = with_logger(NullLogger()) do
                Amalthea.setup(grid, densityfun, responses, input, modes, :y; full=false)
            end

            s_full = RustNativeStepper(transform_full, linop, copy(Eω1), t0, dt, rtol=1e-6, atol=1e-10,
                                        max_dt=dt, min_dt=dt)
            s_radial = RustNativeStepper(transform_radial, linop, copy(Eω2), t0, dt, rtol=1e-6, atol=1e-10,
                                          max_dt=dt, min_dt=dt)
            solve(s_full, L)
            solve(s_radial, L)
            rel = norm(s_full.yn - s_radial.yn) / norm(s_radial.yn)
            println("full=true vs full=false (both native, HE11) rel: ", rel)
            # Not bitwise: `full=true`'s h-adaptive 2-D cubature and
            # `full=false`'s p-adaptive 1-D cubature (with the θ-integral
            # done analytically) converge to the true integral via distinct
            # node placements/subdivision strategies, so a small residual
            # quadrature-convergence difference between the two methods is
            # expected (measured ~5e-8), not a bug — each was already
            # verified independently against the Julia oracle above.
            @test rel < 1e-6
        end
    end
end
