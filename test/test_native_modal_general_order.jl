using TestItems

@testitem "Native-Rust Phase E.1 (modal, general mode orders)" tags=[:rust] begin
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
        # docs/dev/BACKLOG.md Phase E.1: general Marcatili mode orders. Phase 5/D.4
        # only supported `kind=:HE, n=1` (the `J0` special case); this test
        # exercises `:HE,n=2` (needs `J1`), `:TE` and `:TM` (both `J1`, but
        # with the *other* angular projection — `(0,1)`/`(1,0)` vs `:HE`'s
        # `(sin(nϕ),cos(nϕ))`) — see native.rs's `jn` (downward Miller
        # recurrence) and `rhs_modal_pointcalc`'s `modal_order`/
        # `modal_angle_x`/`_y` doc comments for the closed forms.
        gas = :Ar; pres = 1.0; τ = 20e-15; λ0 = 800e-9
        a = 125e-6; energy = 5e-6; L = 0.05

        grid = Grid.RealGrid(L, λ0, (400e-9, 2000e-9), 0.5e-12)
        t0 = 0.0
        dt = 0.001

        for (label, modes) in (
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
                    Amalthea.setup(grid, densityfun, responses, input, modes, :y)
                end
                @assert transform isa Amalthea.NonlinearRHS.TransModal "Expected TransModal"
                @assert !transform.full "Expected full=false (radial modal integral)"

                s_jl = PreconStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10,
                                      max_dt=dt, min_dt=dt)
                s_ru = RustNativeStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10,
                                          max_dt=dt, min_dt=dt)

                solve(s_jl, L)
                solve(s_ru, L)

                rel = norm(s_ru.yn - s_jl.yn) / norm(s_jl.yn)
                println("Modal general-order ($label) full-solve rel: ", rel)
                @test rel < 1e-9
            end
        end
    end
end
