using TestItems

@testitem "Native-Rust Phase E.4 item 5 (modal, EnvGrid)" tags=[:rust] begin
    import Test: @test, @test_skip, @testset, @test_throws
    using Luna
    import Luna: Grid, NonlinearRHS, Fields, LinearOps, PhysData, Nonlinear, Capillary, Modes, Raman
    using Luna.RK45: PreconStepper, RustNativeStepper, NativeIneligible, step!, solve
    import Logging: with_logger, NullLogger
    import LinearAlgebra: norm

    libpath = RK45._LIBLUNA_RUST_RK45
    if !isfile(libpath)
        @test_skip "Rust library not found"
    else
        # EnvGrid counterpart of test_native_modal_general_order.jl/
        # test_native_modal_npol2.jl (RealGrid) — docs/dev/BACKLOG.md Phase E.4 item 5,
        # the last unstarted native-port scope item. `rhs_modal_pointcalc`
        # (native.rs) now branches on `sim.is_real` (already set generically
        # by `native_set_fftw_plans`) between the RealGrid r2c path and a new
        # c2c path using `KerrScalarEnv!`/`KerrVectorEnv!`
        # (src/Nonlinear.jl:120-133) — note `KerrVectorEnv!` is a genuinely
        # different formula from RealGrid's `KerrVector!` (the `2/3`/`1/3`
        # split and the `conj(Ex)*Ey^2` cross term), not just a complex-typed
        # port of it. `ϕ=π/4` for the npol=2 case (as in test_native_modal_
        # npol2.jl) so both `Ex`/`Ey` are genuinely nonzero, exercising the
        # cross term.
        gas = :Ar; pres = 1.0; τ = 20e-15; λ0 = 800e-9
        a = 125e-6; energy = 5e-6; L = 0.05

        grid = Grid.EnvGrid(L, λ0, (400e-9, 2000e-9), 0.5e-12)
        t0 = 0.0
        dt = 0.001

        γ3 = PhysData.γ3_gas(gas)
        dens0 = PhysData.density(gas, pres)
        densityfun(z) = dens0
        responses = (Nonlinear.Kerr_env(γ3),)

        for (label, full) in (("full=false", false), ("full=true", true))
            @testset "npol=1 ($label)" begin
                modes = (Capillary.MarcatiliMode(a, gas, pres; kind=:HE, n=1, m=1),)
                linop = LinearOps.make_const_linop(grid, modes, grid.referenceλ)
                input = Fields.GaussField(λ0=λ0, τfwhm=τ, energy=energy)

                Eω, transform, FT = with_logger(NullLogger()) do
                    Luna.setup(grid, densityfun, responses, input, modes, :y; full)
                end
                @assert transform isa Luna.NonlinearRHS.TransModal "Expected TransModal"
                @assert transform.ts.npol == 1 "Expected npol=1"

                s_jl = PreconStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10,
                                      max_dt=dt, min_dt=dt)
                s_ru = RustNativeStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10,
                                          max_dt=dt, min_dt=dt)

                solve(s_jl, L)
                solve(s_ru, L)

                rel = norm(s_ru.yn - s_jl.yn) / norm(s_jl.yn)
                println("Modal EnvGrid npol=1 ($label) full-solve rel: ", rel)
                @test rel < 1e-9
            end
        end

        @testset "npol=2 (full=false, cross term)" begin
            modes = (Capillary.MarcatiliMode(a, gas, pres; kind=:HE, n=1, m=1, ϕ=π/4),)
            linop = LinearOps.make_const_linop(grid, modes, grid.referenceλ)
            input = Fields.GaussField(λ0=λ0, τfwhm=τ, energy=energy)

            Eω, transform, FT = with_logger(NullLogger()) do
                Luna.setup(grid, densityfun, responses, input, modes, :xy; full=false)
            end
            @assert transform isa Luna.NonlinearRHS.TransModal "Expected TransModal"
            @assert transform.ts.npol == 2 "Expected npol=2"

            s_jl = PreconStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10,
                                  max_dt=dt, min_dt=dt)
            s_ru = RustNativeStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10,
                                      max_dt=dt, min_dt=dt)

            solve(s_jl, L)
            solve(s_ru, L)

            rel = norm(s_ru.yn - s_jl.yn) / norm(s_jl.yn)
            println("Modal EnvGrid npol=2 full-solve rel: ", rel)
            @test rel < 1e-9
        end

        @testset "Raman + EnvGrid rejected (native.rs has no EnvGrid Raman path)" begin
            raman_gas = :N2
            modes = (Capillary.MarcatiliMode(a, raman_gas, pres; kind=:HE, n=1, m=1),)
            dens0_r = PhysData.density(raman_gas, pres)
            densityfun_r(z) = dens0_r
            rr = Raman.raman_response(grid.to, raman_gas; rotation=false, vibration=true)
            responses_r = (
                Nonlinear.Kerr_env(PhysData.γ3_gas(raman_gas)),
                Nonlinear.RamanPolarField(grid.to, rr),
            )
            linop = LinearOps.make_const_linop(grid, modes, grid.referenceλ)
            input = Fields.GaussField(λ0=λ0, τfwhm=τ, energy=energy)

            Eω, transform, FT = with_logger(NullLogger()) do
                Luna.setup(grid, densityfun_r, responses_r, input, modes, :y; full=false)
            end

            @test_throws NativeIneligible RustNativeStepper(transform, linop, copy(Eω), t0, dt,
                                      rtol=1e-6, atol=1e-10, max_dt=dt, min_dt=dt)
        end
    end
end
