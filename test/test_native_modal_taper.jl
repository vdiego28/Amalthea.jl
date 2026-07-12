using TestItems

@testitem "Native-Rust Phase E.2 (modal, tapered radius)" tags=[:rust] begin
    import Test: @test, @test_skip, @testset
    using Luna
    import Luna: Grid, NonlinearRHS, Fields, LinearOps, PhysData, Nonlinear, Capillary, Modes
    using Luna.RK45: PreconStepper, RustNativeStepper, step!, solve
    import Logging: with_logger, NullLogger
    import LinearAlgebra: norm

    libpath = RK45._LIBLUNA_RUST_RK45
    if !isfile(libpath)
        @test_skip "Rust library not found"
    else
        # docs/dev/BACKLOG.md Phase E.2: tapered core radius, `a(z)` a shared Julia
        # `Function` across all modes of one physical fibre. Unlike Phase 7
        # (constant radius, varying density) this is the mirror case: density
        # is constant but the Marcatili waveguide term `nwg(ω;z)` varies
        # through `a(z)`. `Capillary.make_linop`'s specialized method wraps
        # this in `ZDepLinopModalTaper`, transferring `a(z)` as a dense-
        # sampled natural cubic spline (`Maths.CSpline` — NOT a re-fit of an
        # existing spline, so no spline-of-spline risk) plus 4 per-mode
        # BigFloat-differentiated ω0 constants for the closed-form β1(z), the
        # same pattern as Phase 7/D.5. `E.1`'s general order/angle machinery
        # is exercised unchanged (HE n=2, TE, TM all included here too).
        gas = :Ar; pres = 1.0; τ = 20e-15; λ0 = 800e-9
        a0 = 125e-6; aL = 100e-6; energy = 5e-6; L = 0.05

        afun = let a0=a0, aL=aL, L=L
            z -> a0 + (aL - a0) * z / L
        end

        grid = Grid.RealGrid(L, λ0, (400e-9, 2000e-9), 0.5e-12)
        t0 = 0.0
        dt = 0.001

        for (label, modes) in (
            ("HE n=1", (Capillary.MarcatiliMode(afun, gas, pres; kind=:HE, n=1, m=1),)),
            ("HE n=2", (Capillary.MarcatiliMode(afun, gas, pres; kind=:HE, n=2, m=1),)),
            ("TE", (Capillary.MarcatiliMode(afun, gas, pres; kind=:TE, n=0, m=1),)),
            ("TM", (Capillary.MarcatiliMode(afun, gas, pres; kind=:TM, n=0, m=1),)),
        )
            @testset "$label" begin
                dens0 = PhysData.density(gas, pres)
                densityfun(z) = dens0
                responses = (Nonlinear.Kerr_field(PhysData.γ3_gas(gas)),)
                linop = LinearOps.make_linop(grid, modes, λ0)
                @assert linop isa Capillary.ZDepLinopModalTaper "Expected ZDepLinopModalTaper"
                input = Fields.GaussField(λ0=λ0, τfwhm=τ, energy=energy)

                Eω, transform, FT = with_logger(NullLogger()) do
                    Luna.setup(grid, densityfun, responses, input, modes, :y)
                end
                @assert transform isa Luna.NonlinearRHS.TransModal "Expected TransModal"

                s_jl = PreconStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10,
                                      max_dt=dt, min_dt=dt)
                s_ru = RustNativeStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10,
                                          max_dt=dt, min_dt=dt, flength=L)

                solve(s_jl, L)
                solve(s_ru, L)

                # β1(z)'s closed-form (BigFloat-differentiated ω0 constants)
                # vs Julia's own adaptive-FD `dispersion(mode,1,ω0,z=z)` is
                # the same deliberate-divergence tier as Phase 7/D.5 —
                # ~1e-7/1e-8, not the ~1e-13 bitwise tier of the constant-a
                # modal test.
                rel = norm(s_ru.yn - s_jl.yn) / norm(s_jl.yn)
                println("Modal tapered-radius ($label) full-solve rel: ", rel)
                @test rel < 1e-6
            end
        end

        @testset "Non-vacuousness: two-mode HE11+HE12 coupling under the taper" begin
            modes = (Capillary.MarcatiliMode(afun, gas, pres; kind=:HE, n=1, m=1),
                     Capillary.MarcatiliMode(afun, gas, pres; kind=:HE, n=1, m=2))
            dens0 = PhysData.density(gas, pres)
            densityfun(z) = dens0
            responses = (Nonlinear.Kerr_field(PhysData.γ3_gas(gas)),)
            linop = LinearOps.make_linop(grid, modes, λ0)
            input = Fields.GaussField(λ0=λ0, τfwhm=τ, energy=energy)
            Eω, transform, FT = with_logger(NullLogger()) do
                Luna.setup(grid, densityfun, responses, input, modes, :y)
            end
            s_jl = PreconStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10,
                                  max_dt=dt, min_dt=dt)
            solve(s_jl, L)
            he12_frac = sum(abs2, s_jl.yn[:, 2]) / sum(abs2, s_jl.yn)
            println("Fraction of energy transferred to HE12 under taper (Julia, sanity check): ",
                    he12_frac)
            @test he12_frac > 1e-9

            s_ru = RustNativeStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10,
                                      max_dt=dt, min_dt=dt, flength=L)
            solve(s_ru, L)
            rel = norm(s_ru.yn - s_jl.yn) / norm(s_jl.yn)
            println("Modal tapered-radius (two-mode) full-solve rel: ", rel)
            @test rel < 1e-6
        end
    end
end
