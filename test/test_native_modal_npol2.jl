using TestItems

@testitem "Native-Rust Phase E.4 (modal, npol=2)" tags=[:rust] begin
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
        # docs/dev/BACKLOG.md Phase E.4: npol=2 (`components=:xy`) — `Modes.ToSpace`
        # resolves each mode into both its x and y real-space components
        # (`Ems[m,:] = Exy(mode,xs)[1:2]`), and Kerr couples them
        # (`KerrVector!`, src/Nonlinear.jl:85-94: `out[:,1] += fac*(Ex²+Ey²)*Ex`,
        # `out[:,2] += fac*(Ex²+Ey²)*Ey`). native.rs's `rhs_modal_pointcalc`
        # already carried a bit-for-bit port of this formula (`native.rs`,
        # the `npol==2` branch) since Phase 5/E.1-E.3 were written generically
        # over `npol` — this was gated off in `RK45.jl` pending verification
        # against the Julia oracle; this test is that verification. `ϕ=π/4`
        # is chosen (rather than `ϕ=0`) so the mode's own field has genuinely
        # nonzero x AND y components at every (r,θ), exercising the cross
        # term (`ϕ=0` would leave `ey≡0`, silently degenerating to the
        # npol=1 formula).
        gas = :Ar; pres = 1.0; τ = 20e-15; λ0 = 800e-9
        a = 125e-6; energy = 5e-6; L = 0.05

        grid = Grid.RealGrid(L, λ0, (400e-9, 2000e-9), 0.5e-12)
        t0 = 0.0
        dt = 0.001

        for (label, full) in (("full=false", false), ("full=true", true))
            @testset "$label" begin
                modes = (Capillary.MarcatiliMode(a, gas, pres; kind=:HE, n=1, m=1, ϕ=π/4),)
                dens0 = PhysData.density(gas, pres)
                densityfun(z) = dens0
                responses = (Nonlinear.Kerr_field(PhysData.γ3_gas(gas)),)
                linop = LinearOps.make_const_linop(grid, modes, grid.referenceλ)
                input = Fields.GaussField(λ0=λ0, τfwhm=τ, energy=energy)

                extra = full ? (mfcn=2000,) : NamedTuple()
                Eω, transform, FT = with_logger(NullLogger()) do
                    Luna.setup(grid, densityfun, responses, input, modes, :xy; full, extra...)
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
                println("Modal npol=2 ($label) full-solve rel: ", rel)
                @test rel < 1e-9
            end
        end

        @testset "Raman + npol=2 rejected (native.rs Raman inline is npol=1-only)" begin
            raman_gas = :N2
            modes = (Capillary.MarcatiliMode(a, raman_gas, pres; kind=:HE, n=1, m=1, ϕ=π/4),)
            dens0 = PhysData.density(raman_gas, pres)
            densityfun(z) = dens0
            rr = Raman.raman_response(grid.to, raman_gas; rotation=false, vibration=true)
            responses = (
                Nonlinear.Kerr_field(PhysData.γ3_gas(raman_gas)),
                Nonlinear.RamanPolarField(grid.to, rr),
            )
            linop = LinearOps.make_const_linop(grid, modes, grid.referenceλ)
            input = Fields.GaussField(λ0=λ0, τfwhm=τ, energy=energy)

            Eω, transform, FT = with_logger(NullLogger()) do
                Luna.setup(grid, densityfun, responses, input, modes, :xy; full=false)
            end

            @test_throws NativeIneligible RustNativeStepper(transform, linop, copy(Eω), t0, dt,
                                      rtol=1e-6, atol=1e-10, max_dt=dt, min_dt=dt)
        end
    end
end
