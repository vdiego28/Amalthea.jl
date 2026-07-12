using TestItems

@testitem "Native-Rust Phase F.2 (envelope Raman + rotational multi-oscillator)" tags=[:rust] begin
    import Test: @test, @test_skip, @testset
    using Amalthea
    import Amalthea: Grid, NonlinearRHS, Fields, LinearOps, PhysData, Nonlinear, Capillary, Modes, Raman
    using Amalthea.RK45: PreconStepper, RustNativeStepper, step!, solve
    import Logging: with_logger, NullLogger
    import LinearAlgebra: norm

    libpath = RK45._LIBLUNA_RUST_RK45
    if !isfile(libpath)
        @test_skip "Rust library not found"
    else
        gas = :N2; pres = 1.0; τ = 20e-15; λ0 = 800e-9
        a = 125e-6; energy = 5e-6; L = 0.05
        dens0 = PhysData.density(gas, pres)
        densityfun(z) = dens0

        @testset "Envelope Raman (RamanPolarEnv, mode-averaged EnvGrid)" begin
            # docs/dev/BACKLOG.md Phase F item 2: RamanPolarEnv was never natively
            # wired at all (neither the old LUNA_USE_RUST_RAMAN per-kernel
            # path nor the native port) — its intensity formula
            # (sqr!(R::RamanPolarEnv,E) = 1/2*|E|^2, Nonlinear.jl:351-354)
            # has no thg branch, so the resident real-valued ADE solver
            # applies unchanged; only the accumulation (complex field *
            # real polarisation) differs from RamanPolarField's real case.
            grid = Grid.EnvGrid(L, λ0, (400e-9, 2000e-9), 0.5e-12)
            mode = Capillary.MarcatiliMode(a, gas, pres; kind=:HE, n=1, m=1)
            rr = Raman.raman_response(grid.to, gas; rotation=false, vibration=true)
            responses = (Nonlinear.Kerr_env(PhysData.γ3_gas(gas)), Nonlinear.RamanPolarEnv(grid.to, rr))
            linop, βfun!, _, _ = LinearOps.make_const_linop(grid, mode, grid.referenceλ)
            aeff = z -> Modes.Aeff(mode, z=z)
            input = Fields.GaussField(λ0=λ0, τfwhm=τ, energy=energy)

            Eω, transform, FT = with_logger(NullLogger()) do
                Amalthea.setup(grid, densityfun, responses, input, βfun!, aeff)
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
            println("Envelope Raman full-solve rel: ", rel_solve)
            # 2e-5, not the usual 1e-6 tier: same ADE-vs-FFT-convolution
            # method-difference floor as test_native_raman.jl/
            # test_native_radial_raman.jl (measured ~1.5e-5 here — a new
            # formula path, first exercised by this test) — still five
            # orders of magnitude below the non-vacuousness effect
            # confirmed below.
            @test rel_solve < 2e-5

            resp_noraman = (Nonlinear.Kerr_env(PhysData.γ3_gas(gas)),)
            transform_nr = NonlinearRHS.TransModeAvg(grid, transform.FT, resp_noraman, densityfun,
                                                      transform.norm!, aeff)
            s_jl_nr = PreconStepper(transform_nr, linop, copy(Eω), t0, dt,
                                     rtol=1e-6, atol=1e-10, max_dt=dt, min_dt=dt)
            solve(s_jl_nr, L)
            rel_matters = norm(s_jl.yn - s_jl_nr.yn) / norm(s_jl.yn)
            println("Envelope Raman-on vs Raman-off rel (Julia, non-vacuousness): ", rel_matters)
            @test rel_matters > 1e-8
        end

        @testset "Rotational multi-oscillator (RamanPolarField, mode-averaged RealGrid)" begin
            # docs/dev/BACKLOG.md Phase F item 2: RamanRespRotationalNonRigid wraps a
            # per-J Vector{RamanRespSingleDampedOscillator} internally
            # (Raman.jl:232-234) sharing one τ2ρ — Raman.flatten_sdo_oscillators
            # expands it so a rotational (here rotation-only) response
            # reaches the resident ADE solver, which already accepted
            # n_osc>1 (only the Julia-side extraction was SDO-only before).
            grid = Grid.RealGrid(L, λ0, (400e-9, 2000e-9), 0.5e-12)
            mode = Capillary.MarcatiliMode(a, gas, pres; kind=:HE, n=1, m=1)
            rr = Raman.raman_response(grid.to, gas; rotation=true, vibration=false)
            @assert length(rr.Rs) == 1 && rr.Rs[1] isa Raman.RamanRespRotationalNonRigid
            @assert length(rr.Rs[1].Rs) > 1 "Expected multiple rotational J-line oscillators"
            responses = (Nonlinear.Kerr_field(PhysData.γ3_gas(gas)), Nonlinear.RamanPolarField(grid.to, rr))
            linop, βfun!, _, _ = LinearOps.make_const_linop(grid, mode, grid.referenceλ)
            aeff = z -> Modes.Aeff(mode, z=z)
            input = Fields.GaussField(λ0=λ0, τfwhm=τ, energy=energy)

            Eω, transform, FT = with_logger(NullLogger()) do
                Amalthea.setup(grid, densityfun, responses, input, βfun!, aeff)
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
            println("Rotational-Raman full-solve rel: ", rel_solve)
            @test rel_solve < 1e-6

            resp_noraman = (Nonlinear.Kerr_field(PhysData.γ3_gas(gas)),)
            transform_nr = NonlinearRHS.TransModeAvg(grid, transform.FT, resp_noraman, densityfun,
                                                      transform.norm!, aeff)
            s_jl_nr = PreconStepper(transform_nr, linop, copy(Eω), t0, dt,
                                     rtol=1e-6, atol=1e-10, max_dt=dt, min_dt=dt)
            solve(s_jl_nr, L)
            rel_matters = norm(s_jl.yn - s_jl_nr.yn) / norm(s_jl.yn)
            println("Rotational Raman-on vs Raman-off rel (Julia, non-vacuousness): ", rel_matters)
            @test rel_matters > 1e-8
        end

        @testset "Density-dependent τ2 (H2 vibrational, RamanPolarField, mode-averaged)" begin
            # H2's vibrational line is density-dependent (Bρv/Aρv), the
            # negative control MATH.md §5.3 explicitly calls out as
            # ineligible under the old density-independence-only check —
            # now eligible since gammas are evaluated at the sim's actual
            # constant density instead of a placeholder ρ=1.0.
            gasH = :H2
            grid = Grid.RealGrid(L, λ0, (400e-9, 2000e-9), 0.5e-12)
            mode = Capillary.MarcatiliMode(a, gasH, pres; kind=:HE, n=1, m=1)
            densH0 = PhysData.density(gasH, pres)
            densityfunH(z) = densH0
            rr = Raman.raman_response(grid.to, gasH; rotation=false, vibration=true)
            @assert abs(rr.Rs[1].τ2ρ(1.0) - rr.Rs[1].τ2ρ(2.0)) > 1e-6 * abs(rr.Rs[1].τ2ρ(1.0)) "Expected H2's τ2 to be density-dependent"
            responses = (Nonlinear.Kerr_field(PhysData.γ3_gas(gasH)), Nonlinear.RamanPolarField(grid.to, rr))
            linop, βfun!, _, _ = LinearOps.make_const_linop(grid, mode, grid.referenceλ)
            aeff = z -> Modes.Aeff(mode, z=z)
            input = Fields.GaussField(λ0=λ0, τfwhm=τ, energy=energy)

            Eω, transform, FT = with_logger(NullLogger()) do
                Amalthea.setup(grid, densityfunH, responses, input, βfun!, aeff)
            end

            t0 = 0.0
            dt = 0.01
            s_jl = PreconStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10,
                                  max_dt=dt, min_dt=dt)
            s_ru = RustNativeStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10,
                                      max_dt=dt, min_dt=dt)
            solve(s_jl, L)
            solve(s_ru, L)
            rel_solve = norm(s_ru.yn - s_jl.yn) / norm(s_jl.yn)
            println("H2 density-dependent-τ2 Raman full-solve rel: ", rel_solve)
            # Same ADE-vs-FFT-convolution method-difference floor as above
            # (measured ~9.7e-6).
            @test rel_solve < 2e-5
        end
    end
end
