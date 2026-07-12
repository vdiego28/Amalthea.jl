using TestItems

@testitem "Native-Rust Phase D.4 (Raman, radial)" tags=[:rust] begin
    import Test: @test, @test_skip, @testset
    using Amalthea
    import Amalthea: Grid, NonlinearRHS, Fields, LinearOps, PhysData, Nonlinear, Raman
    using Amalthea.RK45: PreconStepper, RustNativeStepper, step!, solve
    import Hankel
    import Logging: with_logger, NullLogger
    import LinearAlgebra: norm

    libpath = RK45._LIBLUNA_RUST_RK45
    if !isfile(libpath)
        @test_skip "Rust library not found"
    else
        # RealGrid + Hankel QDHT, Kerr + Raman (N2 vibrational SDO, thg=true,
        # no plasma) — docs/dev/BACKLOG.md Phase D.4's radial-Raman follow-up. Same
        # geometry as test_native_radial_plasma.jl; :N2 vibration=true,
        # rotation=false gives a single-SDO CombinedRamanResponse eligible
        # for the native path (same criterion as test_native_raman.jl's
        # mode-averaged Phase 4 gate).
        gas = :N2; pres = 1.5; τ = 15e-15; λ0 = 800e-9
        w0 = 150e-6; energy = 6e-5; L = 0.02; R = 4e-3; N = 32

        grid = Grid.RealGrid(L, λ0, (150e-9, 2000e-9), 0.5e-12)
        q    = Hankel.QDHT(R, N, dim=2)

        dens0 = PhysData.density(gas, pres)
        densityfun(z) = dens0
        rr = Raman.raman_response(grid.to, gas; rotation=false, vibration=true)
        raman_resp = Nonlinear.RamanPolarField(grid.to, rr)
        responses = (Nonlinear.Kerr_field(PhysData.γ3_gas(gas)), raman_resp)
        linop   = LinearOps.make_const_linop(grid, q, PhysData.ref_index_fun(gas, pres))
        normfun = NonlinearRHS.const_norm_radial(grid, q, PhysData.ref_index_fun(gas, pres))
        inputs  = Fields.GaussGaussField(λ0=λ0, τfwhm=τ, energy=energy, w0=w0, propz=-0.1)

        Eω, transform, FT = with_logger(NullLogger()) do
            Amalthea.setup(grid, q, densityfun, normfun, responses, inputs)
        end

        @assert transform isa Amalthea.NonlinearRHS.TransRadial "Expected TransRadial"

        t0 = 0.0
        dt = 0.0005

        @testset "Single-step equivalence (radial + Raman, ~1e-7)" begin
            # Not 1e-12/1e-13 like the plasma/Kerr-only radial gates: Julia's
            # oracle `RamanPolarField` here has no `rust_handle` (constructed
            # directly, not via `Interface.makeresponse`), so it takes its
            # own FFT-convolution path (Nonlinear.jl:391-408), while the
            # resident native path always uses the O(Nt) exponential-ADE
            # integrator (`raman.rs`) — a genuine method difference, not a
            # summation-order floor. test_native_raman.jl's mode-averaged
            # Phase 4 gate used 1e-13 here, but never actually exercised this
            # floor: at that test's energy, Raman's single-step contribution
            # is ~2e-16 relative to Kerr (see that file's comment), i.e.
            # below FP noise. This radial geometry's non-vacuousness is
            # ~1e-3 (see below), large enough that the ADE-vs-FFT-convolution
            # difference (~1e-9, empirically) is actually visible.
            s_jl = PreconStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10)
            s_ru = RustNativeStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10)

            step!(s_jl)
            step!(s_ru)

            rel_step = norm(s_ru.yn - s_jl.yn) / norm(s_jl.yn)
            println("Radial+Raman single-step rel: ", rel_step)
            @test rel_step < 1e-7
        end

        @testset "Full-solve equivalence (radial + Raman, fixed dt)" begin
            s_jl = PreconStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10,
                                  max_dt=dt, min_dt=dt)
            s_ru = RustNativeStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10,
                                  max_dt=dt, min_dt=dt)

            solve(s_jl, L)
            solve(s_ru, L)

            rel_solve = norm(s_ru.yn - s_jl.yn) / norm(s_jl.yn)
            println("Radial+Raman full-solve rel: ", rel_solve)
            @test rel_solve < 1e-6
        end

        @testset "Non-vacuousness: Raman actually changes the result vs Raman-off" begin
            no_raman_transform = NonlinearRHS.TransRadial(grid, q, transform.FT,
                (Nonlinear.Kerr_field(PhysData.γ3_gas(gas)),), densityfun, normfun)
            s_jl_raman = PreconStepper(transform, linop, copy(Eω), t0, dt,
                                        rtol=1e-6, atol=1e-10, max_dt=dt, min_dt=dt)
            s_jl_noraman = PreconStepper(no_raman_transform, linop, copy(Eω), t0, dt,
                                          rtol=1e-6, atol=1e-10, max_dt=dt, min_dt=dt)
            solve(s_jl_raman, L)
            solve(s_jl_noraman, L)
            rel_raman_effect = norm(s_jl_raman.yn - s_jl_noraman.yn) / norm(s_jl_noraman.yn)
            println("Radial Raman-on vs Raman-off rel (Julia, non-vacuousness): ", rel_raman_effect)
            @test rel_raman_effect > 1e-6
        end
    end
end
