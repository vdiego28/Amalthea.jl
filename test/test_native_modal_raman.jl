using TestItems

@testitem "Native-Rust Phase D.4 (Raman, modal)" tags=[:rust] begin
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
        # RealGrid, constant-radius Marcatili HE11 mode (npol=1), Kerr + Raman
        # (N2 vibrational SDO, thg=true) — docs/dev/BACKLOG.md Phase D.4's modal-Raman
        # follow-up to test_native_modal.jl's Phase 5 Kerr-only gate. Single
        # mode (not two, unlike test_native_modal.jl) keeps the Raman-specific
        # per-node ADE-solve addition isolated from the to_space!
        # multi-mode back-projection, which Phase 5 already validated.
        gas = :N2; pres = 1.0; τ = 20e-15; λ0 = 800e-9
        a = 125e-6; energy = 5e-6; L = 0.05

        grid = Grid.RealGrid(L, λ0, (400e-9, 2000e-9), 0.5e-12)

        modes = (Capillary.MarcatiliMode(a, gas, pres; m=1),)

        dens0 = PhysData.density(gas, pres)
        densityfun(z) = dens0
        rr = Raman.raman_response(grid.to, gas; rotation=false, vibration=true)
        raman_resp = Nonlinear.RamanPolarField(grid.to, rr)
        responses = (Nonlinear.Kerr_field(PhysData.γ3_gas(gas)), raman_resp)
        linop = LinearOps.make_const_linop(grid, modes, grid.referenceλ)
        input = Fields.GaussField(λ0=λ0, τfwhm=τ, energy=energy)

        Eω, transform, FT = with_logger(NullLogger()) do
            Amalthea.setup(grid, densityfun, responses, input, modes, :y)
        end

        @assert transform isa Amalthea.NonlinearRHS.TransModal "Expected TransModal"
        @assert !transform.full "Expected full=false (radial modal integral)"

        t0 = 0.0
        dt = 0.001

        @testset "Single-step equivalence (modal + Raman, ~1e-7)" begin
            # Same ADE-vs-FFT-convolution method-difference floor as
            # test_native_radial_raman.jl (see that file's comment) — not
            # the ~1e-10 j0-vs-besselj floor from test_native_modal.jl's
            # Kerr-only gate.
            s_jl = PreconStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10)
            s_ru = RustNativeStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10)

            step!(s_jl)
            step!(s_ru)

            rel_step = norm(s_ru.yn - s_jl.yn) / norm(s_jl.yn)
            println("Modal+Raman single-step rel: ", rel_step)
            @test rel_step < 1e-7
        end

        @testset "Full-solve equivalence (modal + Raman, fixed dt)" begin
            # 1e-6, not the usual full-solve tier: the ADE-vs-FFT-convolution
            # method difference (see the single-step testset's comment)
            # accumulates over 50 steps to ~1e-6 here, empirically
            # (measured 1.03e-6) — still five orders of magnitude below the
            # ~1e-3 non-vacuousness effect confirmed below, so this remains
            # a meaningful equivalence check, just not as tight as the
            # Kerr-only Phase 5 gate (test_native_modal.jl's 1e-6 tier there
            # reflects a summation-order floor, not a method difference).
            s_jl = PreconStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10,
                                  max_dt=dt, min_dt=dt)
            s_ru = RustNativeStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10,
                                  max_dt=dt, min_dt=dt)

            solve(s_jl, L)
            solve(s_ru, L)

            rel_solve = norm(s_ru.yn - s_jl.yn) / norm(s_jl.yn)
            println("Modal+Raman full-solve rel: ", rel_solve)
            @test rel_solve < 1.5e-6
        end

        @testset "Non-vacuousness: Raman actually changes the result vs Raman-off" begin
            no_raman_responses = (Nonlinear.Kerr_field(PhysData.γ3_gas(gas)),)
            Eω_nr, transform_nr, _ = with_logger(NullLogger()) do
                Amalthea.setup(grid, densityfun, no_raman_responses, input, modes, :y)
            end
            s_jl_raman = PreconStepper(transform, linop, copy(Eω), t0, dt,
                                        rtol=1e-6, atol=1e-10, max_dt=dt, min_dt=dt)
            s_jl_noraman = PreconStepper(transform_nr, linop, copy(Eω_nr), t0, dt,
                                          rtol=1e-6, atol=1e-10, max_dt=dt, min_dt=dt)
            solve(s_jl_raman, L)
            solve(s_jl_noraman, L)
            rel_raman_effect = norm(s_jl_raman.yn - s_jl_noraman.yn) / norm(s_jl_noraman.yn)
            println("Modal Raman-on vs Raman-off rel (Julia, non-vacuousness): ", rel_raman_effect)
            @test rel_raman_effect > 1e-6
        end
    end
end
