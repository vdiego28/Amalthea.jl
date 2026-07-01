using TestItems

@testitem "Native-Rust Phase 5 (modal, resident libcubature)" tags=[:rust] begin
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
        # RealGrid, constant-radius Marcatili HE1m modes, scalar (npol=1)
        # Kerr-only — Phase 5's first gate. Two modes (HE11 + HE12) so the
        # to_space! sum-over-modes and back-projection matmuls are genuinely
        # exercised (not left implicitly untested by a single-mode
        # shortcut) — see docs/native-port/MATH.md §3.3. `full=false` here
        # is not an artificial restriction: `Interface.needfull` selects
        # exactly this (the radial modal integral) for any all-HE,n=1 mode
        # collection, i.e. this is the common case.
        gas = :Ar; pres = 1.0; τ = 20e-15; λ0 = 800e-9
        a = 125e-6; energy = 5e-6; L = 0.1

        grid = Grid.RealGrid(L, λ0, (400e-9, 2000e-9), 0.5e-12)

        modes = (Capillary.MarcatiliMode(a, gas, pres; m=1),
                 Capillary.MarcatiliMode(a, gas, pres; m=2))
        @assert Modes.orthonormal(modes) "test modes must be orthonormal"

        dens0 = PhysData.density(gas, pres)
        densityfun(z) = dens0
        responses = (Nonlinear.Kerr_field(PhysData.γ3_gas(gas)),)
        linop = LinearOps.make_const_linop(grid, modes, grid.referenceλ)
        input = Fields.GaussField(λ0=λ0, τfwhm=τ, energy=energy)

        Eω, transform, FT = with_logger(NullLogger()) do
            Luna.setup(grid, densityfun, responses, input, modes, :y)
        end

        @assert transform isa Luna.NonlinearRHS.TransModal "Expected TransModal"
        @assert !transform.full "Expected full=false (radial modal integral)"

        t0 = 0.0
        dt = 0.001

        @testset "Single-step equivalence (modal, ~1e-10)" begin
            s_jl = PreconStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10)
            s_ru = RustNativeStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10)

            step!(s_jl)
            step!(s_ru)

            rel_step = norm(s_ru.yn - s_jl.yn) / norm(s_jl.yn)
            println("Modal single-step rel: ", rel_step)
            # Looser tier than Phases 1-4: node placement is bit-identical
            # (same libcubature binary — see MATH.md §3.3) but the mode-field
            # synthesis reuses diffraction.rs's j0, which agrees with
            # SpecialFunctions.besselj to ~1e-15 *absolute* (verified before
            # implementing) — not exactly bitwise, hence ~1e-10 not ~1e-13.
            @test rel_step < 1e-10
        end

        @testset "Full-solve equivalence (modal, fixed dt)" begin
            # Fixed step size (max_dt=min_dt=dt): see TESTING.md §3 — isolates
            # state-accumulation error from adaptive-path agreement.
            s_jl = PreconStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10,
                                  max_dt=dt, min_dt=dt)
            s_ru = RustNativeStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10,
                                  max_dt=dt, min_dt=dt)

            solve(s_jl, L)
            solve(s_ru, L)

            # Self-validating sanity check (see PORT_LOG 2026-07-01's Phase 4
            # lesson): confirm the multimode overlap actually moves energy
            # into HE12 over this propagation before trusting the
            # Rust-vs-Julia comparison — otherwise a test where both paths
            # trivially stay in mode 1 would prove nothing about the
            # to_space!/back-projection matmuls.
            he12_frac = sum(abs2, s_jl.yn[:, 2]) / sum(abs2, s_jl.yn)
            println("Fraction of energy transferred to HE12 (Julia, sanity check): ", he12_frac)
            @test he12_frac > 1e-6

            rel_solve = norm(s_ru.yn - s_jl.yn) / norm(s_jl.yn)
            println("Modal full-solve rel: ", rel_solve)
            @test rel_solve < 1e-6
        end
    end
end
