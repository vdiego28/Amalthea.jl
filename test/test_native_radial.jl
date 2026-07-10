using TestItems

@testitem "Native-Rust Phase 3 (radial, resident QDHT)" tags=[:rust] begin
    import Test: @test, @test_skip, @testset
    using Luna
    import Luna: Grid, NonlinearRHS, Fields, LinearOps, PhysData, Nonlinear
    using Luna.RK45: PreconStepper, RustNativeStepper, step!, solve
    import Hankel
    import Logging: with_logger, NullLogger
    import LinearAlgebra: norm

    libpath = RK45._LIBLUNA_RUST_RK45
    if !isfile(libpath)
        @test_skip "Rust library not found"
    else
        # RealGrid + Hankel QDHT, scalar Kerr only (no plasma, no shot-noise) —
        # Phase 3's first gate. See docs/native-port/MATH.md §3.2.
        gas = :Ar; pres = 1.2; τ = 20e-15; λ0 = 800e-9
        w0 = 40e-6; energy = 1e-12; L = 0.05; R = 4e-3; N = 32

        grid = Grid.RealGrid(L, λ0, (400e-9, 2000e-9), 0.2e-12)
        q    = Hankel.QDHT(R, N, dim=2)

        dens0 = PhysData.density(gas, pres)
        densityfun(z) = dens0
        responses = (Nonlinear.Kerr_field(PhysData.γ3_gas(gas)),)
        linop   = LinearOps.make_const_linop(grid, q, PhysData.ref_index_fun(gas, pres))
        normfun = NonlinearRHS.const_norm_radial(grid, q, PhysData.ref_index_fun(gas, pres))
        inputs  = Fields.GaussGaussField(λ0=λ0, τfwhm=τ, energy=energy, w0=w0, propz=-0.15)

        Eω, transform, FT = with_logger(NullLogger()) do
            Luna.setup(grid, q, densityfun, normfun, responses, inputs)
        end

        @assert transform isa Luna.NonlinearRHS.TransRadial "Expected TransRadial"

        t0 = 0.0
        dt = 0.001

        @testset "Single-step equivalence (radial, ~1e-13)" begin
            s_jl = PreconStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10)
            s_ru = RustNativeStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10)

            step!(s_jl)
            step!(s_ru)

            rel_step = norm(s_ru.yn - s_jl.yn) / norm(s_jl.yn)
            println("Radial single-step rel: ", rel_step)
            # MATH.md §3.2 flags a ~1e-13 BLAS-vs-Rayon QDHT floor (applied
            # twice per RHS) as the expected tier; achieved ~1e-17 in practice
            # at this problem size, but assert at the documented tier so the
            # test doesn't silently mask a QDHT-floor regression as unrelated.
            @test rel_step < 1e-13
        end

        @testset "Full-solve equivalence (radial, fixed dt)" begin
            # Fixed step size (max_dt=min_dt=dt): see test_native_phase1.jl /
            # test_native_phase2.jl and docs/native-port/TESTING.md §3 — the
            # adaptive PI controller's near-cancellation error estimate makes
            # raw adaptive-dt agreement an unreliable equivalence signal.
            # Forcing an identical step-size sequence isolates genuine
            # multi-step state-accumulation error.
            s_jl = PreconStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10,
                                  max_dt=dt, min_dt=dt)
            s_ru = RustNativeStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10,
                                  max_dt=dt, min_dt=dt)

            solve(s_jl, L)
            solve(s_ru, L)

            rel_solve = norm(s_ru.yn - s_jl.yn) / norm(s_jl.yn)
            println("Radial full-solve rel: ", rel_solve)
            @test rel_solve < 1e-6
        end
    end
end
