using TestItems

@testitem "Native-Rust Phase D.3 (free-space, EnvGrid, 3-D FFT)" tags=[:rust] begin
    import Test: @test, @test_skip, @testset
    using Luna
    import Luna: Grid, NonlinearRHS, Fields, LinearOps, PhysData, Nonlinear
    using Luna.RK45: PreconStepper, RustNativeStepper, step!, solve
    import Logging: with_logger, NullLogger
    import LinearAlgebra: norm

    libpath = RK45._LIBLUNA_RUST_RK45
    if !isfile(libpath)
        @test_skip "Rust library not found"
    else
        # EnvGrid + joint 3-D c2c FFT, Kerr_env only — docs/dev/BACKLOG.md Phase D.3 /
        # MATH.md §3.4's deferred EnvGrid-free-space follow-up. Same geometry
        # as test_native_free.jl (Phase 6, RealGrid), but EnvGrid + Kerr_env
        # instead of RealGrid + Kerr_field. Nx != Ny deliberately, same
        # rationale as Phase 6's test (a square grid with a radially-symmetric
        # input can't detect a swapped-axis bug in the 3-D transform).
        gas = :Ar; pres = 1.0; τ = 20e-15; λ0 = 800e-9
        w0 = 60e-6; energy = 1e-9; L = 0.01; R = 300e-6; Nx = 8; Ny = 6

        grid = Grid.EnvGrid(L, λ0, (400e-9, 2000e-9), 0.5e-12)
        xygrid = Grid.FreeGrid(R, Nx, R, Ny)

        dens0 = PhysData.density(gas, pres)
        densityfun(z) = dens0
        responses = (Nonlinear.Kerr_env(PhysData.γ3_gas(gas)),)
        linop = LinearOps.make_const_linop(grid, xygrid, PhysData.ref_index_fun(gas, pres))
        normfun = NonlinearRHS.const_norm_free(grid, xygrid, PhysData.ref_index_fun(gas, pres))
        inputs = Fields.GaussGaussField(λ0=λ0, τfwhm=τ, energy=energy, w0=w0)

        Eω, transform, FT = with_logger(NullLogger()) do
            Luna.setup(grid, xygrid, densityfun, normfun, responses, inputs)
        end

        @assert transform isa Luna.NonlinearRHS.TransFree "Expected TransFree"

        t0 = 0.0
        dt = 0.0005

        @testset "Single-step equivalence (free-space EnvGrid, ~1e-13)" begin
            s_jl = PreconStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10)
            s_ru = RustNativeStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10)

            step!(s_jl)
            step!(s_ru)

            rel_step = norm(s_ru.yn - s_jl.yn) / norm(s_jl.yn)
            println("Free-space EnvGrid single-step rel: ", rel_step)
            @test rel_step < 1e-13
        end

        @testset "Full-solve equivalence (free-space EnvGrid, fixed dt)" begin
            # Fixed step size (max_dt=min_dt=dt): see TESTING.md §3 — isolates
            # state-accumulation error from adaptive-path agreement.
            s_jl = PreconStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10,
                                  max_dt=dt, min_dt=dt)
            s_ru = RustNativeStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10,
                                  max_dt=dt, min_dt=dt)

            solve(s_jl, L)
            solve(s_ru, L)

            rel_solve = norm(s_ru.yn - s_jl.yn) / norm(s_jl.yn)
            println("Free-space EnvGrid full-solve rel: ", rel_solve)
            @test rel_solve < 1e-6
        end

        @testset "Non-vacuousness: Kerr actually changes the result vs Kerr-off" begin
            # Same rationale as test_native_radial_env.jl's Phase D.1
            # non-vacuousness testset: a stronger, independent field, compared
            # Julia-only (Kerr-on vs Kerr-off), rather than folding this into
            # the equivalence tests above (whose weak field keeps Kerr's own
            # contribution below the FP-noise floor by design).
            energy_strong = 2e-7
            inputs_strong = Fields.GaussGaussField(λ0=λ0, τfwhm=τ, energy=energy_strong, w0=w0)
            Eω_strong, transform_strong, _ = with_logger(NullLogger()) do
                Luna.setup(grid, xygrid, densityfun, normfun, responses, inputs_strong)
            end
            no_kerr_transform = NonlinearRHS.TransFree(ComplexF64, transform_strong.scale, grid,
                xygrid, transform_strong.FT, (), densityfun, normfun)
            s_jl_kerr = PreconStepper(transform_strong, linop, copy(Eω_strong), t0, dt,
                                       rtol=1e-6, atol=1e-10, max_dt=dt, min_dt=dt)
            s_jl_nokerr = PreconStepper(no_kerr_transform, linop, copy(Eω_strong), t0, dt,
                                         rtol=1e-6, atol=1e-10, max_dt=dt, min_dt=dt)
            solve(s_jl_kerr, L)
            solve(s_jl_nokerr, L)
            rel_kerr_effect = norm(s_jl_kerr.yn - s_jl_nokerr.yn) / norm(s_jl_nokerr.yn)
            println("Free-space EnvGrid Kerr-on vs Kerr-off rel (Julia, non-vacuousness): ",
                    rel_kerr_effect)
            @test rel_kerr_effect > 1e-6
        end
    end
end
