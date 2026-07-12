using TestItems

@testitem "Native-Rust Phase D.5 (free-space, z-dependent linop+normfun)" tags=[:rust] begin
    import Test: @test, @test_skip, @testset
    using Amalthea
    import Amalthea: Grid, NonlinearRHS, Fields, LinearOps, PhysData, Nonlinear
    using Amalthea.RK45: PreconStepper, RustNativeStepper, step!, solve
    import Logging: with_logger, NullLogger
    import LinearAlgebra: norm

    libpath = RK45._LIBLUNA_RUST_RK45
    if !isfile(libpath)
        @test_skip "Rust library not found"
    else
        # RealGrid free-space, two-point pressure-gradient gas cell —
        # docs/dev/BACKLOG.md Phase D.5. `LinearOps.make_linop_free_gradient` +
        # `NonlinearRHS.norm_free_gradient` (mirroring `Capillary.gradient`
        # for TransFree) give `n(ω;z) = sqrt(1+γ(λ(ω))·ρ(z))`, exact (not
        # approximate — confirmed against `PhysData.ref_index_fun`'s own
        # source before implementing). Same Nx≠Ny geometry as
        # test_native_free.jl to keep catching an axis-swap bug.
        gas = :Ar; L = 0.01; p0 = 0.5; p1 = 1.5
        λ0 = 800e-9; τ = 20e-15; w0 = 60e-6; energy = 1e-9
        Nx = 8; Ny = 6; R = 300e-6

        grid = Grid.RealGrid(L, λ0, (400e-9, 2000e-9), 0.5e-12)
        xygrid = Grid.FreeGrid(R, Nx, R, Ny)

        linop, densf = LinearOps.make_linop_free_gradient(grid, xygrid, gas, L, p0, p1)
        normfun = NonlinearRHS.norm_free_gradient(grid, xygrid, gas, densf)
        responses = (Nonlinear.Kerr_field(PhysData.γ3_gas(gas)),)
        inputs = Fields.GaussGaussField(λ0=λ0, τfwhm=τ, energy=energy, w0=w0)

        Eω, transform, FT = with_logger(NullLogger()) do
            Amalthea.setup(grid, xygrid, densf, normfun, responses, inputs)
        end

        @assert transform isa Amalthea.NonlinearRHS.TransFree "Expected TransFree"
        @assert linop isa Amalthea.LinearOps.ZDepLinopFree "Expected ZDepLinopFree"

        t0 = 0.0
        dt = 0.0005

        @testset "Single-step equivalence (free-space z-dependent, ~1e-9)" begin
            # Julia's oracle β1(z) comes from `PhysData.dispersion_func`'s
            # adaptive finite difference (via `LinearOps.make_linop`'s own
            # `nfunλ`), while Rust's is the exact closed form (BigFloat
            # derivative of γ at construction — see `LinearOps.ZDepLinopFree`'s
            # doc). Unlike Phase 7's Marcatili case (~1e-4, FD noise from a
            # much more complex neff), this index law is simple enough that
            # the FD/closed-form gap is empirically ~1e-11 here.
            s_jl = PreconStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10)
            s_ru = RustNativeStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10,
                                      flength=L)

            step!(s_jl)
            step!(s_ru)

            rel_step = norm(s_ru.yn - s_jl.yn) / norm(s_jl.yn)
            println("Free-space z-dependent single-step rel: ", rel_step)
            @test rel_step < 1e-9
        end

        @testset "Full-solve equivalence (free-space z-dependent, fixed dt)" begin
            s_jl = PreconStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10,
                                  max_dt=dt, min_dt=dt)
            s_ru = RustNativeStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10,
                                      max_dt=dt, min_dt=dt, flength=L)

            solve(s_jl, L)
            solve(s_ru, L)

            rel_solve = norm(s_ru.yn - s_jl.yn) / norm(s_jl.yn)
            println("Free-space z-dependent full-solve rel: ", rel_solve)
            @test rel_solve < 1e-6
        end

        @testset "Non-vacuousness: the pressure gradient actually changes the result vs constant p0" begin
            # Julia-only comparison against a constant-density (p1=p0) cell
            # over the same length — confirms the z-dependence is doing
            # something, not just reproducing the p0 constant-density case.
            linop_const, densf_const = LinearOps.make_linop_free_gradient(grid, xygrid, gas, L, p0, p0)
            normfun_const = NonlinearRHS.norm_free_gradient(grid, xygrid, gas, densf_const)
            Eω_const, transform_const, _ = with_logger(NullLogger()) do
                Amalthea.setup(grid, xygrid, densf_const, normfun_const, responses, inputs)
            end
            s_jl_grad = PreconStepper(transform, linop, copy(Eω), t0, dt,
                                       rtol=1e-6, atol=1e-10, max_dt=dt, min_dt=dt)
            s_jl_const = PreconStepper(transform_const, linop_const, copy(Eω_const), t0, dt,
                                        rtol=1e-6, atol=1e-10, max_dt=dt, min_dt=dt)
            solve(s_jl_grad, L)
            solve(s_jl_const, L)
            rel_grad_effect = norm(s_jl_grad.yn - s_jl_const.yn) / norm(s_jl_const.yn)
            println("Free-space pressure-gradient vs constant-p0 rel (Julia, non-vacuousness): ",
                    rel_grad_effect)
            @test rel_grad_effect > 1e-6
        end
    end
end
