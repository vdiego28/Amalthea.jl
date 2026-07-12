using TestItems

@testitem "Native-Rust Phase I item 6 (plasma, free-space)" tags=[:rust] begin
    import Test: @test, @test_skip, @testset
    using Amalthea
    import Amalthea: Grid, NonlinearRHS, Fields, LinearOps, PhysData, Nonlinear, Ionisation
    using Amalthea.RK45: PreconStepper, RustNativeStepper, step!, solve
    import Logging: with_logger, NullLogger
    import LinearAlgebra: norm

    libpath = RK45._LIBLUNA_RUST_RK45
    if !isfile(libpath)
        @test_skip "Rust library not found"
    else
        # RealGrid + joint 3-D FFT, Kerr + plasma (PPT) — docs/dev/BACKLOG.md Phase I
        # item 6: free-space gains plasma the same way radial did in Phase
        # D.2 (`apply_plasma_free` in native.rs mirrors `apply_plasma_radial`
        # per (y,x) column instead of per r-column, since `TransFree`'s
        # `Et_to_Pt!` dispatches each response over `CartesianIndices((Ny,Nx))`
        # exactly like `TransRadial` dispatches over r). Same w0/energy/gas
        # regime as test_native_radial_plasma.jl (chosen so PPT ionisation has
        # a real, non-chaotic effect); Nx != Ny kept deliberately (see
        # test_native_free.jl's comment on why a square grid can't detect a
        # swapped-axis bug).
        gas = :Ar; pres = 1.5; τ = 15e-15; λ0 = 800e-9
        w0 = 150e-6; energy = 6e-5; L = 0.02; R = 1.5e-3; Nx = 10; Ny = 8

        grid = Grid.RealGrid(L, λ0, (150e-9, 2000e-9), 0.5e-12)
        xygrid = Grid.FreeGrid(R, Nx, R, Ny)

        dens0 = PhysData.density(gas, pres)
        densityfun(z) = dens0
        ionpot = PhysData.ionisation_potential(gas)
        ionrate = withenv("LUNA_USE_RUST_IONISATION" => "1") do
            Ionisation.IonRatePPTCached(gas, λ0)
        end
        plasma_resp = Nonlinear.PlasmaCumtrapz(grid.to, grid.to, ionrate, ionpot)
        responses = (Nonlinear.Kerr_field(PhysData.γ3_gas(gas)), plasma_resp)
        linop   = LinearOps.make_const_linop(grid, xygrid, PhysData.ref_index_fun(gas, pres))
        normfun = NonlinearRHS.const_norm_free(grid, xygrid, PhysData.ref_index_fun(gas, pres))
        inputs  = Fields.GaussGaussField(λ0=λ0, τfwhm=τ, energy=energy, w0=w0, propz=-0.1)

        Eω, transform, FT = with_logger(NullLogger()) do
            Amalthea.setup(grid, xygrid, densityfun, normfun, responses, inputs)
        end

        @assert transform isa Amalthea.NonlinearRHS.TransFree "Expected TransFree"

        t0 = 0.0
        dt = 0.0005

        @testset "Single-step equivalence (free-space + plasma, ~1e-12)" begin
            s_jl = PreconStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10)
            s_ru = withenv("LUNA_USE_RUST_NATIVE" => "1",
                           "LUNA_USE_RUST_IONISATION" => "1") do
                RustNativeStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10)
            end

            step!(s_jl)
            step!(s_ru)

            rel_step = norm(s_ru.yn - s_jl.yn) / norm(s_jl.yn)
            println("Free-space+plasma single-step rel: ", rel_step)
            @test rel_step < 1e-12
        end

        @testset "Full-solve equivalence (free-space + plasma, fixed dt)" begin
            s_jl = PreconStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10,
                                  max_dt=dt, min_dt=dt)
            s_ru = withenv("LUNA_USE_RUST_NATIVE" => "1",
                           "LUNA_USE_RUST_IONISATION" => "1") do
                RustNativeStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10,
                                  max_dt=dt, min_dt=dt)
            end

            solve(s_jl, L)
            solve(s_ru, L)

            rel_solve = norm(s_ru.yn - s_jl.yn) / norm(s_jl.yn)
            println("Free-space+plasma full-solve rel: ", rel_solve)
            @test rel_solve < 1e-6
        end

        @testset "Non-vacuousness: plasma actually changes the result vs plasma-off" begin
            # Same reasoning as test_native_radial_plasma.jl: PPT's extreme
            # field-sensitivity makes the equivalence-test energy and the
            # non-vacuousness threshold pull in opposite directions, so this
            # uses an independent, much stronger, Julia-only-compared field.
            energy_strong = 3e-4
            inputs_strong = Fields.GaussGaussField(λ0=λ0, τfwhm=τ, energy=energy_strong,
                                                    w0=w0, propz=-0.1)
            Eω_strong, transform_strong, _ = with_logger(NullLogger()) do
                Amalthea.setup(grid, xygrid, densityfun, normfun, responses, inputs_strong)
            end
            no_plasma_transform = NonlinearRHS.TransFree(grid, xygrid, transform_strong.FT,
                (Nonlinear.Kerr_field(PhysData.γ3_gas(gas)),), densityfun, normfun)
            s_jl_plasma = PreconStepper(transform_strong, linop, copy(Eω_strong), t0, dt,
                                         rtol=1e-6, atol=1e-10, max_dt=dt, min_dt=dt)
            s_jl_noplasma = PreconStepper(no_plasma_transform, linop, copy(Eω_strong), t0, dt,
                                           rtol=1e-6, atol=1e-10, max_dt=dt, min_dt=dt)
            solve(s_jl_plasma, L)
            solve(s_jl_noplasma, L)
            rel_plasma_effect = norm(s_jl_plasma.yn - s_jl_noplasma.yn) / norm(s_jl_noplasma.yn)
            println("Free-space plasma-on vs plasma-off rel (Julia, non-vacuousness): ", rel_plasma_effect)
            @test rel_plasma_effect > 1e-6

            # Triangulate (docs/dev/BACKLOG.md Phase I item 1/3 postmortem lesson):
            # native vs Julia AT the plasma-matters energy, and native vs
            # Julia's plasma-OFF result (must NOT match) — a weak-field-only
            # equivalence check would let a "native silently contributes
            # zero" bug hide exactly like the Phase I preamble's density bug
            # did for mode-averaged/radial plasma.
            s_ru_strong = withenv("LUNA_USE_RUST_NATIVE" => "1",
                           "LUNA_USE_RUST_IONISATION" => "1") do
                RustNativeStepper(transform_strong, linop, copy(Eω_strong), t0, dt,
                                   rtol=1e-6, atol=1e-10, max_dt=dt, min_dt=dt)
            end
            solve(s_ru_strong, L)
            rel_native_vs_jl = norm(s_ru_strong.yn - s_jl_plasma.yn) / norm(s_jl_plasma.yn)
            rel_native_vs_noplasma = norm(s_ru_strong.yn - s_jl_noplasma.yn) / norm(s_jl_noplasma.yn)
            println("Free-space native vs Julia (both plasma-on, strong energy): ", rel_native_vs_jl)
            println("Free-space native vs Julia no-plasma (must NOT match — regression guard): ",
                     rel_native_vs_noplasma)
            @test rel_native_vs_jl < 1e-6
            @test rel_native_vs_noplasma > 1e-6
        end
    end
end
