using TestItems

@testitem "Native-Rust Kerr thg-variant guard (Kerr_field_nothg/Kerr_env_thg rejected)" tags=[:rust] begin
    import Test: @test, @test_skip, @testset
    using Luna
    import Luna: Grid, Fields, LinearOps, PhysData, Nonlinear, Capillary
    using Luna.RK45: PreconStepper, RustNativeStepper, NativeIneligible, step!, solve
    import Logging: with_logger, NullLogger
    import LinearAlgebra: norm

    libpath = RK45._LIBLUNA_RUST_RK45
    if !isfile(libpath)
        @test_skip "Rust library not found"
    else
        # native.rs's mode-averaged/radial/modal/free-space RHS kernels pick
        # RealGrid-vs-EnvGrid Kerr formulas purely from `sim.is_real` — there
        # is no FFI-level THG flag. A naive "does this response have a field
        # named γ3" eligibility scan can't tell `Kerr_field`/`Kerr_env` (the
        # two formulas native.rs implements) apart from `Kerr_field_nothg`/
        # `Kerr_env_thg` (which flip the THG treatment for the same grid
        # type) since both variants also capture a γ3 field. Passing either
        # nothg/thg variant used to silently construct a `RustNativeStepper`
        # that ran with the WRONG THG treatment instead of falling back to
        # Julia. `RK45._is_plain_kerr_resp` closes this gap; this test
        # guards the mode-averaged path (the cheapest geometry to exercise)
        # against a regression, and also re-confirms the plain-Kerr case
        # still natively agrees with Julia.
        gas = :Ar; pres = 1.0; τ = 20e-15; λ0 = 800e-9
        a = 125e-6; energy = 5e-6; L = 0.01
        modes = (Capillary.MarcatiliMode(a, gas, pres; kind=:HE, n=1, m=1),)
        input = Fields.GaussField(λ0=λ0, τfwhm=τ, energy=energy)
        densityfun(z) = PhysData.density(gas, pres)
        γ3 = PhysData.γ3_gas(gas)
        t0 = 0.0
        dt = 0.001

        @testset "RealGrid, thg=false (Kerr_field_nothg) rejected" begin
            grid = Grid.RealGrid(L, λ0, (400e-9, 2000e-9), 0.5e-12)
            linop = LinearOps.make_const_linop(grid, modes, grid.referenceλ)
            responses = (Nonlinear.Kerr_field_nothg(γ3, length(grid.to)),)
            Eω, transform, FT = with_logger(NullLogger()) do
                Luna.setup(grid, densityfun, responses, input, modes, :y)
            end
            @test_throws NativeIneligible RustNativeStepper(transform, linop, copy(Eω), t0, dt,
                                      rtol=1e-6, atol=1e-10, max_dt=dt, min_dt=dt)
        end

        @testset "EnvGrid, thg=true (Kerr_env_thg) rejected" begin
            grid = Grid.EnvGrid(L, λ0, (400e-9, 2000e-9), 0.5e-12; thg=true)
            linop = LinearOps.make_const_linop(grid, modes, grid.referenceλ)
            ω0 = 2π * PhysData.c / λ0
            responses = (Nonlinear.Kerr_env_thg(γ3, ω0, grid.to),)
            Eω, transform, FT = with_logger(NullLogger()) do
                Luna.setup(grid, densityfun, responses, input, modes, :y)
            end
            @test_throws NativeIneligible RustNativeStepper(transform, linop, copy(Eω), t0, dt,
                                      rtol=1e-6, atol=1e-10, max_dt=dt, min_dt=dt)
        end

        @testset "RealGrid, default thg (Kerr_field) still natively eligible" begin
            grid = Grid.RealGrid(L, λ0, (400e-9, 2000e-9), 0.5e-12)
            linop = LinearOps.make_const_linop(grid, modes, grid.referenceλ)
            responses = (Nonlinear.Kerr_field(γ3),)
            Eω, transform, FT = with_logger(NullLogger()) do
                Luna.setup(grid, densityfun, responses, input, modes, :y)
            end
            s_jl = PreconStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10,
                                  max_dt=dt, min_dt=dt)
            s_ru = RustNativeStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10,
                                      max_dt=dt, min_dt=dt)
            solve(s_jl, L)
            solve(s_ru, L)
            rel = norm(s_ru.yn - s_jl.yn) / norm(s_jl.yn)
            @test rel < 1e-9
        end

        @testset "EnvGrid, default thg (Kerr_env) still natively eligible" begin
            grid = Grid.EnvGrid(L, λ0, (400e-9, 2000e-9), 0.5e-12)
            linop = LinearOps.make_const_linop(grid, modes, grid.referenceλ)
            responses = (Nonlinear.Kerr_env(γ3),)
            Eω, transform, FT = with_logger(NullLogger()) do
                Luna.setup(grid, densityfun, responses, input, modes, :y)
            end
            s_jl = PreconStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10,
                                  max_dt=dt, min_dt=dt)
            s_ru = RustNativeStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10,
                                      max_dt=dt, min_dt=dt)
            solve(s_jl, L)
            solve(s_ru, L)
            rel = norm(s_ru.yn - s_jl.yn) / norm(s_jl.yn)
            @test rel < 1e-9
        end
    end
end
