using TestItems

# Julia ↔ Rust QDHT batch-transform equivalence test.
#
# Tagged :rust so it runs in the `rust` CI group and is automatically skipped
# (not failed) when the Rust shared library has not been built.
@testitem "Rust QDHT equivalence" tags=[:rust] begin
    import Test: @test, @testset
    using Amalthea
    import Amalthea: Grid, NonlinearRHS, Fields, Output, LinearOps, PhysData, Nonlinear, Ionisation
    import Hankel
    import LinearAlgebra: mul!, ldiv!, norm
    import Logging: with_logger, NullLogger

    # ── skip guard ────────────────────────────────────────────────────────────
    libname = if Sys.iswindows(); "luna_rust.dll"
              elseif Sys.isapple(); "libluna_rust.dylib"
              else; "libluna_rust.so"; end
    libpath = joinpath(@__DIR__, "..", "luna-rust", "target", "release", libname)
    if !isfile(libpath)
        @warn "Skipping Rust QDHT test: shared library not found at $libpath. " *
              "Build with `cargo build --release` in luna-rust/ (or `]build Amalthea`)."
        return
    end

    # ─────────────────────────────────────────────────────────────────────────
    # Unit-level: compare _qdht_mul! / _qdht_ldiv! against Julia mul!/ldiv!
    # ─────────────────────────────────────────────────────────────────────────
    @testset "QDHT mul! / ldiv! unit equivalence" begin
        R = 4e-3; N = 64
        q = Hankel.QDHT(R, N, dim=2)
        n_time = 256

        # Build Rust handle
        h = withenv("LUNA_USE_RUST_QDHT" => "1") do
            with_logger(NullLogger()) do
                NonlinearRHS._make_rust_qdht_handle(q, n_time)
            end
        end
        @test !isnothing(h)

        @testset "Real (Float64) mul!" begin
            A_ref  = randn(n_time, N)
            A_rust = copy(A_ref)
            mul!(A_ref, q, A_ref)
            NonlinearRHS._qdht_mul!(A_rust, h, q)
            @test A_rust ≈ A_ref rtol=1e-13
        end

        @testset "Real (Float64) ldiv!" begin
            A_ref  = randn(n_time, N)
            A_rust = copy(A_ref)
            ldiv!(A_ref, q, A_ref)
            NonlinearRHS._qdht_ldiv!(A_rust, h, q)
            @test A_rust ≈ A_ref rtol=1e-13
        end

        @testset "Complex (ComplexF64) mul!" begin
            A_ref  = randn(ComplexF64, n_time, N)
            A_rust = copy(A_ref)
            mul!(A_ref, q, A_ref)
            NonlinearRHS._qdht_mul!(A_rust, h, q)
            @test A_rust ≈ A_ref rtol=1e-13
        end

        @testset "Complex (ComplexF64) ldiv!" begin
            A_ref  = randn(ComplexF64, n_time, N)
            A_rust = copy(A_ref)
            ldiv!(A_ref, q, A_ref)
            NonlinearRHS._qdht_ldiv!(A_rust, h, q)
            @test A_rust ≈ A_ref rtol=1e-13
        end

        @testset "Round-trip: ldiv(mul(A)) ≈ A" begin
            A_orig = randn(n_time, N)
            A_rt   = copy(A_orig)
            NonlinearRHS._qdht_mul!(A_rt, h, q)
            NonlinearRHS._qdht_ldiv!(A_rt, h, q)
            # T² ≈ I for the QDHT (involutory) to ~1e-8; BLAS vs Rayon summation order
            # means we can lose a few ULPs per operation, so use rtol=1e-8 here.
            @test A_rt ≈ A_orig rtol=1e-8
        end
    end

    # ─────────────────────────────────────────────────────────────────────────
    # Integration: run a full radial free-space simulation with and without Rust,
    # compare the output electric field at the final z position.
    # ─────────────────────────────────────────────────────────────────────────
    @testset "Full radial simulation equivalence" begin
        gas = :Ar; pres = 1.2; τ = 20e-15; λ0 = 800e-9
        w0 = 40e-6; energy = 1e-12; L = 0.3; R = 4e-3; N = 64

        grid = Grid.RealGrid(L, λ0, (400e-9, 2000e-9), 0.2e-12)
        q    = Hankel.QDHT(R, N, dim=2)

        dens0      = PhysData.density(gas, pres)
        densityfun(z) = dens0
        ionpot  = PhysData.ionisation_potential(gas)
        ionrate = Ionisation.IonRatePPTCached(gas, λ0)
        responses = (Nonlinear.Kerr_field(PhysData.γ3_gas(gas)),
                     Nonlinear.PlasmaCumtrapz(grid.to, grid.to, ionrate, ionpot))
        linop   = LinearOps.make_const_linop(grid, q, PhysData.ref_index_fun(gas, pres))
        normfun = NonlinearRHS.const_norm_radial(grid, q, PhysData.ref_index_fun(gas, pres))
        inputs  = Fields.GaussGaussField(λ0=λ0, τfwhm=τ, energy=energy, w0=w0, propz=-0.15)

        # ── Julia path ────────────────────────────────────────────────────────
        Eω_julia, tr_julia, FT_julia = with_logger(NullLogger()) do
            Amalthea.setup(grid, q, densityfun, normfun, responses, inputs)
        end
        out_julia = Output.MemoryOutput(0, grid.zmax, 2)
        Amalthea.run(Eω_julia, grid, linop, tr_julia, FT_julia, out_julia)
        Eω_final_julia = out_julia.data["Eω"][:, :, end]

        # ── Rust path ─────────────────────────────────────────────────────────
        Eω_rust, tr_rust, FT_rust = withenv("LUNA_USE_RUST_QDHT" => "1") do
            with_logger(NullLogger()) do
                Amalthea.setup(grid, q, densityfun, normfun, responses, inputs)
            end
        end
        out_rust = Output.MemoryOutput(0, grid.zmax, 2)
        Amalthea.run(Eω_rust, grid, linop, tr_rust, FT_rust, out_rust)
        Eω_final_rust = out_rust.data["Eω"][:, :, end]

        @test tr_rust.rust_ht isa NonlinearRHS.RustQdhtHandle

        rel_err = norm(Eω_final_rust - Eω_final_julia) / norm(Eω_final_julia)
        @test rel_err < 1e-10
    end

    # ─────────────────────────────────────────────────────────────────────────
    # Toggle-off: LUNA_USE_RUST_QDHT=0 → rust_ht is Nothing
    # ─────────────────────────────────────────────────────────────────────────
    @testset "Toggle-off leaves rust_ht as Nothing" begin
        R = 2e-3; N = 32
        q    = Hankel.QDHT(R, N, dim=2)
        grid = Grid.RealGrid(0.1, 800e-9, (400e-9, 2000e-9), 0.1e-12)
        h = with_logger(NullLogger()) do
            NonlinearRHS._make_rust_qdht_handle(q, length(grid.to))
        end
        @test isnothing(h)
    end
end
