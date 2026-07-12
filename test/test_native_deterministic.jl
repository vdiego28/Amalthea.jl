using TestItems

@testitem "Native deterministic mode (LUNA_NATIVE_DETERMINISTIC)" tags=[:rust] begin
    import Test: @test, @test_skip, @testset
    using Amalthea
    import Amalthea: Grid, NonlinearRHS, Fields, LinearOps, PhysData, Nonlinear
    using Amalthea.RK45: RustNativeStepper, solve
    import Hankel
    import Logging: with_logger, NullLogger

    libpath = RK45._LIBLUNA_RUST_RK45
    if !isfile(libpath)
        @test_skip "Rust library not found"
    else
        # Radial geometry (resident QDHT) — the only backend surface
        # `LUNA_NATIVE_DETERMINISTIC` currently affects (docs/dev/BACKLOG.md S5.2):
        # it forces the native path's QDHT to skip the opt-in BLAS-3
        # `dgemm` path and always use the row-parallel Rayon fallback.
        #
        # The subtlety this test exists to catch: `crate::blas::BLAS_API`
        # is a *process-global* `OnceLock` in the Rust library, populated
        # only by `NonlinearRHS._init_rust_qdht_blas` (the older per-kernel
        # `LUNA_USE_RUST_QDHT` wiring). The native-port radial path never
        # calls that itself — so in a process where no per-kernel Rust QDHT
        # handle has ever been built, the native path *always* takes the
        # Rayon fallback regardless of `deterministic`, and a naive
        # "two runs bit-identical" test would pass whether or not the flag
        # does anything at all. This test deliberately contaminates the
        # process-global BLAS state first (mirroring a real session that
        # mixes the per-kernel and native-port Rust paths), then checks
        # that `deterministic=true` measurably changes which codepath the
        # native QDHT takes.
        gas = :Ar; pres = 1.2; τ = 20e-15; λ0 = 800e-9
        w0 = 40e-6; energy = 1e-12; L = 0.01; R = 4e-3; N = 32

        grid = Grid.RealGrid(L, λ0, (400e-9, 2000e-9), 0.2e-12)
        q    = Hankel.QDHT(R, N, dim=2)

        dens0 = PhysData.density(gas, pres)
        densityfun(z) = dens0
        responses = (Nonlinear.Kerr_field(PhysData.γ3_gas(gas)),)
        linop   = LinearOps.make_const_linop(grid, q, PhysData.ref_index_fun(gas, pres))
        normfun = NonlinearRHS.const_norm_radial(grid, q, PhysData.ref_index_fun(gas, pres))
        inputs  = Fields.GaussGaussField(λ0=λ0, τfwhm=τ, energy=energy, w0=w0, propz=-0.15)

        Eω, transform, FT = with_logger(NullLogger()) do
            Amalthea.setup(grid, q, densityfun, normfun, responses, inputs)
        end

        @assert transform isa Amalthea.NonlinearRHS.TransRadial "Expected TransRadial"

        t0 = 0.0
        dt = 0.001

        @testset "backend_config default is off" begin
            withenv("LUNA_NATIVE_DETERMINISTIC" => nothing) do
                @test !Amalthea.Config.backend_config().deterministic
            end
        end

        @testset "T1: deterministic=1, two runs bit-identical (BLAS never engaged)" begin
            withenv("LUNA_NATIVE_DETERMINISTIC" => "1") do
                @test Amalthea.Config.backend_config().deterministic

                s1 = RustNativeStepper(transform, linop, copy(Eω), t0, dt,
                                        rtol=1e-6, atol=1e-10, max_dt=dt, min_dt=dt)
                s2 = RustNativeStepper(transform, linop, copy(Eω), t0, dt,
                                        rtol=1e-6, atol=1e-10, max_dt=dt, min_dt=dt)

                solve(s1, L)
                solve(s2, L)

                @test s1.yn == s2.yn
            end
        end

        # Deliberately populate the process-global `BLAS_API` OnceLock via
        # the per-kernel path — the only call site that ever does. Once
        # set, it stays set for the rest of this process, which is exactly
        # the scenario `deterministic` must guard against for the native
        # path (docs/dev/BACKLOG.md S1.6/S1 item 1's "process-global state
        # contaminates later constructions" class of bug).
        h = withenv("LUNA_USE_RUST_QDHT" => "1", "LUNA_QDHT_BLAS" => "1") do
            with_logger(NullLogger()) do
                NonlinearRHS._make_rust_qdht_handle(q, length(grid.to))
            end
        end
        @test !isnothing(h)

        @testset "T2: after BLAS_API is populated, deterministic changes the QDHT codepath" begin
            s_blas = withenv("LUNA_NATIVE_DETERMINISTIC" => nothing) do
                s = RustNativeStepper(transform, linop, copy(Eω), t0, dt,
                                       rtol=1e-6, atol=1e-10, max_dt=dt, min_dt=dt)
                solve(s, L)
                s
            end
            s_det = withenv("LUNA_NATIVE_DETERMINISTIC" => "1") do
                s = RustNativeStepper(transform, linop, copy(Eω), t0, dt,
                                       rtol=1e-6, atol=1e-10, max_dt=dt, min_dt=dt)
                solve(s, L)
                s
            end
            @test all(isfinite, s_blas.yn)
            @test all(isfinite, s_det.yn)
            # BLAS-3 dgemm and the row-parallel Rayon fallback sum in a
            # different order, so with BLAS_API now live, `deterministic`
            # must produce a numerically distinct (not bit-identical)
            # result from the default — proof the flag actually gates the
            # BLAS branch rather than being an inert no-op.
            @test s_blas.yn != s_det.yn
        end

        @testset "T3: deterministic=1, still bit-identical across repeats after contamination" begin
            withenv("LUNA_NATIVE_DETERMINISTIC" => "1") do
                s1 = RustNativeStepper(transform, linop, copy(Eω), t0, dt,
                                        rtol=1e-6, atol=1e-10, max_dt=dt, min_dt=dt)
                s2 = RustNativeStepper(transform, linop, copy(Eω), t0, dt,
                                        rtol=1e-6, atol=1e-10, max_dt=dt, min_dt=dt)

                solve(s1, L)
                solve(s2, L)

                @test s1.yn == s2.yn
            end
        end

        @testset "T4: toggle off restores default behavior (no crash, sane result)" begin
            withenv("LUNA_NATIVE_DETERMINISTIC" => nothing) do
                @test !Amalthea.Config.backend_config().deterministic
                s = RustNativeStepper(transform, linop, copy(Eω), t0, dt,
                                       rtol=1e-6, atol=1e-10, max_dt=dt, min_dt=dt)
                solve(s, L)
                @test all(isfinite, s.yn)
            end
        end
    end
end
