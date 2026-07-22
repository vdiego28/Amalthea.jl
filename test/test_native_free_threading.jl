using TestItems

@testitem "Native-Rust S2 item 4 (free-space 3-D FFT threading, bit-identical)" tags=[:rust] begin
    import Test: @test, @test_skip, @testset
    using Amalthea
    import Amalthea: Grid, NonlinearRHS, Fields, LinearOps, PhysData, Nonlinear
    using Amalthea.RK45: PreconStepper, RustNativeStepper, solve
    import Logging: with_logger, NullLogger
    import LinearAlgebra: norm

    libpath = RK45._LIBAMALTHEA_RK45
    if !isfile(libpath)
        @test_skip "Rust library not found"
    else
        # docs/dev/BACKLOG.md S2 item 4: unlike radial/modal threading (rayon
        # over independent per-column work against a shared *single-threaded*
        # FFTW plan), free-space has no per-column loop to parallelize — the
        # 3-D transform (`rhs_free`'s Steps 2/5) is ONE joint FFT per RK
        # stage. Threading here means baking FFTW's OWN internal thread count
        # into that one plan (`fftw.rs`'s `RealFft3d`/`ComplexFft3d`
        # `nthreads` argument, `native_set_free_params` passing
        # `sim.n_threads`) so a single `fftw_execute_dft_r2c`/`_c2r` call
        # fans out to FFTW's private worker pool and blocks the one calling
        # thread until done — no concurrent Rust-side access to the plan,
        # which is also why `RealFft3d`/`ComplexFft3d` deliberately do NOT
        # implement `Sync` (contrast the radial/modal 1-D plans, which do).
        # A moderate transverse grid (Nx=Ny=24, 576 columns) is used here
        # instead of the tiny 8x6 grid `test_native_free.jl` uses for its
        # cheap parity checks — FFTW's threaded decomposition needs a large
        # enough transform to actually engage (see the load-bearing
        # `fftw.rs::r2c_3d_threaded_matches_single_threaded`/
        # `c2c_3d_threaded_matches_single_threaded` unit tests, run at an
        # even larger size, which established bit-identical output AND a
        # measured wall-clock speedup before this Julia-level wiring existed
        # at all).
        gas = :Ar; pres = 1.0; τ = 20e-15; λ0 = 800e-9
        w0 = 60e-6; energy = 1e-9; L = 0.004; R = 300e-6; Nx = 24; Ny = 24
        t0 = 0.0; dt = 0.0005

        grid = Grid.RealGrid(L, λ0, (400e-9, 2000e-9), 0.5e-12)
        xygrid = Grid.FreeGrid(R, Nx, R, Ny)

        dens0 = PhysData.density(gas, pres)
        densityfun(z) = dens0
        responses = (Nonlinear.Kerr_field(PhysData.γ3_gas(gas)),)
        linop = LinearOps.make_const_linop(grid, xygrid, PhysData.ref_index_fun(gas, pres))
        normfun = NonlinearRHS.const_norm_free(grid, xygrid, PhysData.ref_index_fun(gas, pres))
        inputs = Fields.GaussGaussField(λ0=λ0, τfwhm=τ, energy=energy, w0=w0)

        Eω, transform, FT = with_logger(NullLogger()) do
            Amalthea.setup(grid, xygrid, densityfun, normfun, responses, inputs)
        end
        @assert transform isa NonlinearRHS.TransFree "Expected TransFree"

        @testset "RealGrid Kerr: n_threads=1 vs n_threads=4 bit-identical" begin
            s_jl = PreconStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10,
                                  max_dt=dt, min_dt=dt)
            s1 = RustNativeStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10,
                                    max_dt=dt, min_dt=dt, native_threads=1)
            s4 = RustNativeStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10,
                                    max_dt=dt, min_dt=dt, native_threads=4)
            solve(s_jl, L); solve(s1, L); solve(s4, L)
            @test s1.yn == s4.yn                     # bit-identical — the S2 gate
            rel = norm(s1.yn - s_jl.yn) / norm(s_jl.yn)
            println("free-space-threading (RealGrid Kerr): native-vs-julia rel = ", rel)
            @test rel < 1e-9
        end

        @testset "EnvGrid Kerr: n_threads=1 vs n_threads=4 bit-identical" begin
            egrid = Grid.EnvGrid(L, λ0, (400e-9, 2000e-9), 0.5e-12)
            elinop = LinearOps.make_const_linop(egrid, xygrid, PhysData.ref_index_fun(gas, pres))
            enormfun = NonlinearRHS.const_norm_free(egrid, xygrid, PhysData.ref_index_fun(gas, pres))
            einputs = Fields.GaussGaussField(λ0=λ0, τfwhm=τ, energy=energy, w0=w0)
            eresponses = (Nonlinear.Kerr_env(PhysData.γ3_gas(gas)),)
            EEω, etransform, EFT = with_logger(NullLogger()) do
                Amalthea.setup(egrid, xygrid, densityfun, enormfun, eresponses, einputs)
            end
            @assert etransform isa NonlinearRHS.TransFree "Expected TransFree"

            es_jl = PreconStepper(etransform, elinop, copy(EEω), t0, dt, rtol=1e-6, atol=1e-10,
                                   max_dt=dt, min_dt=dt)
            es1 = RustNativeStepper(etransform, elinop, copy(EEω), t0, dt, rtol=1e-6, atol=1e-10,
                                     max_dt=dt, min_dt=dt, native_threads=1)
            es4 = RustNativeStepper(etransform, elinop, copy(EEω), t0, dt, rtol=1e-6, atol=1e-10,
                                     max_dt=dt, min_dt=dt, native_threads=4)
            solve(es_jl, L); solve(es1, L); solve(es4, L)
            @test es1.yn == es4.yn                    # bit-identical — the S2 gate
            rel = norm(es1.yn - es_jl.yn) / norm(es_jl.yn)
            println("free-space-threading (EnvGrid Kerr): native-vs-julia rel = ", rel)
            @test rel < 1e-9
        end

        @testset "GC-stress: forced GC between threaded solves (FFI safety)" begin
            # Free-space threading introduces no new Julia-owned persistent
            # raw pointer into Rust (the 3-D FFTW plan is Rust-owned,
            # created once at `native_set_free_params` time and never
            # re-dereferences a Julia pointer afterward) — unlike the
            # radial-plasma S2 crash (a missing GC root on a Julia-owned
            # ionization-LUT pointer, BACKLOG.md's S2 postmortem). This is a
            # standing regression guard, not evidence of a specific new
            # hazard here.
            ref = nothing
            ok = true
            for cyc in 1:4
                nt = isodd(cyc) ? 1 : 4
                s = RustNativeStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10,
                                       max_dt=dt, min_dt=dt, native_threads=nt)
                solve(s, L)
                GC.gc()
                if ref === nothing
                    ref = copy(s.yn)
                else
                    ok &= (s.yn == ref)
                end
            end
            @test ok
        end
    end
end
