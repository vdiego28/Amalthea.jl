using TestItems

@testitem "Native-Rust S2 Phase 4 (modal RHS threading, bit-identical)" tags=[:rust] begin
    import Test: @test, @test_skip, @testset
    using Amalthea
    import Amalthea: Grid, NonlinearRHS, Fields, LinearOps, PhysData, Nonlinear, Capillary, Raman
    using Amalthea.RK45: PreconStepper, RustNativeStepper, solve
    import Logging: with_logger, NullLogger
    import LinearAlgebra: norm

    libpath = RK45._LIBAMALTHEA_RK45
    if !isfile(libpath)
        @test_skip "Rust library not found"
    else
        # docs/dev/BACKLOG.md S2 Phase 4 (modal): the modal integrand
        # (`modal_pointcalc`) is evaluated per cubature node; measured ~90-96%
        # of `rhs_modal` wall time, so the batch's nodes are parallelized across
        # rayon workers, each owning a disjoint `ModalScratch` (Kerr scratch,
        # plus — for Raman — its own cloned `TimeDomainRamanSolver` + Hilbert
        # scratch). Every node writes only its own `out` slice with no
        # cross-node reduction, so `native_threads=1` and `=4` must agree
        # EXACTLY, not within tolerance — any mismatch is a real aliasing or
        # chunk-alignment bug. We also re-assert native-vs-Julia parity is
        # unchanged by the refactor (~1e-15/1e-16 for Kerr, ~1e-7 for Raman's
        # ADE-vs-FFT-convolution method-difference floor).
        gas = :Ar; pres = 1.0; τ = 20e-15; λ0 = 800e-9
        a = 125e-6; energy = 5e-6; L = 0.05
        grid = Grid.RealGrid(L, λ0, (400e-9, 2000e-9), 0.5e-12)
        t0 = 0.0; dt = 0.001
        dens0 = PhysData.density(gas, pres)
        densityfun(z) = dens0

        HE1 = Capillary.MarcatiliMode(a, gas, pres; kind=:HE, n=1, m=1)
        HE2 = Capillary.MarcatiliMode(a, gas, pres; kind=:HE, n=1, m=2)
        HEphi = Capillary.MarcatiliMode(a, gas, pres; kind=:HE, n=1, m=1, ϕ=π/4)
        kerr = (Nonlinear.Kerr_field(PhysData.γ3_gas(gas)),)

        function kerr_case(label, modes, pol, full)
            @testset "$label" begin
                linop = LinearOps.make_const_linop(grid, modes, grid.referenceλ)
                input = Fields.GaussField(λ0=λ0, τfwhm=τ, energy=energy)
                Eω, transform, FT = with_logger(NullLogger()) do
                    Amalthea.setup(grid, densityfun, kerr, input, modes, pol; full=full, mfcn=2000)
                end
                @assert transform isa NonlinearRHS.TransModal
                @assert transform.full == full
                s_jl = PreconStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10,
                                      max_dt=dt, min_dt=dt)
                s1 = RustNativeStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10,
                                        max_dt=dt, min_dt=dt, native_threads=1)
                s4 = RustNativeStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10,
                                        max_dt=dt, min_dt=dt, native_threads=4)
                solve(s_jl, L); solve(s1, L); solve(s4, L)
                @test s1.yn == s4.yn                     # bit-identical
                rel = norm(s1.yn - s_jl.yn) / norm(s_jl.yn)
                println("modal-threading ($label): native-vs-julia rel = ", rel)
                @test rel < 1e-9
            end
        end

        kerr_case("HE11 full=false", (HE1,), :y, false)
        kerr_case("HE11 full=true", (HE1,), :y, true)
        kerr_case("2-mode HE11+HE12 full=false", (HE1, HE2), :y, false)
        kerr_case("npol=2 (ϕ=π/4) full=false", (HEphi,), :xy, false)

        @testset "Raman modal (cloned per-worker solver)" begin
            # Exercises the per-worker cloned `TimeDomainRamanSolver` path.
            # solve() resets oscillator state at entry → cloned == shared →
            # bit-identical. Native-vs-Julia at the ~1e-7 ADE-vs-FFT-conv floor.
            rgas = :N2
            rdens0 = PhysData.density(rgas, pres)
            rdensityfun(z) = rdens0
            rmodes = (Capillary.MarcatiliMode(a, rgas, pres; m=1),)
            rr = Raman.raman_response(grid.to, rgas; rotation=false, vibration=true)
            responses = (Nonlinear.Kerr_field(PhysData.γ3_gas(rgas)),
                         Nonlinear.RamanPolarField(grid.to, rr))
            linop = LinearOps.make_const_linop(grid, rmodes, grid.referenceλ)
            input = Fields.GaussField(λ0=λ0, τfwhm=τ, energy=energy)
            Eω, transform, FT = with_logger(NullLogger()) do
                Amalthea.setup(grid, rdensityfun, responses, input, rmodes, :y)
            end
            s_jl = PreconStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10,
                                  max_dt=dt, min_dt=dt)
            s1 = RustNativeStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10,
                                    max_dt=dt, min_dt=dt, native_threads=1)
            s4 = RustNativeStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10,
                                    max_dt=dt, min_dt=dt, native_threads=4)
            solve(s_jl, L); solve(s1, L); solve(s4, L)
            @test s1.yn == s4.yn
            rel = norm(s1.yn - s_jl.yn) / norm(s_jl.yn)
            println("modal-threading (Raman :N2): native-vs-julia rel = ", rel)
            # ~1e-6 ADE-vs-FFT-convolution method floor (matches
            # test_native_modal_raman.jl's own full-solve threshold), not a
            # threading effect — bit-identical (line above) is the S2 gate.
            @test rel < 1.5e-6
        end

        @testset "GC-stress: forced GC between threaded solves (FFI safety)" begin
            # The radial-plasma S2 crash was a Julia-owned pointer freed mid-solve
            # (BACKLOG.md S2 postmortem). Modal threading introduces no new
            # persistent raw pointer into Julia memory (the Raman solver is
            # Rust-owned and only *cloned*), but force GC between alternating
            # thread-count solves as a standing regression guard.
            linop = LinearOps.make_const_linop(grid, (HE1,), grid.referenceλ)
            input = Fields.GaussField(λ0=λ0, τfwhm=τ, energy=energy)
            Eω, transform, FT = with_logger(NullLogger()) do
                Amalthea.setup(grid, densityfun, kerr, input, (HE1,), :y; full=false)
            end
            ref = nothing
            ok = true
            for cyc in 1:6
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
