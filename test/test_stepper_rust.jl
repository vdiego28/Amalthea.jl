using TestItems

@testitem "Rust PreconStepper equivalence" tags=[:rust] begin
    import Test: @test, @test_throws, @testset
    using Amalthea
    import Logging: with_logger, NullLogger
    import LinearAlgebra: norm
    using Amalthea.RK45: NativeIneligible, PreconStepper, RustNativeStepper,
                         RustPreconStepper, maxnorm, step!, weaknorm

    # ── skip guard ────────────────────────────────────────────────────────────
    libname = if Sys.iswindows(); "amalthea.dll"
              elseif Sys.isapple(); "libamalthea.dylib"
              else; "libamalthea.so"; end
    libpath = joinpath(@__DIR__, "..", "amalthea", "target", "release", libname)
    if !isfile(libpath)
        @warn "Skipping Rust PreconStepper test: shared library not found at $libpath. " *
              "Build with `cargo build --release` in amalthea/."
        return
    end

    # A deterministic time-only RHS whose embedded error is spread over nine
    # components while the state norm is anchored by one unit component. This
    # makes the choice of error norm observable: maxnorm accepts the initial
    # trial while weaknorm rejects it.
    amplitude = 3.4
    y0 = zeros(ComplexF64, 10)
    y0[1] = 1
    linop = zeros(ComplexF64, length(y0))
    function distributed_error_rhs!(out, _, t)
        fill!(out, amplitude * t^4)
        out[1] = 0
        nothing
    end

    @testset "Unsupported norm routes to the Julia oracle" begin
        s_max = PreconStepper(distributed_error_rhs!, linop, copy(y0), 0.0, 1.0;
                              rtol=1e-3, atol=0.0, norm=maxnorm)
        s_weak = PreconStepper(distributed_error_rhs!, linop, copy(y0), 0.0, 1.0;
                               rtol=1e-3, atol=0.0, norm=weaknorm)
        @test step!(s_max)
        @test !step!(s_weak)
        @test 0.89 < s_max.err < 0.90
        @test 1.17 < s_weak.err < 1.19

        @test_throws NativeIneligible RustPreconStepper(
            distributed_error_rhs!, linop, copy(y0), 0.0, 1.0; norm=maxnorm)
        @test_throws NativeIneligible RustNativeStepper(
            distributed_error_rhs!, linop, copy(y0), 0.0, 1.0; norm=maxnorm)

        withenv("AMALTHEA_USE_RUST_NATIVE" => "1",
                "AMALTHEA_USE_RUST_STEPPER" => "1") do
            with_logger(NullLogger()) do
                RK45.solve_precon(distributed_error_rhs!, linop, copy(y0),
                                  0.0, 1.0, 0.5;
                                  rtol=1e-3, atol=0.0, norm=maxnorm)
            end
        end
        @test RK45._LAST_STEPPER_TYPE[] <: PreconStepper
    end

    @testset "Legacy Rust locextrap=false matches Julia" begin
        common = (; rtol=1e-2, atol=0.0, locextrap=false)
        s_jl = PreconStepper(distributed_error_rhs!, linop, copy(y0), 0.0, 1.0;
                             common...)
        s_ru = RustPreconStepper(distributed_error_rhs!, linop, copy(y0), 0.0, 1.0;
                                 common...)
        s_true = PreconStepper(distributed_error_rhs!, linop, copy(y0), 0.0, 1.0;
                               rtol=1e-2, atol=0.0, locextrap=true)

        @test step!(s_jl)
        @test step!(s_ru)
        @test step!(s_true)
        @test s_ru.ok == s_jl.ok
        @test isapprox(s_ru.err, s_jl.err; rtol=1e-13)
        @test isapprox(s_ru.dtn, s_jl.dtn; rtol=1e-13)
        @test norm(s_ru.yn - s_jl.yn) / norm(s_jl.yn) < 1e-13
        @test norm(s_jl.yn - s_true.yn) / norm(s_jl.yn) > 1e-5

        # Rejection must restore the old field even though the error was
        # evaluated against the final internal-stage candidate.
        rejected_jl = PreconStepper(distributed_error_rhs!, linop, copy(y0),
                                    0.0, 1.0;
                                    rtol=1e-4, atol=0.0, locextrap=false)
        rejected_ru = RustPreconStepper(distributed_error_rhs!, linop, copy(y0),
                                        0.0, 1.0;
                                        rtol=1e-4, atol=0.0, locextrap=false)
        @test !step!(rejected_jl)
        @test !step!(rejected_ru)
        @test rejected_jl.yn == y0
        @test rejected_ru.yn == y0
        @test isapprox(rejected_ru.err, rejected_jl.err; rtol=1e-13)
        @test isapprox(rejected_ru.dtn, rejected_jl.dtn; rtol=1e-13)

        # Fixed accepted steps exercise the deferred k7→k1 FSAL carry.
        fixed = (; rtol=1e-2, atol=0.0, locextrap=false,
                 min_dt=0.25, max_dt=0.25)
        multi_jl = PreconStepper(distributed_error_rhs!, linop, copy(y0),
                                 0.0, 0.25; fixed...)
        multi_ru = RustPreconStepper(distributed_error_rhs!, linop, copy(y0),
                                     0.0, 0.25; fixed...)
        for _ in 1:4
            @test step!(multi_jl)
            @test step!(multi_ru)
        end
        @test norm(multi_ru.yn - multi_jl.yn) / norm(multi_jl.yn) < 1e-13
    end

    # ─────────────────────────────────────────────────────────────────────────
    # Integration: run the same capillary simulation with Julia and Rust steppers
    # and compare the final output field spectrum.
    # ─────────────────────────────────────────────────────────────────────────
    @testset "Full capillary simulation equivalence" begin
        # A short run with Kerr nonlinearity — enough to exercise all 6 Runge-Kutta
        # stages, FSAL propagation, PI step control, and dense-output interpolation.
        radius = 125e-6; L = 0.1; gas = :Ar; pres = 1.0
        λ0 = 800e-9; τ = 30e-15; energy = 1e-7

        # ── Julia stepper (default) ───────────────────────────────────────────
        out_julia = with_logger(NullLogger()) do
            prop_capillary(radius, L, gas, pres;
                           λ0=λ0, τfwhm=τ, energy=energy,
                           modes=:HE11, loss=false,
                           saveN=2, trange=0.5e-12,
                           λlims=(200e-9, 4e-6))
        end
        Eω_julia = out_julia["Eω"][:, end]

        # ── Rust stepper path ─────────────────────────────────────────────────
        out_rust = withenv("AMALTHEA_USE_RUST_STEPPER" => "1") do
            with_logger(NullLogger()) do
                prop_capillary(radius, L, gas, pres;
                               λ0=λ0, τfwhm=τ, energy=energy,
                               modes=:HE11, loss=false,
                               saveN=2, trange=0.5e-12,
                               λlims=(200e-9, 4e-6))
            end
        end
        Eω_rust = out_rust["Eω"][:, end]

        # Both steppers call the same Julia fbar!/prop! callbacks, so the physics
        # matches.  The Julia nonlinear RHS uses FFTW and LLVM-vectorised broadcasts
        # whose FP summation order varies run-to-run (Julia-vs-Julia reproducibility
        # is itself ~2e-8 for this setup), so we accept any error below rtol=1e-6.
        rel_err = norm(Eω_rust - Eω_julia) / norm(Eω_julia)
        @test rel_err < 1e-6
    end
end
