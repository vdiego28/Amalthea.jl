using TestItems

@testitem "Rust PreconStepper equivalence" tags=[:rust] begin
    import Test: @test, @testset
    using Amalthea
    import Logging: with_logger, NullLogger
    import LinearAlgebra: norm

    # ── skip guard ────────────────────────────────────────────────────────────
    libname = if Sys.iswindows(); "luna_rust.dll"
              elseif Sys.isapple(); "libluna_rust.dylib"
              else; "libluna_rust.so"; end
    libpath = joinpath(@__DIR__, "..", "luna-rust", "target", "release", libname)
    if !isfile(libpath)
        @warn "Skipping Rust PreconStepper test: shared library not found at $libpath. " *
              "Build with `cargo build --release` in luna-rust/."
        return
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
        out_rust = withenv("LUNA_USE_RUST_STEPPER" => "1") do
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
