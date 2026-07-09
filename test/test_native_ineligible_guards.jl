using TestItems

@testitem "Native-Rust Phase I item 7 (NativeIneligible guard regressions)" tags=[:rust] begin
    import Test: @test, @test_skip, @testset
    using Luna
    using Luna.RK45: RustNativeStepper, NativeIneligible
    import Logging: with_logger, NullLogger

    libpath = RK45._LIBLUNA_RUST_RK45
    if !isfile(libpath)
        @test_skip "Rust library not found"
    else
        # BACKLOG.md Phase I item 7: these three `NativeIneligible` sites are
        # documented as "correct-by-design edge-case guards, not gaps to
        # close" — but until now none had a dedicated regression test, so a
        # future refactor that silently broke one of them (e.g. by loosening
        # the `isa` check, or removing the guard while "cleaning up") would
        # only be caught if it happened to also change a numerical result in
        # some other test. Phase I item 1's postmortem is exactly this class
        # of gap: "documented as correct" is not the same as "verified to
        # actually fire". Lock in all three explicitly.

        @testset "Unrecognized bare f! closure is rejected, not silently zero-nonlinearity" begin
            # Phase 8's own postmortem (BACKLOG.md / CLAUDE.md): before this
            # guard existed, a raw ad-hoc closure (as RK45.jl's own low-level
            # unit tests use to exercise the Julia solver plumbing directly)
            # silently ran with zero nonlinearity under the native path,
            # since none of the `native_set_*_params` calls recognize it.
            bogus_f! = (nl, Eω, z) -> (nl .= 0)
            linop = zeros(ComplexF64, 8)
            y0 = zeros(ComplexF64, 8)
            @test_throws NativeIneligible RustNativeStepper(bogus_f!, linop, y0, 0.0, 0.01)
        end

        @testset "Non-finite flength with a z-dependent linop is rejected" begin
            # A z-dependent linop (Marcatili graded-core capillary) needs a
            # finite `flength` to build the resident z-LUTs at construction
            # time — `RustNativeStepper`'s default `flength=Inf` must not
            # silently proceed as if the linop were constant.
            radius = 125e-6
            flength = 0.5
            gas = :Ar
            pressure = (0.5, 5.0)  # a real two-point pressure gradient
            λ0 = 800e-9
            λlims = (200e-9, 4e-6)
            trange = 1e-12
            kw = (; λ0, λlims, trange, raman=false, plasma=false, kerr=true,
                    shotnoise=false, energy=1e-6, τfwhm=30e-15)

            Eω, grid, linop, transform, FT, output = with_logger(NullLogger()) do
                Interface.prop_capillary_args(radius, flength, gas, pressure; kw...)
            end
            @test linop isa Luna.Capillary.ZDepLinopMarcatili

            # Deliberately omit `flength` (defaults to `Inf`) — must throw,
            # not silently build a bogus resident linop-LUT.
            @test_throws NativeIneligible RustNativeStepper(transform, linop, copy(Eω), 0.0, 0.001)

            # Sanity check the guard's other side: passing the real, finite
            # `flength` must succeed (confirms this is a targeted rejection
            # of non-finite `flength`, not the z-dependent linop itself).
            s = RustNativeStepper(transform, linop, copy(Eω), 0.0, 0.001; flength=flength)
            @test s isa RustNativeStepper
        end

        @testset "Missing Rust ionisation handle is rejected, not silently ignored" begin
            # A plasma response whose `ratefunc` was never built with a Rust
            # handle (LUNA_USE_RUST_IONISATION unset/disabled at construction
            # time) must not silently propagate with plasma switched off —
            # that would be the same class of silent-wrong-physics bug as the
            # Phase I preamble's missing-density-factor finding, just for a
            # totally-absent contribution instead of a partially-wrong one.
            gas = :Ar; pres = 1.5; τ = 15e-15; λ0 = 800e-9
            w0 = 150e-6; energy = 6e-5; L = 0.02
            import Luna: Grid, NonlinearRHS, Fields, LinearOps, PhysData, Nonlinear, Ionisation
            import Hankel
            grid = Grid.RealGrid(L, λ0, (150e-9, 2000e-9), 0.5e-12)
            q    = Hankel.QDHT(4e-3, 32, dim=2)
            dens0 = PhysData.density(gas, pres)
            densityfun(z) = dens0
            ionpot = PhysData.ionisation_potential(gas)
            # `_make_rust_ionization_handle` (Ionisation.jl) builds a Rust
            # handle whenever the library is present and EITHER
            # `LUNA_USE_RUST_IONISATION=1` OR the native stepper is enabled
            # (`LUNA_USE_RUST_NATIVE`, on by default since Phase 8) — so
            # `ionrate.rust_handle` is only genuinely `nothing` when BOTH are
            # explicitly turned off at construction time.
            ionrate = withenv("LUNA_USE_RUST_IONISATION" => "0",
                               "LUNA_USE_RUST_NATIVE" => "0") do
                Ionisation.IonRatePPTCached(gas, λ0)
            end
            plasma_resp = Nonlinear.PlasmaCumtrapz(grid.to, grid.to, ionrate, ionpot)
            responses = (Nonlinear.Kerr_field(PhysData.γ3_gas(gas)), plasma_resp)
            linop   = LinearOps.make_const_linop(grid, q, PhysData.ref_index_fun(gas, pres))
            normfun = NonlinearRHS.const_norm_radial(grid, q, PhysData.ref_index_fun(gas, pres))
            inputs  = Fields.GaussGaussField(λ0=λ0, τfwhm=τ, energy=energy, w0=w0, propz=-0.1)

            Eω, transform, FT = with_logger(NullLogger()) do
                Luna.setup(grid, q, densityfun, normfun, responses, inputs)
            end

            @test_throws NativeIneligible RustNativeStepper(transform, linop, copy(Eω), 0.0, 0.0005)
        end
    end
end
