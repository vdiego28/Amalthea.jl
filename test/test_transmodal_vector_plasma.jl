using TestItems

@testitem "TransModal full vector plasma" tags=[:sim_multimode] begin
    import Test: @test, @testset
    using Amalthea
    import LinearAlgebra: norm
    import Logging: with_logger, NullLogger

    @testset "PlasmaCumtrapz reports a mismatched vector scratch shape" begin
        t = collect(range(-1e-15, 1e-15; length=8))
        ionrate = (rate, E) -> fill!(rate, 0.0)
        plasma = Nonlinear.PlasmaCumtrapz(t, t, ionrate, 1.0)
        Et = zeros(length(t), 2)
        out = similar(Et)

        err = try
            plasma(out, Et, 1.0)
            nothing
        catch e
            e
        end
        @test err isa DimensionMismatch
        @test occursin("time × polarisation shape", sprint(showerror, err))
    end

    @testset "full=true, npol=2, Kerr + plasma evaluates" begin
        a = 13e-6
        gas = :Ar
        pres = 5.0
        λ0 = 800e-9
        grid = Grid.RealGrid(1e-3, λ0, (300e-9, 2000e-9), 0.25e-12)
        modes = (
            Capillary.MarcatiliMode(
                a, gas, pres; n=1, m=1, kind=:HE, ϕ=0.0, loss=false),
            Capillary.MarcatiliMode(
                a, gas, pres; n=1, m=2, kind=:HE, ϕ=π/2, loss=false),
        )
        density = PhysData.density(gas, pres)
        densityfun(z) = density
        ionpot = PhysData.ionisation_potential(gas)
        ionrate = Ionisation.IonRateADK(ionpot)
        plasma = Nonlinear.PlasmaCumtrapz(
            grid.to, zeros(length(grid.to), 2), ionrate, ionpot)
        responses = (
            Nonlinear.Kerr_field(PhysData.γ3_gas(gas)),
            plasma,
        )
        input = Fields.GaussField(λ0=λ0, τfwhm=30e-15, energy=1e-6)

        Eω, transform, _ = with_logger(NullLogger()) do
            Amalthea.setup(
                grid, densityfun, responses, input, modes, :xy;
                full=true, mfcn=256)
        end
        @test transform isa NonlinearRHS.TransModal
        @test transform.full
        @test transform.ts.npol == 2

        nl = similar(Eω)
        transform(nl, Eω, 0.0)
        @test all(isfinite, nl)
        @test norm(nl) > 0

        _, kerr_transform, _ = with_logger(NullLogger()) do
            Amalthea.setup(
                grid, densityfun, (responses[1],), input, modes, :xy;
                full=true, mfcn=256)
        end
        nl_kerr = similar(Eω)
        kerr_transform(nl_kerr, Eω, 0.0)
        rel_plasma = norm(nl - nl_kerr) / norm(nl)
        @test rel_plasma > 1e-8
    end
end
