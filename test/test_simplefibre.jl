using TestItems

@testitem "SimpleFibre" tags=[:physics] begin
import Test: @test, @testset
using Amalthea
using Amalthea: SimpleFibre, Modes
import Amalthea.PhysData: c

@testset "Constructors" begin
    # 0th to 2nd order Taylor coeffs
    βs = [1.0, 2.0, 3.0]
    ωref = 2.0

    m = SimpleFibre.SimpleMode(ωref, βs; Aeff=2.5, loss=10.0)

    @test m.ωref == 2.0
    @test m.Aeff == 2.5
    @test m.α == log(10)/10 * 10.0
    # Polynomials coeffs: β0 / 0!, β1 / 1!, β2 / 2!
    @test m.poly.coeffs ≈ [1.0, 2.0, 1.5]

    # 1-argument constructor
    m2 = SimpleFibre.SimpleMode(βs)
    @test m2.ωref == 0.0
    @test m2.Aeff == 1.0
    @test m2.α == 0.0
    @test m2.poly.coeffs ≈ [1.0, 2.0, 1.5]
end

@testset "Methods" begin
    βs = [1.0, 2.0, 3.0]
    ωref = 2.0
    loss_dB_per_m = 10.0
    α_val = log(10)/10 * loss_dB_per_m
    m = SimpleFibre.SimpleMode(ωref, βs; Aeff=2.5, loss=loss_dB_per_m)

    ω = 2.5
    Δω = ω - ωref

    # β = β0 + β1 Δω + β2/2 Δω^2
    β_expected = 1.0 + 2.0*Δω + 1.5*Δω^2
    @test Modes.β(m, ω) ≈ β_expected

    @test Modes.α(m, ω) == α_val
    @test Modes.Aeff(m) == 2.5

    neff_expected = c / ω * (β_expected + 0.5im * α_val)
    @test Modes.neff(m, ω) ≈ neff_expected

    # dispersion_func
    d1_func = Modes.dispersion_func(m, 1)
    # 1st derivative of β wrt ω: β1 + β2 Δω
    @test d1_func(ω) ≈ 2.0 + 3.0*Δω

    d2_func = Modes.dispersion_func(m, 2)
    # 2nd derivative: β2
    @test d2_func(ω) ≈ 3.0
end
end
