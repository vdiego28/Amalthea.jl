using Luna
using Test

@testset "Greek aliases" begin
    # Compare prop_capillary with Unicode and ASCII aliases
    kwargs_unicode = (λ0=800e-9, τfwhm=10e-15, energy=1e-12, trange=400e-15,
                      λlims=(200e-9, 4e-6), shotnoise=false)
    kwargs_ascii   = (lambda0=800e-9, tau_fwhm=10e-15, energy=1e-12, trange=400e-15,
                      lambda_lims=(200e-9, 4e-6), shotnoise=false)
                      
    o1 = prop_capillary(100e-6, 0.1, :He, 1; kwargs_unicode...)
    o2 = prop_capillary(100e-6, 0.1, :He, 1; kwargs_ascii...)
    @test o1["Eω"] == o2["Eω"]

    # Test GaussPulse with ASCII and Unicode
    gp1 = Pulses.GaussPulse(λ0=800e-9, τfwhm=10e-15, energy=1e-12)
    gp2 = Pulses.GaussPulse(lambda0=800e-9, tau_fwhm=10e-15, energy=1e-12)
    @test gp1.field.λ0 == gp2.field.λ0
    
    # Test duplicate detection
    @test_throws ErrorException prop_capillary(100e-6, 0.1, :He, 1; λ0=800e-9, lambda0=800e-9, kwargs_unicode...)
end
