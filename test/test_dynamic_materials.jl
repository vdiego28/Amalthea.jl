using TestItems

@testitem "Dynamic material registry" tags=[:physics] begin
    using Luna
    # Register a custom gas with known Sellmeier coefficients (using HeB values)
    PhysData.register_material!(:TestGas;
        B=[4977.77e-8, 1856.94e-8],
        C=[28.54e-6, 7.76e-3],
        chi3=3.43e-28,
        ionisation_potential=24.587,
        quantum_numbers=(1, 0, 1),
        reference_density=PhysData.dens_1bar_0degC[:HeB])
    
    # Should give same index as HeB at 800 nm
    n_test = PhysData.ref_index(:TestGas, 800e-9)
    n_heb   = PhysData.ref_index(:HeB, 800e-9)
    @test isapprox(n_test, n_heb, rtol=1e-6)

    # Test custom glass
    PhysData.register_material!(:TestGlass;
        B=[0.6961663, 0.4079426, 0.8974794],
        C=[0.0684043^2, 0.1162414^2, 9.896161^2],
        type=:glass,
        n2=2.7e-20)
    n_glass = PhysData.ref_index(:TestGlass, 800e-9)
    n_sio2  = PhysData.ref_index(:SiO2, 800e-9, lookup=false)
    @test isapprox(n_glass, n_sio2, rtol=1e-6)
    @test PhysData.n2(:TestGlass) == 2.7e-20

    o = prop_capillary(100e-6, 0.1, :TestGas, 1;
                       λ0=800e-9, τfwhm=10e-15, energy=1e-12,
                       trange=400e-15, λlims=(200e-9, 4e-6), shotnoise=false)
    @test haskey(o.data, "Eω")
end
