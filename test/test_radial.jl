using TestItems

@testitem "Radial" tags=[:sim_propagation] begin
import Test: @test, @testset
import Luna
import Luna: Grid, Maths, PhysData, Nonlinear, Ionisation, NonlinearRHS, Output, LinearOps, Fields
import Hankel
import LinearAlgebra: norm

@testset "Radial propagation" begin
gas = :Ar
pres = 1.2
τ = 20e-15
λ0 = 800e-9
w0 = 40e-6
energy = 1e-12
L = 0.6
R = 4e-3
N = 128

grid = Grid.RealGrid(L, 800e-9, (400e-9, 2000e-9), 0.2e-12)
q = Hankel.QDHT(R, N, dim=2)

dens0 = PhysData.density(gas, pres)
densityfun(z) = dens0
ionpot = PhysData.ionisation_potential(gas)
ionrate = Ionisation.IonRatePPTCached(gas, λ0)
responses = (Nonlinear.Kerr_field(PhysData.γ3_gas(gas)),
             Nonlinear.PlasmaCumtrapz(grid.to, grid.to, ionrate, ionpot))
linop = LinearOps.make_const_linop(grid, q, PhysData.ref_index_fun(gas, pres))
normfun = NonlinearRHS.const_norm_radial(grid, q, PhysData.ref_index_fun(gas, pres))

inputs = Fields.GaussGaussField(λ0=λ0, τfwhm=τ, energy=energy, w0=w0, propz=-0.3)

Eω, transform, FT = Luna.setup(grid, q, densityfun, normfun, responses, inputs)
output = Output.MemoryOutput(0, grid.zmax, 201)
Luna.run(Eω, grid, linop, transform, FT, output)

Erout = (q \ output.data["Eω"])
Iωr = abs2.(Erout)

ω0idx = argmin(abs.(grid.ω .- 2π*PhysData.c/λ0))

Iλ0 = Iωr[ω0idx, :, :]
λ0 = 2π*PhysData.c/grid.ω[ω0idx]
w1 = w0*sqrt(1+(L/2*λ0/(π*w0^2))^2)
Iλ0_analytic = Maths.gauss.(q.r, w1/2)*(w0/w1)^2 # analytical solution (in paraxial approx)
Ir = Maths.normbymax(Iλ0[:, end])
Ira = Maths.normbymax(Iλ0_analytic)
@test maximum(abs.(Ir .- Ira)/norm(Ir)) < 1.5e-4
end

end
