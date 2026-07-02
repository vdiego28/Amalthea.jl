using TestItems

@testitem "Mixtures" tags=[:io] begin
using Luna
import Test: @test, @testset
import LinearAlgebra: norm

@testset "refractive index" begin
@testset "$gi" for gi in PhysData.gas
    @testset "$Pi bar" for Pi in (0.01, 0.1, 1, 10)
        @testset "$(λi*1e9) nm" for λi in (200e-9, 400e-9, 800e-9, 1600e-9)
            for divisor in (2, 3, 4, 5)
                divdens = PhysData.density(gi, Pi)/divisor
                divP = PhysData.pressure(gi, divdens)
                ns = PhysData.ref_index(gi, λi, Pi)
                gt = Tuple([gi for _ in 1:divisor])
                pt = Tuple([divP for _ in 1:divisor])
                nm = PhysData.ref_index(gt, λi, pt)
                nm2 = PhysData.ref_index_fun(gt)(λi, [divdens for _ = 1:divisor])
                @test (nm ≈ ns)
                @test (nm2 ≈ ns)
            end
        end
    end
end
end

@testset "propagation" begin
a = 13e-6
gas = :Ar
pres = 5
τ = 30e-15
λ0 = 800e-9
L = 5e-2

# Pressure at which argon density is exactly half of that at 5 bar
halfpres = PhysData.pressure(:Ar, PhysData.density(:Ar, pres)/2)

# Common setup
grid = Grid.RealGrid(L, λ0, (160e-9, 3000e-9), 0.5e-12)
inputs = Fields.GaussField(λ0=λ0, τfwhm=τ, energy=1e-6)

# Single gas
m = Capillary.MarcatiliMode(a, gas, pres; loss=false)
aeff(z) = Modes.Aeff(m; z=z)
densityfun = let dens0=PhysData.density(gas, pres)
    z -> dens0
end
responses = (Nonlinear.Kerr_field(PhysData.γ3_gas(gas)),)

linop, βfun!, _, _ = LinearOps.make_const_linop(grid, m, λ0)
Eω, transform, FT = Luna.setup(grid, densityfun, responses, inputs, βfun!, aeff)
statsfun = Stats.default(grid, Eω, m, linop, transform; gas=gas)
output_single = Output.MemoryOutput(0, grid.zmax, 201, statsfun)
Luna.run(Eω, grid, linop, transform, FT, output_single)

# Mixture
m = Capillary.MarcatiliMode(a, (gas, gas), (halfpres, halfpres); loss=false)
aeff(z) = Modes.Aeff(m; z=z)
densityfun = let dens0=PhysData.density(gas, halfpres)
    z -> [dens0, dens0]
end
responses = (
    (Nonlinear.Kerr_field(PhysData.γ3_gas(gas)),),
    (Nonlinear.Kerr_field(PhysData.γ3_gas(gas)),)
)

linop, βfun!, _, _ = LinearOps.make_const_linop(grid, m, λ0)
Eω, transform, FT = Luna.setup(grid, densityfun, responses, inputs, βfun!, aeff)
statsfun = Stats.default(grid, Eω, m, linop, transform; gas=(gas, gas))
output_mix = Output.MemoryOutput(0, grid.zmax, 201, statsfun)
Luna.run(Eω, grid, linop, transform, FT, output_mix)


# Gas mixtures give `densityfun(z)` a per-species Vector return, which the
# native mode-averaged path's `RustNativeStepper` rejects as
# `NativeIneligible` (falls back to the Julia stepper — see RK45.jl); the
# single-gas config above has a scalar `densityfun(z)` and is native-eligible.
# So under Phase 8's native-by-default, these two runs legitimately execute
# on different backends, and can only be expected to agree to the
# established native-vs-Julia method tolerance (~1e-13, Phase 1), not
# bit-for-bit as before Phase 8 (when both always ran in pure Julia).
rel = norm(output_mix.data["Eω"][grid.sidx, :] - output_single.data["Eω"][grid.sidx, :]) /
      norm(output_single.data["Eω"][grid.sidx, :])
@test rel < 1e-8
end
end
