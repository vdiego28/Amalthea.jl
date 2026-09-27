using TestItems

@testitem "Spectral phase API compatibility" tags=[:fields] begin
using Amalthea
using Test

# Small dispatch fixtures keep this API check independent of propagation,
# FFT planning and output-file creation.
struct PhaseGridFixture <: Processing.AbstractGrid
    ω::Vector{Float64}
    t::Vector{Float64}
end
struct PhaseOutputFixture
    grid::PhaseGridFixture
    field::Array{ComplexF64, 3}
end
Processing.makegrid(output::PhaseOutputFixture) = output.grid
Processing.getEω(output::PhaseOutputFixture) = (output.grid.ω, output.field)
Processing.getEω(output::PhaseOutputFixture, idx::Int) =
    (output.grid.ω, output.field[:, :, idx:idx], [Float64(idx)])

ω = collect(range(0.0, 1.0; length=9))
grid = PhaseGridFixture(ω, [-0.5, -0.25, 0.0, 0.25])
τ = 0.5
φ = 0.2 .+ 0.3 .* ω.^2
# A forward FFT of a pulse centered at array time τ has a negative ramp.
field = cis.(φ .- ω .* τ)
fields = cat(hcat(field, field .* cis(0.1)), hcat(field .* cis(0.2), field); dims=3)
original = copy(fields)
expected = cat(hcat(φ, φ .+ 0.1), hcat(φ .+ 0.2, φ); dims=3)
output = PhaseOutputFixture(grid, fields)

@test Processing.spectral_phase(ω, field, τ) ≈ φ atol=1e-14 rtol=0
@test Processing.spectral_phase(ω, fields, τ) ≈ expected atol=1e-14 rtol=0
@test Processing.spectral_phase(grid, fields) == Processing.spectral_phase(ω, fields, τ)
@test Processing.spectral_phase(output) == Processing.spectral_phase(grid, fields)
@test Processing.spectral_phase(output, 2) == Processing.spectral_phase(grid, fields[:, :, 2:2])
@test Processing.getφ(ω, fields, τ) == Processing.spectral_phase(ω, fields, τ)
@test Processing.getφ(grid, fields) == Processing.spectral_phase(grid, fields)
@test Processing.getφ(output, 2) == Processing.spectral_phase(output, 2)
@test fields == original
end
