using TestItems

@testitem "Spectral diagnostics analytic checks" tags=[:fields] begin
using Amalthea
using Test
import FFTW
import DSP: unwrap

# A dimensionless Fourier grid avoids propagation/setup. Only the transform
# accessors are specialized; the production width/summation routines run below.
struct DiagnosticGrid <: Processing.AbstractGrid
    t::Vector{Float64}
    ω::Vector{Float64}
end
function Processing.getEt(grid::DiagnosticGrid, Eω::AbstractArray; oversampling=1, bandpass=nothing)
    @assert oversampling == 1 && isnothing(bandpass)
    grid.t, FFTW.ifft(Eω, 1)
end
Processing.getEω(grid::DiagnosticGrid, Eω) =
    (FFTW.fftshift(grid.ω), FFTW.fftshift(Eω, 1))

@testset "FFT centering precedes unwrapping" begin
    n = 128
    δt = 0.125
    τ = n * δt / 2
    impulse = zeros(n)
    impulse[n ÷ 2 + 1] = 1
    real_ω = (2π / (n * δt)) .* collect(0:n÷2)
    complex_ω = 2π .* collect(FFTW.fftfreq(n, 1 / δt))
    for (ω, spectrum) in ((real_ω, FFTW.rfft(impulse)),
                          (complex_ω, FFTW.fft(impulse)))
        phase = Processing.spectral_phase(ω, spectrum, τ)
        error = maximum(abs.(phase))
        legacy = unwrap(angle.(spectrum); dims=1) .- ω .* τ
        legacy_span = maximum(legacy) - minimum(legacy)
        @info "Centered impulse phase" bins=length(ω) error legacy_span
        @test error < 1e-13
        @test legacy_span > 50  # Not an irrelevant constant phase offset.
    end

    ω = collect(range(-3.0, 3.0; length=65))
    τ = 13.4  # Not an integer/half-window sample delay.
    φ = @. 0.17 + 0.08 * ω + 0.05 * ω^2
    spectrum = @. exp(-ω^2 / 2) * cis(φ - ω * τ)
    fields = cat(hcat(spectrum, spectrum .* cis(0.3)),
                 hcat(spectrum .* cis(0.2), spectrum); dims=3)
    expected = cat(hcat(φ, φ .+ 0.3), hcat(φ .+ 0.2, φ); dims=3)
    original = copy(fields)
    phase = Processing.spectral_phase(ω, fields, τ)
    error = maximum(abs.(phase .- expected))
    @info "Analytic chirp phase" error
    @test error < 8e-14
    @test size(phase) == size(fields)
    @test fields == original
    @test Processing.getφ(ω, fields, τ) == phase

    # Empty bins have no physical phase. In the occupied band compare phase
    # variation, allowing only the physically arbitrary constant branch.
    band = 21:45
    sparse = zero(spectrum)
    sparse[band] .= spectrum[band]
    phase = Processing.spectral_phase(ω, sparse, τ)
    @test maximum(abs.((phase[band] .- phase[first(band)]) .-
                      (φ[band] .- φ[first(band)]))) < 8e-14
end

@testset "Time-bandwidth product sums modal intensities" begin
    # Independent scalar half-height oracle, not the production interpolation.
    function halfwidth(f)
        lo, hi = 0.0, 10.0
        half = f(0.0) / 2
        for _ in 1:70
            mid = (lo + hi) / 2
            if f(mid) > half
                lo = mid
            else
                hi = mid
            end
        end
        (lo + hi) / 2
    end
    function exact_mixture(weight)
        th = halfwidth(t -> exp(-t^2) + weight * exp(-t^2 / 4))
        ωh = halfwidth(ω -> exp(-ω^2) + 4weight * exp(-4ω^2))
        2th * ωh / π
    end

    n = 32768
    δt = 512 / n
    t = collect(-n÷2:n÷2-1) .* δt
    ω = 2π .* collect(FFTW.fftfreq(n, 1 / δt))
    grid = DiagnosticGrid(t, ω)
    weights = [0.5, 2.0]
    temporal = Array{ComplexF64}(undef, n, 2, length(weights))
    for (save, weight) in enumerate(weights)
        temporal[:, 1, save] .= exp.(-t.^2 ./ 2)
        temporal[:, 2, save] .= sqrt(weight) .* exp.(-t.^2 ./ 8)
    end
    spectral = FFTW.fft(temporal, 1)
    expected = exact_mixture.(weights)
    per_mode = Processing.time_bandwidth(grid, spectral)
    summed = Processing.time_bandwidth(grid, spectral; sumdims=2)
    @test size(per_mode) == (2, 2)
    @test size(summed) == (2,)
    # Linear interpolation at the sampled half-height crossings has an O(Δ²)
    # error; this tolerance is separate from the roundoff phase tests above.
    error = maximum(abs.(summed ./ expected .- 1))
    @info "Analytic modal time-bandwidth product" summed expected error
    @test error < 2e-4
    @test maximum(abs.(per_mode ./ (2log(2) / π) .- 1)) < 2e-4
    @test minimum(abs.(expected .- 2log(2) / π)) > 0.07
    @test Processing.time_bandwidth(grid, spectral; sumdims=(2,)) == summed
    @test isapprox(Processing.time_bandwidth(grid, spectral; sumdims=(2, 3)),
                   exact_mixture(sum(weights) / 2); rtol=2e-4)
end
end
