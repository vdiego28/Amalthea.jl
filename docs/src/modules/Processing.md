# Processing.jl

Use `Processing.spectral_phase(grid, Eω)` to extract unwrapped spectral phase
relative to the middle of the time window. The function also accepts explicit
`(ω, Eω, τ)` arguments or an output object with an optional saved-position
selection. The former spelling `Processing.getφ` remains a deprecated alias.
The FFT centering phase is removed before unwrapping, preventing artificial
phase jumps for centered pulses. Phase in bins with zero amplitude is undefined.

For multimode outputs, `Processing.time_bandwidth(grid, Eω; sumdims=2)` measures
the widths after summing modal intensities in both time and frequency. Omitting
`sumdims` returns the per-mode values.

```@autodocs
Modules = [Processing]
```
