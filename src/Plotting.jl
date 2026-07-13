module Plotting
import Amalthea: Grid, Maths, PhysData, Processing
import Amalthea.PhysData: wlfreq, c, ε_0
import Amalthea.Output: AbstractOutput
import Amalthea.Processing: makegrid, getIω, getEω, getEt, nearest_z
import FFTW
import Printf: @sprintf

#=
The functions below require PyPlot, which is a weak dependency of Amalthea (see
[weakdeps]/[extensions] in Project.toml). Their bodies are supplied by
`ext/AmaltheaPyPlotExt.jl`, which is only loaded when a user does `using PyPlot`
alongside `using Amalthea` — so `using Amalthea` on its own never touches PyCall
(see docs/dev/BACKLOG.md / CLAUDE.md for why: PyCall embeds its own libpython,
which crashes if Amalthea is loaded from Python via juliacall/PythonCall).
=#

"""
    displayall()

`display` all currently open PyPlot figures. Requires `using PyPlot`.
"""
function displayall end

"""
    cmap_white(cmap, N=512, n=8)

Replace the lowest colour stop of `cmap` (after splitting into `n` stops) with white and
create a new colourmap with `N` stops. Requires `using PyPlot`.
"""
function cmap_white end

"""
    cmap_colours(num, cmap="viridis"; cmin=0, cmax=0.8)

Make an array of `num` different colours that follow the colourmap `cmap` between the values
`cmin` and `cmax`. Requires `using PyPlot`.
"""
function cmap_colours end

"""
    subplotgrid(N, portrait=true, kwargs...)

Create a figure with `N` subplots laid out in a grid that is as close to square as possible.
If `portrait` is `true`, try to lay out the grid in portrait orientation (taller than wide),
otherwise landscape (wider than tall). Requires `using PyPlot`.
"""
function subplotgrid end

"""
    get_modes(output)

Determine whether `output` contains a multimode simulation, and if so, return the names
of the modes.
"""
function get_modes(output)
    t = output["simulation_type"]["transform"]
    !startswith(t, "TransModal") && return false, nothing
    lines = split(t, "\n")
    modeline = findfirst(li -> startswith(li, "  modes:"), lines)
    endline = findnext(li -> !startswith(li, " "^4), lines, modeline+1)
    mlines = lines[modeline+1 : endline-1]
    labels = [match(r"{([^,]*),", li).captures[1] for li in mlines]
    angles = zeros(length(mlines))
    for (ii, li) in enumerate(mlines)
        m = match(r"ϕ=(-?[0-9]+.[0-9]+)π", li)
        isnothing(m) && continue # no angle information in mode label)
        angles[ii] = parse(Float64, m.captures[1])
    end
    if !all(angles .== 0)
        for i in eachindex(labels)
            if startswith(labels[i], "HE")
                if angles[i] == 0
                    θs = "x"
                elseif angles[i] == 0.5
                    θs = "y"
                else
                    θs = "$(angles[i])π"
                end
                labels[i] *= " ($θs)"
            end
        end
    end
    return true, labels
end

"""
    stats(output; kwargs...)

Plot all statistics available in `output`. Additional `kwargs` are passed onto `plt.plot()`.
Requires `using PyPlot`.
"""
function stats end

"""
    should_log10(A, tolfac=10)

For multi-line plots, determine whether data for different lines contained in `A` spans
a sufficiently large range that a logarithmic scale should be used. By default, this is the
case when there is any point where the lines are different by more than a factor of 10.
"""
function should_log10(A, tolfac=10)
    mi = minimum(A; dims=2)
    ma = maximum(A; dims=2)
    any(ma./mi .> 10)
end

window_str(::Nothing) = ""
window_str(win::NTuple{4, Number}) = @sprintf("%.1f nm to %.1f nm", 1e9.*win[2:3]...)
window_str(win::NTuple{2, Number}) = @sprintf("%.1f nm to %.1f nm", 1e9.*win...)
window_str(window) = "custom bandpass"

"""
    prop_2D(output, specaxis=:f)

Make false-colour propagation plots for `output`, using spectral x-axis `specaxis` (see
[`getIω`](@ref)). For multimode simulations, create one figure for each mode plus one for
the sum of all modes.

# Keyword arguments
- `λrange::Tuple(Float64, Float64)` : x-axis limits for spectral plot (wavelength in metres)
- `trange::Tuple(Float64, Float64)` : x-axis limits for time-domain plot (time in seconds)
- `dBmin::Float64` : lower colour-scale limit for logarithmic spectral plot
- `resolution::Real` smooth the spectral energy density as defined by [`getIω`](@ref).

Requires `using PyPlot`.
"""
function prop_2D end

modeidcs(m::Int, ml) = [m]
modeidcs(m::Symbol, ml) = (m == :sum) ? [] : error("modes must be :sum, a single integer, or iterable")
modeidcs(m::Nothing, ml) = 1:length(ml)
modeidcs(m, ml) = m

# Helper function to convert λrange to the correct numbers depending on specaxis
function getspeclims(λrange, specaxis)
    if specaxis == :f
        specxfac = 1e-15
        speclims = (specxfac*c/maximum(λrange), specxfac*c/minimum(λrange))
        speclabel = "Frequency (PHz)"
    elseif specaxis == :ω
        specxfac = 1e-15
        speclims = (specxfac*wlfreq(maximum(λrange)), specxfac*wlfreq(minimum(λrange)))
        speclabel = "Angular frequency (rad/fs)"
    elseif specaxis == :λ
        specxfac = 1e9
        speclims = λrange .* specxfac
        speclabel = "Wavelength (nm)"
    else
        error("Unknown specaxis $specaxis")
    end
    return speclims, speclabel, specxfac
end

"""
    time_1D(output, zslice, y=:Pt, kwargs...)

Create lineplots of time-domain slice(s) of the propagation.

The keyword argument `y` determines
what is plotted: `:Pt` (power, default), `:Esq` (squared electric field) or `:Et` (electric field).

The keyword argument `modes` selects which modes (if present) are to be plotted, and can be
a single index, a `range` or `:sum`. In the latter case, the sum of modes is plotted.

The keyword argument `oversampling` determines the amount of oversampling done before plotting.

Other `kwargs` are passed onto `plt.plot`. Requires `using PyPlot`.
"""
function time_1D end

# Automatically find power unit depending on scale of electric field.
function power_unit(Pt, y=:Pt)
    units = ["kW", "MW", "GW", "TW", "PW"]
    Pmax = maximum(Pt)
    oom = clamp(floor(Int, log10(Pmax)/3), 1, 5) # maximum unit is PW
    powerfac = 1/10^(oom*3)
    if y == :Et
        sqrt(powerfac), "$(units[oom])\$^{1/2}\$"
    else
        return powerfac, units[oom]
    end
end

"""
    spec_1D(output, zslice, specaxis=:λ, log10=true, log10min=1e-6)

Create lineplots of spectral-domain slices of the propagation.

The x-axis is determined by `specaxis` (see [`getIω`](@ref)).

If `log10` is true, plot on a logarithmic scale, with a y-axis range of `log10min`.

The keyword argument `modes` selects which modes (if present) are to be plotted, and can be
a single index, a `range` or `:sum`. In the latter case, the sum of modes is plotted.

Other `kwargs` are passed onto `plt.plot`. Requires `using PyPlot`.
"""
function spec_1D end

dashes = [(0, (10, 1)),
          (0, (5, 1)),
          (0, (1, 0.5)),
          (0, (1, 0.5, 1, 0.5, 3, 1)),
          (0, (5, 1, 1, 1))]

spectrogram(output::AbstractOutput, args...; kwargs...) = spectrogram(
    makegrid(output), output, args...; kwargs...)

"""
    spectrogram(grid, output, zslice, specaxis=:λ; kwargs...)

Compute (and, for the `t, Et` method, plot) a spectrogram. Requires `using PyPlot`.
"""
function spectrogram end

"""
    energy(output; modes=nothing, bandpass=nothing, figsize=(7, 5))

Plot the pulse energy (and conversion efficiency) as a function of propagation distance.
Requires `using PyPlot`.
"""
function energy end

"""
    auto_fwhm_arrows(ax, x, y; kwargs...)

Annotate a plot with arrows indicating the FWHM of `y(x)`. Requires `using PyPlot`.
"""
function auto_fwhm_arrows end

"""
    add_fwhm_legends(ax, unit)

Append the FWHM of each line in the legend of `ax` to its label. Requires `using PyPlot`.
"""
function add_fwhm_legends end

"""
    cornertext(ax, text; corner="ul", pad=0.02, xpad=nothing, ypad=nothing, kwargs...)

Place a `text` in the axes `ax` in the corner defined by `corner`. Padding can be
defined for `x` and `y` together via `pad` or separately via `xpad` and `ypad`. Further
keyword arguments are passed to `plt.text`.

Possible values for `corner` are `ul`, `ur`, `ll`, `lr` where the first letter
defines upper/lower and the second defines left/right. Requires `using PyPlot`.
"""
function cornertext end

"""
    pygui(state)

Forwards to `PyPlot.pygui`. Requires `using PyPlot`.
"""
function pygui end

end
