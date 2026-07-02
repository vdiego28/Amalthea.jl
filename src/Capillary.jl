module Capillary
import FunctionZeros: besselj_zero
import SpecialFunctions: besselj
import StaticArrays: SVector
import Cubature: hquadrature
using Reexport
@reexport using Luna.Modes
import Luna: Maths, Grid
import Luna.PhysData: c, ε_0, μ_0, ref_index_fun, roomtemp, densityspline, sellmeier_gas
import Luna.Modes: AbstractMode, dimlimits, neff, field, Aeff, N, modeinfo
import Luna.LinearOps: make_linop, conj_clamp, neff_grid, neff_β_grid
import Luna.PhysData: wlfreq, roomtemp
import Luna.Utils: subscript, lunadir
import Logging: @warn
import Base: show

export MarcatiliMode, dimlimits, neff, field, N, Aeff

"""
    MarcatiliMode

Type representing a mode of a hollow capillary as presented in:

Marcatili, E. & Schmeltzer, R.
"Hollow metallic and dielectric waveguides for long distance optical transmission and lasers
(Long distance optical transmission in hollow dielectric and metal circular waveguides,
examining normal mode propagation)."
Bell System Technical Journal 43, 1783–1809 (1964).
"""
struct MarcatiliMode{Ta, Tcore, Tclad, LT} <: AbstractMode
    a::Ta # core radius callable as function of z only, or fixed core radius if a Number
    n::Int # azimuthal mode index
    m::Int # radial mode index
    kind::Symbol # kind of mode (transverse magnetic/electric or hybrid)
    unm::Float64 # mth zero of the nth Bessel function of the first kind
    ϕ::Float64 # overall rotation angle of the mode
    coren::Tcore # callable, returns (possibly complex) core ref index as function of ω
    cladn::Tclad # callable, returns (possibly complex) cladding ref index as function of ω
    model::Symbol # if :full, includes complete influence of complex cladding ref index
    loss::LT # Val{true}() or Val{false}() - whether to include the loss
    aeff_intg::Float64 # Pre-calculated integral fraction for effective area
end

function show(io::IO, m::MarcatiliMode)
    a = radius_string(m)
    loss = "loss=" * (m.loss == Val(true) ? "true" : "false")
    model = "model="*string(m.model)
    angle = "ϕ=$(m.ϕ/π)π"
    out = "MarcatiliMode{"*join([mode_string(m), a, loss, model, angle], ", ")*"}"
    print(io, out)
end

mode_string(m::MarcatiliMode) = string(m.kind)*subscript(m.n)*subscript(m.m)
radius_string(m::MarcatiliMode{<:Number, Tco, Tcl, LT}) where {Tco, Tcl, LT} = "a=$(m.a)"
radius_string(m::MarcatiliMode) = "a(z=0)=$(radius(m, 0))"

modeinfo(m::MarcatiliMode) = Dict(:kind => m.kind, :n => m.n, :m => m.m,
                                  :radius => radius(m, 0), :ϕ => m.ϕ, :model => m.model,
                                  :loss => m.loss == Val(true))

"""
    MarcatiliMode(a, n, m, kind, ϕ, coren, cladn; model=:full, loss=true)

Create a MarcatiliMode.

# Arguments
- `a` : Either a `Number` for constant core radius, or a function `a(z)` for variable radius.
- `n::Int` : Azimuthal mode index (number of nodes in the field along azimuthal angle).
- `m::Int` : Radial mode index (number of nodes in the field along radial coordinate).
- `kind::Symbol` : `:TE` for transverse electric, `:TM` for transverse magnetic,
                   `:HE` for hybrid mode.
- `ϕ::Float` : Azimuthal offset angle (for linearly polarised modes, this is the angle
                between the mode polarisation and the `:y` axis)
- `coren` : Callable `coren(ω; z)` which returns the refractive index of the core
- `cladn` : Callable `cladn(ω; z)` which returns the refractive index of the cladding
- `model::Symbol=:full` : If `:full`, use the complete Marcatili model which takes into
                          account the dispersive influence of the cladding refractive index.
                          If `:reduced`, use the simplified model common in the literature
- `loss::Bool=true` : Whether to include loss.

"""
function MarcatiliMode(a, n, m, kind, ϕ, coren, cladn; model=:full, loss=true)
    # chkzkwarg makes sure that coren and cladn take z as a keyword argument
    aeff_intg = Aeff_Jintg(n, get_unm(n, m, kind), kind)
    MarcatiliMode(a, n, m, kind, get_unm(n, m, kind), ϕ,
                   chkzkwarg(coren), chkzkwarg(cladn),
                   model, Val(loss), aeff_intg)
end

"""
    MarcatiliMode(a, gas, P; kwargs...)

Create a MarcatiliMode for a capillary with radius `a` which is filled with `gas` to
pressure `P`.
"""
function MarcatiliMode(a, gas, P;
                        n=1, m=1, kind=:HE, ϕ=0.0, T=roomtemp, model=:full,
                        clad=:SiO2, loss=true)
    rfg = ref_index_fun(gas, P, T)
    rfs = ref_index_fun(clad)
    coren = (ω; z) -> rfg(wlfreq(ω))
    cladn = (ω; z) -> rfs(wlfreq(ω))
    MarcatiliMode(a, n, m, kind, ϕ, coren, cladn, model=model, loss=loss)
end

"""
    MarcatiliMode(a, gas, P, cladn; kwargs...)

Create a MarcatiliMode for a capillary made of a cladding material defined by the refractive
index `cladn(ω; z)` with a core radius `a` which is filled with `gas` to pressure `P`.
"""
function MarcatiliMode(a, gas, P, cladn;
                        n=1, m=1, kind=:HE, ϕ=0.0, T=roomtemp, model=:full, loss=true)
    rfg = ref_index_fun(gas, P, T)
    coren = (ω; z) -> rfg(wlfreq(ω))
    MarcatiliMode(a, n, m, kind, ϕ, coren, cladn, model=model, loss=loss)
end

"""
    MarcatiliMode(a, coren; kwargs...)

Create a MarcatiliMode for a capillary with radius `a` with `z`-dependent gas fill determined
by `coren(ω; z)`.
"""
function MarcatiliMode(a, coren;
                        n=1, m=1, kind=:HE, ϕ=0.0, model=:full, clad=:SiO2, loss=true)
    rfs = ref_index_fun(clad)
    cladn = (ω; z) -> rfs(wlfreq(ω))
    MarcatiliMode(a, n, m, kind, ϕ, coren, cladn, model=model, loss=loss)
end


"""
    MarcatiliMode(a; kwargs...)

Create a `MarcatiliMode` for a capillary with radius `a` and no gas fill.
"""
MarcatiliMode(a; kwargs...) = MarcatiliMode(a, (ω; z) -> 1; kwargs...)

"""
    neff(m::MarcatiliMode, ω; z=0)

Calculate the complex effective index of Marcatili mode with dielectric core and arbitrary
(metal or dielectric) cladding.

Adapted from:

Marcatili, E. & Schmeltzer, R.
"Hollow metallic and dielectric waveguides for long distance optical transmission and lasers
(Long distance optical transmission in hollow dielectric and metal circular waveguides,
examining normal mode propagation)."
Bell System Technical Journal 43, 1783–1809 (1964).
"""
function neff(m::MarcatiliMode, ω; z=0)
    εcl = m.cladn(ω, z=z)^2
    εco = m.coren(ω, z=z)^2
    vn = get_vn(εcl, m.kind)
    neff(m, ω, εco, vn, radius(m, z))
end

# Dispatch on loss to make neff type stable
# m.loss = Val{true}() (returns ComplexF64)
function neff(m::MarcatiliMode{Ta, Tco, Tcl, Val{true}}, ω, εco, vn, a) where {Ta, Tcl, Tco}
    if m.model == :full
        k = ω/c
        n = sqrt(complex(εco - (m.unm/(k*a))^2*(1 - im*vn/(k*a))^2))
        return (real(n) < 1e-3) ? (1e-3 + im*clamp(imag(n), 0, Inf)) : n
    elseif m.model == :reduced
        return ((1 + (εco - 1)/2 - c^2*m.unm^2/(2*ω^2*a^2))
                    + im*(c^3*m.unm^2)/(a^3*ω^3)*vn)
    else
        error("model must be :full or :reduced")
    end
end

# m.loss = Val{false}() (returns Float64)
function neff(m::MarcatiliMode{Ta, Tco, Tcl, Val{false}}, ω, εco, vn, a) where {Ta, Tcl, Tco}
    if m.model == :full
        k = ω/c
        n = real(sqrt(εco - (m.unm/(k*a))^2*(1 - im*vn/(k*a))^2))
        return (n < 1e-3) ? 1e-3 : n
    elseif m.model == :reduced
        return real(1 + (εco - 1)/2 - c^2*m.unm^2/(2*ω^2*a^2))
    else
        error("model must be :full or :reduced")
    end
end

function neff_wg(m::MarcatiliMode{Ta, Tco, Tcl, Val{true}}, ω; z=0) where {Ta, Tcl, Tco}
    εcl = m.cladn(ω, z=z)^2
    vn = get_vn(εcl, m.kind)
    a = radius(m, z)
    if m.model == :full
        k = ω/c
        return (m.unm/(k*a))^2*(1 - im*vn/(k*a))^2
    elseif m.model == :reduced
        return c^2*m.unm^2/(2*ω^2*a^2) + im*(c^3*m.unm^2)/(a^3*ω^3)*vn
    else
        error("model must be :full or :reduced")
    end
end

function neff_wg(m::MarcatiliMode{Ta, Tco, Tcl, Val{false}}, ω; z=0) where {Ta, Tcl, Tco}
    εcl = m.cladn(ω, z=z)^2
    vn = get_vn(εcl, m.kind)
    a = radius(m, z)
    if m.model == :full
        k = ω/c
        return (m.unm/(k*a))^2*(1 - im*vn/(k*a))^2
    elseif m.model == :reduced
        return c^2*m.unm^2/(2*ω^2*a^2)
    else
        error("model must be :full or :reduced")
    end
end

function neff(m::MarcatiliMode{Ta, Tco, Tcl, Val{true}}, εco, nwg) where {Ta, Tcl, Tco}
    if m.model == :full
        return sqrt(complex(εco - nwg))
    elseif m.model == :reduced
        return complex((1 + (εco - 1)/2 - nwg))
    else
        error("model must be :full or :reduced")
    end
end

function neff(m::MarcatiliMode{Ta, Tco, Tcl, Val{false}}, εco, nwg) where {Ta, Tcl, Tco}
    if m.model == :full
        return real(sqrt(complex(εco - nwg)))
    elseif m.model == :reduced
        return real((1 + (εco - 1)/2 - nwg))
    else
        error("model must be :full or :reduced")
    end
end

function get_vn(εcl, kind)
    if kind == :HE
        (εcl + 1)/(2*sqrt(complex(εcl - 1)))
    elseif kind == :TE
        1/sqrt(complex(εcl - 1))
    elseif kind == :TM
        εcl/sqrt(complex(εcl - 1))
    else
        error("kind must be :TE, :TM or :HE")
    end
end

function get_unm(n, m, kind)
    if (kind == :TE) || (kind == :TM)
        if (n != 0)
            error("n=0 for TE or TM modes")
        end
        besselj_zero(1, m)
    elseif kind == :HE
        if n == 0
            error("n ≠ 0 for HE modes")
        end
        besselj_zero(n-1, m)
    else
        error("kind must be :TE, :TM or :HE")
    end
end

radius(m::MarcatiliMode{<:Number, Tco, Tcl, LT}, z) where {Tcl, Tco, LT} = m.a
radius(m::MarcatiliMode, z) = m.a(z)

dimlimits(m::MarcatiliMode; z=0) = (:polar, (0.0, 0.0), (radius(m, z), 2π))

# we use polar coords, so xs = (r, θ)
function field(m::MarcatiliMode, xs; z=0)
    r, θ = xs
    if m.kind == :HE
        return (besselj(m.n-1, r*m.unm/radius(m, z)) .* SVector(
            cos(θ)*sin(m.n*(θ + m.ϕ)) - sin(θ)*cos(m.n*(θ + m.ϕ)),
            sin(θ)*sin(m.n*(θ + m.ϕ)) + cos(θ)*cos(m.n*(θ + m.ϕ))
            ))
    elseif m.kind == :TE
        return besselj(1, r*m.unm/radius(m, z)) .* SVector(-sin(θ), cos(θ))
    elseif m.kind == :TM
        return besselj(1, r*m.unm/radius(m, z)) .* SVector(cos(θ), sin(θ))
    end
end

function N(m::MarcatiliMode; z=0)
    np1 = (m.kind == :HE) ? m.n : 2
    π/2 * radius(m, z)^2 * besselj(np1, m.unm)^2 * sqrt(ε_0/μ_0)
end

function Aeff_Jintg(n, unm, kind)
    den, err = hquadrature(r -> r*besselj(n-1, unm*r)^4, 0, 1)
    np1 = (kind == :HE) ? n : 2
    num = 1/4 * besselj(np1, unm)^4
    return 2π*num/den
end

Aeff(m::MarcatiliMode; z=0) = radius(m, z)^2 * m.aeff_intg


"""
    TwoPointGradient(L, p0, p1)

Callable pressure profile `p(z)` for the simple two-point gradient fill built
by [`gradient(gas,L,p0,p1)`](@ref) — a distinct, concrete type (rather than
an anonymous closure) so the native-Rust resident path (Phase 7) can detect
this *specific* closed-form z→pressure map and port it exactly (elementary
`sqrt` arithmetic, no interpolation error), as opposed to the general
multi-point `gradient(gas,Z,P)` profile (out of Phase 7 scope — falls back
to the non-native path). See `docs/native-port/MATH.md` §3.5.
"""
struct TwoPointGradient
    L::Float64
    p0::Float64
    p1::Float64
end
(pg::TwoPointGradient)(z) = z > pg.L ? pg.p1 :
                            z <= 0  ? pg.p0 :
                            sqrt(pg.p0^2 + z/pg.L*(pg.p1^2 - pg.p0^2))

"""
    GradedCoreIndex(γ, densf, pfun, dspl)

Callable core-index profile `coren(ω; z) = sqrt(1 + γ(λ_μm(ω))·densf(z))` for a
graded (pressure-gradient) gas fill, as built by [`gradient`](@ref). A distinct,
concrete type (rather than an anonymous closure) so that the native-Rust
resident path (`RustNativeStepper`, Phase 7) can detect — via dispatch on
`MarcatiliMode{<:Number, GradedCoreIndex, ...}` in the `make_linop` method
below — that `εco(ω;z)-1` is separable into a fixed per-ω `γ` array and a
scalar `densf(z)`, and build a cheap resident z-dependent linop instead of
falling back to the (non-native) per-stage Julia callback path. `pfun` is the
pressure profile `p(z)` — a [`TwoPointGradient`](@ref) for the simple
two-point `gradient` method (Phase 7's supported case) or `nothing` for the
general multi-point `gradient` (native path not attempted for that case).
`dspl` is `PhysData.densityspline`'s density-of-pressure spline (`densf(z) =
dspl(pfun(z))`) exposed directly so the native path can **transfer** its
`(x,y,D)` to Rust rather than re-fitting a different spline through sampled
values — real-gas density (via CoolProp) is not perfectly smooth at the
scale a from-scratch refit would need to resolve, so re-fitting doesn't
converge; transferring the identical piecewise-cubic sidesteps this
entirely (see `docs/native-port/MATH.md` §3.5). Purely additive: calling a
`GradedCoreIndex` gives byte-identical results to the old anonymous-closure
`coren`.
"""
struct GradedCoreIndex{Γ, D, P, S}
    γ::Γ
    densf::D
    pfun::P
    dspl::S
end
(g::GradedCoreIndex)(ω; z) = sqrt(1 + g.γ(wlfreq(ω)*1e6)*g.densf(z))

"""
    gradient(gas, L, p0, p1; T=roomtemp)

Convenience function to create density and core index profiles for
simple two-point gradient fills defined by the waveguide length `L` and the pressures at
`z=0` and `z=L`.
"""
function gradient(gas, L, p0, p1; T=roomtemp)
    γ = sellmeier_gas(gas)
    dspl = densityspline(gas, Pmin=p0==p1 ? 0 : min(p0, p1), Pmax=max(p0, p1); T)
    pfun = TwoPointGradient(Float64(L), Float64(p0), Float64(p1))
    dens(z) = dspl(pfun(z))
    coren = GradedCoreIndex(γ, dens, pfun, dspl)
    return coren, dens
end

"""
    gradient(gas, Z, P; T=roomtemp)

Convenience function to create density and core index profiles for
multi-point gradient fills defined by positions `Z` and pressures `P`.
"""
function gradient(gas, Z, P; T=roomtemp)
    γ = sellmeier_gas(gas)
    ex = extrema(P)
    dspl = densityspline(gas, Pmin=ex[1]==ex[2] ? 0 : ex[1], Pmax=ex[2]; T)
    function p(z)
        if z <= Z[1]
            return P[1]
        elseif z >= Z[end]
            return P[end]
        else
            i = findlast(x -> x < z, Z)
            return sqrt(P[i]^2 + (z - Z[i])/(Z[i+1] - Z[i])*(P[i+1]^2 - P[i]^2))
        end
    end
    dens(z) = dspl(p(z))
    # pfun=nothing: the multi-point piecewise ramp is out of Phase 7's native
    # scope (see GradedCoreIndex docstring) -- the specialized `make_linop`
    # method below returns the plain (unwrapped) linop!/βfun! for this case,
    # so RustNativeStepper's native path is simply not attempted.
    coren = GradedCoreIndex(γ, dens, nothing, dspl)
    return coren, dens
end

# ─── Rust FFI scaffolding for MarcatiliMode dispersion ───────────────────────
#
# When LUNA_USE_RUST_DISPERSION=1 the per-step batch neff evaluation in
# neff_β_grid for constant-core-radius MarcatiliMode is offloaded to luna-rust.
# The precomputed nwg(ω) (cladding-dependent, z-independent) is stored in a
# Rust-side handle at setup time.  Per step only nco(ω; z) changes and is
# passed as a packed array; Rust returns neff_re/neff_im in-place.
#
# ccall library-path argument MUST be a module-level const (Julia constraint).

function _libluna_rust_path_cap()
    libname = if Sys.iswindows()
        "luna_rust.dll"
    elseif Sys.isapple()
        "libluna_rust.dylib"
    else
        "libluna_rust.so"
    end
    joinpath(lunadir(), "luna-rust", "target", "release", libname)
end

const _LIBLUNA_RUST_CAP = _libluna_rust_path_cap()

mutable struct RustMarcatiliHandle
    ptr::Ptr{Cvoid}
    function RustMarcatiliHandle(ptr::Ptr{Cvoid})
        h = new(ptr)
        finalizer(h) do self
            if self.ptr != C_NULL
                ccall((:free_marcatili_neff, _LIBLUNA_RUST_CAP),
                      Cvoid, (Ptr{Cvoid},), self.ptr)
                self.ptr = C_NULL
            end
        end
        return h
    end
end

function _make_rust_marcatili_handle(nwg_re_s, nwg_im_s, model_code::Cuint, loss_on::Cuint)
    get(ENV, "LUNA_USE_RUST_DISPERSION", "0") == "1" || return nothing
    if !isfile(_LIBLUNA_RUST_CAP)
        @warn "luna-rust library not found at $(_LIBLUNA_RUST_CAP) — MarcatiliMode neff_β_grid using Julia."
        return nothing
    end
    n = length(nwg_re_s)
    nwg_re = nwg_re_s; nwg_im = nwg_im_s
    ptr = GC.@preserve nwg_re nwg_im begin
        ccall((:init_marcatili_neff, _LIBLUNA_RUST_CAP), Ptr{Cvoid},
              (Ptr{Float64}, Ptr{Float64}, Csize_t, Cuint, Cuint),
              pointer(nwg_re), pointer(nwg_im), Csize_t(n), model_code, loss_on)
    end
    if ptr == C_NULL
        @warn "init_marcatili_neff returned NULL — MarcatiliMode neff_β_grid using Julia."
        return nothing
    end
    RustMarcatiliHandle(ptr)
end

#= Avoid repeated calculation of the waveguide part of the effective index for
   modes with constant core radius.  Extended from the original single-closure
   form to add:
     1. z-level memoization — the first closure call at a new z fills the full
        neff_cache for all sidcs; subsequent calls at the same z are O(1) lookups.
     2. Optional Rust batch path (LUNA_USE_RUST_DISPERSION=1) — nwg is passed
        to the Rust handle at setup; only nco(ω; z) is supplied per step.
   This is used by LinearOps.make_linop. =#
function neff_β_grid(grid,
                   mode::MarcatiliMode{<:Number, Tco, Tcl, LT} where {Tco, Tcl, LT},
                   λ0)
    sidcs  = (1:length(grid.ω))[grid.sidx]
    n_side = length(sidcs)

    # Precompute nwg once (z=0, constant radius → same at all z)
    nwg = complex(zero(grid.ω))
    for iω in sidcs
        nwg[iω] = neff_wg(mode, grid.ω[iω]; z=0)
    end

    # Packed arrays for the Rust handle (one entry per sidcs element)
    nwg_re_s = Float64[real(nwg[iω]) for iω in sidcs]
    nwg_im_s = Float64[imag(nwg[iω]) for iω in sidcs]
    model_code = mode.model == :full ? Cuint(0) : Cuint(1)
    loss_on    = mode.loss isa Val{true} ? Cuint(1) : Cuint(0)
    rh = _make_rust_marcatili_handle(nwg_re_s, nwg_im_s, model_code, loss_on)

    # Per-step scratch buffers (reused across calls)
    nco_re_s   = zeros(Float64, n_side)
    nco_im_s   = zeros(Float64, n_side)
    neff_re_s  = zeros(Float64, n_side)
    neff_im_s  = zeros(Float64, n_side)
    neff_cache = complex(zero(grid.ω))
    cache_z    = Ref(NaN)

    function _ensure!(z)
        cache_z[] == z && return
        # Fill nco(ω; z) for every packed grid point
        for (k, iω) in enumerate(sidcs)
            nc = mode.coren(grid.ω[iω], z=z)
            nco_re_s[k] = real(nc)
            nco_im_s[k] = imag(nc)
        end
        rust_done = false
        if rh !== nothing
            rh_ptr = rh.ptr
            nco_re = nco_re_s; nco_im = nco_im_s
            neff_re = neff_re_s; neff_im = neff_im_s
            ret = GC.@preserve nco_re nco_im neff_re neff_im begin
                ccall((:marcatili_neff_vector, _LIBLUNA_RUST_CAP), Cint,
                      (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Csize_t),
                      rh_ptr, pointer(nco_re), pointer(nco_im),
                      pointer(neff_re), pointer(neff_im), Csize_t(n_side))
            end
            if ret == 0
                for (k, iω) in enumerate(sidcs)
                    neff_cache[iω] = complex(neff_re_s[k], neff_im_s[k])
                end
                rust_done = true
            else
                @warn "marcatili_neff_vector returned $ret — falling back to Julia."
            end
        end
        if !rust_done
            for (k, iω) in enumerate(sidcs)
                nc = complex(nco_re_s[k], nco_im_s[k])
                neff_cache[iω] = neff(mode, nc^2, nwg[iω])
            end
        end
        cache_z[] = z
    end

    _neff_cl = let
        (iω; z) -> (_ensure!(z); neff_cache[iω])
    end
    _β_cl = let ω=grid.ω
        (iω; z) -> (_ensure!(z); ω[iω]/c * real(neff_cache[iω]))
    end
    _neff_cl, _β_cl
end

# Collection of modes with fixed core radius
FixedCoreCollection = Union{
    Tuple{Vararg{MarcatiliMode{<:Number, Tco, Tcl, LT}} where {Tco, Tcl, LT}},
    AbstractArray{MarcatiliMode{<:Number, Tco, Tcl, LT} where {Tco, Tcl, LT}}
    }

function neff_grid(grid, modes::FixedCoreCollection, λ0; ref_mode=1)
    nwg = Array{ComplexF64, 2}(undef, (length(grid.ω), length(modes)))
    sidcs = (1:length(grid.ω))[grid.sidx]
    for (i, mi) in enumerate(modes)
        for iω in sidcs
            nwg[iω, i] = neff_wg(mi, grid.ω[iω]; z=0)
        end
    end
    _neff = let nwg=nwg, ω=grid.ω, modes=modes
        _neff(iω, iim; z) = neff(modes[iim], modes[iim].coren(ω[iω], z=z)^2, nwg[iω, iim])
    end
    _neff
end

"""
    transmission(a, λ, L; kind=:HE, n=1, m=1)

Calculate the transmission through a capillary with core radius `a` and length `L` at the
wavelength `λ` when propagating the `MarcatiliMode` defined by `kind`, `n` and `m`.
"""
function transmission(a, λ, L; kind=:HE, n=1, m=1)
    # TODO hardcoded fill needs to be updated if using absorbing materials
    mode = MarcatiliMode(a, :He, 0; n=n, m=m, kind=kind)
    Modes.transmission(mode, wlfreq(λ), L)
end

# ─── Phase 7: z-dependent linop for graded-core, constant-radius MarcatiliMode ──
#
# See docs/native-port/MATH.md §3.5 and docs/native-port/BETA1_ANALYTIC.md.
# `ZDepLinopMarcatili` wraps the ordinary z-dependent `linop!(out,z)` closure
# (so it behaves identically everywhere a z-dependent linop is used —
# `LinearOps.make_prop!`, `RustPreconStepper`, the plain Julia `PreconStepper`)
# while additionally carrying the precomputed metadata `RustNativeStepper`
# needs to build a resident native z-dependent linop.
#
# `dspl_x`/`dspl_y`/`dspl_d` are `densityspline`'s own knots/values/first-
# derivatives, **transferred** to Rust rather than re-fit: an earlier version
# of this design LUT'd `dens(pressure)` by resampling and refitting a fresh
# natural cubic spline, which never converged — real-gas density (via
# CoolProp, inside `PhysData.density`) is not perfectly smooth at the scale
# a from-scratch refit needs, and worse, `dspl` is *itself* already a cubic
# spline, so refitting is a spline-of-a-spline: the error concentrates at
# `dspl`'s own knots and shrinks only ~O(h), not ~O(h⁴), so no amount of
# resampling closes the gap (see PORT_LOG for the full postmortem, and the
# z-domain postmortem before that — even N=262144 uniform z-samples left
# ~1e-10 error near z=0, and adaptive z-refinement hit ill-conditioning
# before converging). Transferring the *identical* piecewise cubic
# sidesteps the whole problem: Rust's `dens(pressure)` becomes exactly
# Julia's `dspl(pressure)` (reassociation tier), not an approximation of it.
#
# `β1(z) = d/dω[ω/c·Re(neff(ω,z))]|_{ω0}` is **not** LUT'd — `εco(ω;z)-1 =
# γ(λ(ω))·dens(z)` is separable and `nwg(ω)` is z-independent (constant
# radius), so β1(z) collapses to a closed form in the *single* scalar
# `dens(z)`, needing only 4 z-independent constants computed once here:
# `γ0=γ(λ(ω0))`, `dγ0=dγ/dω|_{ω0}`, `nwg0=nwg(ω0)`, `dnwg0=dnwg/dω|_{ω0}`.
# These are computed to near-BigFloat precision (`Maths.derivative` fed a
# `BigFloat` argument — the same adaptive-FD algorithm Julia already uses
# for `dispersion`, just with the rounding floor pushed far below `Float64`
# epsilon by the wider input type), not hand-derived symbolically: `γ`/
# `cladn` are opaque per-gas/per-glass closures (arbitrary Sellmeier terms,
# or user-registered materials), so a hand-derived chain rule would need to
# special-case every material, whereas BigFloat promotion differentiates
# *whatever* closure is passed in, generically and exactly. See
# `docs/native-port/BETA1_ANALYTIC.md` for the derivation, the adaptive-FD
# noise floor this replaces, and verification against both Julia's own
# `dispersion` and an independent BigFloat cross-check.
struct ZDepLinopMarcatili{F, DF}
    linop!::F
    densf::DF        # dens(z) = dspl(pfun(z))
    L::Float64
    p0::Float64
    p1::Float64
    dspl_x::Vector{Float64}
    dspl_y::Vector{Float64}
    dspl_d::Vector{Float64}
    gamma::Vector{Float64}
    nwg_re::Vector{Float64}
    nwg_im::Vector{Float64}
    omega::Vector{Float64}
    sidx::BitVector
    model::Symbol
    loss::Bool
    ω0::Float64
    γ0::Float64
    dγ0::Float64
    nwg0_re::Float64
    nwg0_im::Float64
    dnwg0_re::Float64
    dnwg0_im::Float64
end
(w::ZDepLinopMarcatili)(out, z) = w.linop!(out, z)

"""
    make_linop(grid::Grid.RealGrid, mode::MarcatiliMode{<:Number,<:GradedCoreIndex,...}, λ0)

Specialized z-dependent linop for a constant-radius, graded-core
(pressure-gradient, [`gradient`](@ref)) `MarcatiliMode`. Builds the ordinary
`linop!`/`βfun!` pair exactly as the generic `LinearOps.make_linop` method
would. For the two-point gradient (`mode.coren.pfun isa TwoPointGradient`),
additionally wraps `linop!` in [`ZDepLinopMarcatili`](@ref) carrying the
extra metadata the native-Rust resident path (Phase 7) needs; for the
general multi-point gradient (`pfun === nothing`), returns the plain
`linop!`/`βfun!` unwrapped — out of Phase 7's scope, so `RustNativeStepper`'s
native path is simply not attempted (no behavior change from before Phase 7).
Behaviourally identical to the generic method for every existing caller
either way (the wrapper is directly callable as `linop!(out,z)`) — only
`RustNativeStepper` inspects the wrapper type.
"""
function make_linop(grid::Grid.RealGrid,
                     mode::MarcatiliMode{<:Number, <:GradedCoreIndex, Tcl, LT} where {Tcl, LT},
                     λ0)
    sidcs = (1:length(grid.ω))[grid.sidx]
    neff, β = neff_β_grid(grid, mode, λ0)
    ω0 = wlfreq(λ0)
    linop! = let neff=neff, ω=grid.ω, mode=mode, ω0=ω0
        function linop!(out, z)
            fill!(out, 0.0)
            β1 = dispersion(mode, 1, ω0, z=z)::Float64
            for iω in sidcs
                nc = conj_clamp(neff(iω; z=z), ω[iω])
                out[iω] = -im*(ω[iω]/c*nc - ω[iω]*β1)
            end
        end
    end
    βfun! = let β=β, ω=grid.ω
        function βfun!(out, z)
            fill!(out, 1.0)
            for iω in sidcs
                out[iω] = β(iω; z=z)
            end
        end
    end

    pfun = mode.coren.pfun
    pfun isa TwoPointGradient || return linop!, βfun!

    # ── Phase 7 metadata for the native resident path (MATH.md §3.5) ────────
    γ = mode.coren.γ
    densf = mode.coren.densf
    dspl = mode.coren.dspl
    n_full = length(grid.ω)
    gamma_arr = zeros(n_full)
    nwg_re = zeros(n_full)
    nwg_im = zeros(n_full)
    for iω in sidcs
        gamma_arr[iω] = γ(wlfreq(grid.ω[iω])*1e6)
        nwg = neff_wg(mode, grid.ω[iω]; z=0.0)
        nwg_re[iω] = real(nwg)
        nwg_im[iω] = imag(nwg)
    end

    L, p0, p1 = pfun.L, pfun.p0, pfun.p1

    # ── β1(z) closed form: 4 z-independent constants (BETA1_ANALYTIC.md) ────
    # γ0, dγ0: γ(λ_μm(ω)) evaluated/differentiated at ω0. `Maths.derivative`
    # fed a BigFloat ω0 reuses Julia's own adaptive-FD algorithm, but with
    # the rounding floor pushed far below Float64 epsilon by the wider type
    # — no need to know γ's internal Sellmeier coefficients.
    γ_of_ω(ω) = γ(wlfreq(ω)*1e6)
    γ0 = γ_of_ω(ω0)
    dγ0 = Float64(Maths.derivative(γ_of_ω, BigFloat(ω0), 1))

    # nwg0, dnwg0: the z-independent waveguide term (constant core radius),
    # already model-correct (`neff_wg` branches :full/:reduced internally).
    nwg0 = neff_wg(mode, ω0; z=0.0)
    nwg_re_of_ω(ω) = real(neff_wg(mode, ω; z=0.0))
    nwg_im_of_ω(ω) = imag(neff_wg(mode, ω; z=0.0))
    dnwg0_re = Float64(Maths.derivative(nwg_re_of_ω, BigFloat(ω0), 1))
    dnwg0_im = Float64(Maths.derivative(nwg_im_of_ω, BigFloat(ω0), 1))

    wrapped = ZDepLinopMarcatili(linop!, densf, L, p0, p1,
                                  dspl.x, dspl.y, dspl.D,
                                  gamma_arr, nwg_re, nwg_im,
                                  collect(Float64, grid.ω), BitVector(grid.sidx),
                                  mode.model, mode.loss isa Val{true},
                                  ω0, γ0, dγ0, real(nwg0), imag(nwg0), dnwg0_re, dnwg0_im)
    return wrapped, βfun!
end

end
