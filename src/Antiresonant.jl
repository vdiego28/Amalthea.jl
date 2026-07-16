module Antiresonant
using Reexport
import Logging: @warn
import Printf: @sprintf
import Amalthea: Capillary, Utils, Config
import Amalthea.PhysData: c, wlfreq, ref_index_fun
import Amalthea.LinearOps: neff_β_grid
@reexport using Amalthea.Modes
import Amalthea.Modes: AbstractMode, dimlimits, neff, field, Aeff, N, α, chkzkwarg, modeinfo

# ─── Rust FFI helpers for Zeisberger dispersion ──────────────────────────────
#
# When AMALTHEA_USE_RUST_DISPERSION=1 the per-step neff_β_grid computation for
# ZeisbergerMode is offloaded to amalthea.  Julia still evaluates nco(ω) and
# ncl(ω) via its own multi-term Sellmeier (ref_index_fun), then passes the
# resulting arrays to Rust which applies only the Zeisberger geometry (eq. 15).
# This guarantees near-machine-epsilon equivalence (same formula + same inputs).
#
# The ccall library-path argument MUST be a module-level const — not a struct
# field or local variable (Julia constraint, fixed in ec76fd2 / 1262e5b).

function _libamalthea_path()
    libname = if Sys.iswindows()
        "amalthea.dll"
    elseif Sys.isapple()
        "libamalthea.dylib"
    else
        "libamalthea.so"
    end
    joinpath(Utils.lunadir(), "amalthea", "target", "release", libname)
end

const _LIBAMALTHEA = _libamalthea_path()

"""
Mutable wrapper around a heap-allocated `ZeisbergerNeff` in the Rust shared
library.  A GC finalizer calls `free_zeisberger_neff` when the handle is no
longer reachable, so the Rust heap allocation is always reclaimed.
"""
mutable struct RustZeisbergerHandle
    ptr::Ptr{Cvoid}
    function RustZeisbergerHandle(ptr::Ptr{Cvoid})
        h = new(ptr)
        finalizer(h) do self
            if self.ptr != C_NULL
                ccall((:free_zeisberger_neff, _LIBAMALTHEA),
                      Cvoid, (Ptr{Cvoid},), self.ptr)
                self.ptr = C_NULL
            end
        end
        return h
    end
end

"""
    _kind_code(kind::Symbol) -> Cuint

Map a Julia mode-kind symbol to the integer code expected by Rust:
  :HE → 0, :EH → 1, :TE → 2, :TM → 3.
"""
function _kind_code(kind::Symbol)::Cuint
    if kind == :HE;  return Cuint(0)
    elseif kind == :EH; return Cuint(1)
    elseif kind == :TE; return Cuint(2)
    elseif kind == :TM; return Cuint(3)
    else
        error("_kind_code: unknown mode kind $kind")
    end
end

"""
    _loss_args(loss) -> (Cuint, Float64)

Encode a Julia loss value into `(loss_on, loss_scale)` for the Rust FFI:
  Val{true}  → (1, 1.0)
  Val{false} → (0, 0.0)
  Number L   → (1, Float64(L))
"""
_loss_args(::Val{true})    = (Cuint(1), 1.0)
_loss_args(::Val{false})   = (Cuint(0), 0.0)
_loss_args(loss::Number)   = (Cuint(1), Float64(loss))

# ─────────────────────────────────────────────────────────────────────────────

struct ZeisbergerMode{mT<:Capillary.MarcatiliMode, LT} <: AbstractMode
    m::mT
    wallthickness::Float64
    loss::LT # Val{true}(), Val{false}() or a number (scaling factor)
end

"""
    ZeisbergerMode(args...; wallthickness, kwargs...)

Create a capillary-like mode with the effective index given by eq. (15) in [1].

`wallthickness` (mandatory kwarg) sets the thickness of the anti-resonant struts and
`loss` (optional, defaults to `true`) can be either a `Bool` (to switch on/off loss
completely) or a `Real` (to up/down-scale the loss given by the model).
 Other kwargs are passed on to the constructor of a [`Capillary.MarcatiliMode`](@ref).

[1] Zeisberger, M., Schmidt, M.A. Analytic model for the complex effective index of the
leaky modes of tube-type anti-resonant hollow core fibers. Sci Rep 7, 11761 (2017).
https://doi.org/10.1038/s41598-017-12234-5
"""
function ZeisbergerMode(args...; wallthickness, loss=true, kwargs...)
    return ZeisbergerMode(Capillary.MarcatiliMode(args...; kwargs...),
                          wallthickness, wraptype(loss))
end

function ZeisbergerMode(m::Capillary.MarcatiliMode; wallthickness, loss=true)
    ZeisbergerMode(m, wallthickness, wraptype(loss))
end

wraptype(loss::Bool) = Val(loss)
wraptype(loss::Real) = loss
wraptype(loss) = throw(
    ArgumentError("loss has to be a Bool or Real, not $(typeof(loss))"))

# Effective index is given by eq (15) in [1]
neff(m::ZeisbergerMode, ω; z=0) = _neff(m.m, ω, m.wallthickness, m.loss; z=z)

# All other mode properties are identical to a MarcatiliMode. `modeinfo` is
# included so callers that key off `:kind`/`:n`/`:m` (e.g. Interface.jl's
# `needfull`/`_findmode`/`needpol_modes`) see the wrapped mode's values
# instead of the generic empty-Dict `AbstractMode` fallback.
for fun in (:Aeff, :field, :N, :dimlimits, :modeinfo)
    @eval ($fun)(m::ZeisbergerMode, args...; kwargs...) = ($fun)(m.m, args...; kwargs...)
end

function _neff(m::Capillary.MarcatiliMode, ω, wallthickness, loss; z=0)
    nco = m.coren(ω, z=z)
    ncl = m.cladn(ω, z=z)
    ϵ = ncl^2 / nco^2
    k0 = ω / c
    ka = k0*nco
    ϕ = k0*wallthickness*sqrt(ncl^2 - nco^2)
    σ = 1/(ka*Capillary.radius(m, z))
    if (m.kind == :TE) || (m.kind == :TM)
        A = m.unm^2/2;
        if m.kind == :TE
            B = A/(sqrt(ϵ - 1))/tan(ϕ);
            C = m.unm^4/8 + 2*m.unm^2/(ϵ - 1)/tan(ϕ)^2
            D = m.unm^3*(1+1/tan(ϕ)^2)/(ϵ - 1)
        elseif m.kind == :TM
            B = A*ϵ/(sqrt(ϵ - 1))/tan(ϕ)
            C = m.unm^4/8 + 2*m.unm^2*ϵ^2/(ϵ - 1)/tan(ϕ)^2
            D = m.unm^3*ϵ^2*(1+1/tan(ϕ)^2)/(ϵ - 1)
        end
        return __neff(A, B, C, D, σ, nco, loss)
    else
        if m.kind == :EH
            s = 1
        elseif m.kind == :HE
            s = -1
        end
        A = m.unm^2/2
        B = m.unm^2*(ϵ+1)/(sqrt(ϵ-1)*tan(ϕ))
        C = (m.unm^4/8
            + m.unm^2*m.m*s/2
            + (m.unm^2/4*(2+m.m*s)*(ϵ+1)^2/(ϵ-1)
            - m.unm^4*(ϵ-1)/(8*m.m))*1/tan(ϕ)^2)
        D = m.unm^3/2 * (ϵ^2+1)/(ϵ-1) * (1 + 1/tan(ϕ)^2)
        __neff(A, B, C, D, σ, nco, loss)
    end
end

#= If ncl and nco are real, the type of neff depends on whether loss is included.
    Despatching on the type makes sure the type can be inferred by the compiler =#
__neff(A, B, C, D, σ, nco, loss::Val{true}) = nco*(1 - A*σ^2 - B*σ^3 - C*σ^4 + 1im*D*σ^4)
__neff(A, B, C, D, σ, nco, loss::Val{false}) = real(nco*(1 - A*σ^2 - B*σ^3 - C*σ^4))
__neff(A, B, C, D, σ, nco, loss::Number) = nco*(1 - A*σ^2 - B*σ^3 - C*σ^4 + 1im*loss*D*σ^4)


"""
    _make_rust_zeisberger_handle(m::ZeisbergerMode) -> Union{Nothing, RustZeisbergerHandle}

Build a Rust-side `ZeisbergerNeff` geometry handle from a `ZeisbergerMode`.
Returns `nothing` when the toggle is off, the lib is missing, or init fails.
"""
function _make_rust_zeisberger_handle(m::ZeisbergerMode)
    Config.backend_config().dispersion || return nothing
    if !isfile(_LIBAMALTHEA)
        @warn "AMALTHEA_USE_RUST_DISPERSION=1 but Rust lib not found at $_LIBAMALTHEA — " *
              "falling back to Julia. Build with `cargo build --release` in amalthea/."
        return nothing
    end
    kind_c = _kind_code(m.m.kind)
    loss_on, loss_scale = _loss_args(m.loss)
    ptr = ccall((:init_zeisberger_neff, _LIBAMALTHEA),
                Ptr{Cvoid},
                (Float64, Float64, Cuint, Float64, Cuint, Float64),
                m.m.unm, Float64(m.m.m), kind_c, m.wallthickness, loss_on, loss_scale)
    if ptr == C_NULL
        @warn "init_zeisberger_neff returned null — falling back to Julia dispersion."
        return nothing
    end
    RustZeisbergerHandle(ptr)
end

"""
    neff_β_grid(grid, mode::ZeisbergerMode, λ0)

Specialised `neff_β_grid` for anti-resonant fibre modes.  When
`AMALTHEA_USE_RUST_DISPERSION=1` this uses the Rust `zeisberger_neff_vector` FFI to
batch-compute the complex effective index for all positive-frequency grid points
in a single Rust call per propagation step, replacing the per-ω Julia loop.

Julia still evaluates `nco(ω)` and `ncl(ω)` via its own (multi-term) Sellmeier
model; Rust applies only the Zeisberger geometry (eq. 15).  The first call for
a given `z` triggers the Rust batch; subsequent calls for the same `z` return
cached values in O(1).

Falls back silently to the Julia `_neff` loop if the Rust lib is not built,
the toggle is off, or the ccall returns a non-zero error code.
"""
function neff_β_grid(grid, mode::ZeisbergerMode, λ0)
    sidcs  = (1:length(grid.ω))[grid.sidx]
    n_side = length(sidcs)

    # Try to build the Rust handle (returns nothing when toggle off or lib absent)
    rh = _make_rust_zeisberger_handle(mode)

    # Contiguous scratch buffers for the n_side positive-frequency points
    ω_pack   = zeros(Float64, n_side)
    nco_re_s = zeros(Float64, n_side)
    nco_im_s = zeros(Float64, n_side)
    ncl_re_s = zeros(Float64, n_side)
    ncl_im_s = zeros(Float64, n_side)
    neff_re_s = zeros(Float64, n_side)
    neff_im_s = zeros(Float64, n_side)
    neff_cache = complex(zero(grid.ω))

    # Pre-fill ω_pack — doesn't change with z
    for (k, iω) in enumerate(sidcs)
        ω_pack[k] = grid.ω[iω]
    end

    cache_z = Ref(NaN)  # NaN sentinel → "no valid cache yet"

    function _ensure!(z)
        cache_z[] == z && return
        radius = Capillary.radius(mode.m, z)

        # Fill nco/ncl arrays (Julia Sellmeier — always correct)
        for (k, iω) in enumerate(sidcs)
            ω = grid.ω[iω]
            nc = mode.m.coren(ω; z=z)
            nl = mode.m.cladn(ω; z=z)
            nco_re_s[k] = real(nc);  nco_im_s[k] = imag(nc)
            ncl_re_s[k] = real(nl);  ncl_im_s[k] = imag(nl)
        end

        rust_done = false
        if rh !== nothing
            # GC.@preserve requires plain symbols (not struct field access)
            rh_ptr   = rh.ptr
            ωa       = ω_pack
            ncore_re = nco_re_s;  ncore_im = nco_im_s
            nclad_re = ncl_re_s;  nclad_im = ncl_im_s
            neff_re  = neff_re_s; neff_im  = neff_im_s
            ret = GC.@preserve ωa ncore_re ncore_im nclad_re nclad_im neff_re neff_im begin
                ccall((:zeisberger_neff_vector, _LIBAMALTHEA),
                      Cint,
                      (Ptr{Cvoid},
                       Ptr{Float64}, Ptr{Float64}, Ptr{Float64},
                       Ptr{Float64}, Ptr{Float64},
                       Float64,
                       Ptr{Float64}, Ptr{Float64}, Csize_t),
                      rh_ptr,
                      pointer(ωa), pointer(ncore_re), pointer(ncore_im),
                      pointer(nclad_re), pointer(nclad_im),
                      radius,
                      pointer(neff_re), pointer(neff_im),
                      Csize_t(n_side))
            end
            if ret == 0
                for (k, iω) in enumerate(sidcs)
                    neff_cache[iω] = complex(neff_re_s[k], neff_im_s[k])
                end
                rust_done = true
            else
                @warn "zeisberger_neff_vector returned $ret — falling back to Julia dispersion."
            end
        end

        if !rust_done
            for iω in sidcs
                neff_cache[iω] = _neff(mode.m, grid.ω[iω], mode.wallthickness, mode.loss; z=z)
            end
        end

        cache_z[] = z
    end

    # Return the standard (iω; z) closure pair consumed by make_linop
    _neff_cl = let
        function (iω; z)
            _ensure!(z)
            neff_cache[iω]
        end
    end
    _β_cl = let ω = grid.ω
        function (iω; z)
            _ensure!(z)
            ω[iω] / c * real(neff_cache[iω])
        end
    end
    _neff_cl, _β_cl
end

struct VincettiMode{mT<:Capillary.MarcatiliMode, Tclad, LT} <: AbstractMode
    m::mT
    t::Float64 # tube wall thickness
    r_ext::Float64 # tube radius
    Ntubes::Int # number of tubes
    cladn::Tclad # cladding refractive index (constant, defaults to 1.45)
    Nterms::Int # number of terms (cladding modes) to include in the sum
    loss::LT # Val{true}(), Val{false}() or a number (scaling factor)
end

"""
    VincettiMode(a, args...; wallthickness, tube_radius, Ntubes, cladn, Nterms,
                             loss=true, kwargs...)

Create a mode with Marcatili-like mode fields but loss, dispersion and effective area given
by the semi-empirical model developed by Vincetti et al. in refs [1-3]. Arguments are identical
to `Capillary.MarcatiliMode` but with the following additions/changes as keyword arguments:

# Mandatory keyword arguments
- `wallthickness` : wall thickness of the resonators (cladding capillaries)
- `tube_radius` : radius of the resonators
- `Ntubes` : number of resonators

# Optional keyword arguments
- `cladn` : refractive index of the resonators as a function of (ω; z). Defaults
            to the refractive index of silica (SiO2).
- `Nterms` : number of resonator dielectric modes to include in the model. Defaults to 8.
- `loss` : can be `true` or `false` to switch loss on/off, or a `Real` to scale the loss.

To specify the gap between resonators, calculate the core radius with [`getRco(r_ext, N, δ)`](@ref)
or calculate the external radius of the resonators with [`getr_ext(Rco, N, δ)`](@ref).

# References

[1] L. Vincetti
Empirical formulas for calculating loss in hollow core tube lattice fibers, 
Opt. Express, OE, vol. 24, no. 10, pp. 10313-10325, May 2016, doi: 10.1364/OE.24.010313.

[2] L. Vincetti and L. Rosa
A simple analytical model for confinement loss estimation in hollow-core Tube Lattice Fibers
Opt. Express, OE, vol. 27, no. 4, pp. 5230-5237, Feb. 2019, doi: 10.1364/OE.27.005230.

[3]L. Rosa, F. Melli, and L. Vincetti
Analytical Formulas for Dispersion and Effective Area in Hollow-Core Tube Lattice Fibers
Fibers, vol. 9, no. 10, Art. no. 10, Oct. 2021, doi: 10.3390/fib9100058.
"""
function VincettiMode(Rco, args...; wallthickness, tube_radius, Ntubes,
                                    cladn=nothing, Nterms=8, loss=true, kwargs...)
    if getδ(Rco, tube_radius, Ntubes) < 0
        @warn("the given fibre parameters correspond to a negative gap between resonators")
    end
    if isnothing(cladn)
        rfs = ref_index_fun(:SiO2)
        cladn = (ω; z) -> rfs(wlfreq(ω))
    end
    return VincettiMode(Capillary.MarcatiliMode(Rco, args...; kwargs...),
                          wallthickness, tube_radius, Ntubes, chkzkwarg(cladn), Nterms, wraptype(loss))
end

# create complex effective index
neff(m::VincettiMode, ω; z=0) = neff_real(m, ω; z) + 1im*c/ω*α(m, ω; z)

α(m::VincettiMode{mT, cT, Val{true}}, ω; z=0) where {mT, cT} = log(10)/10 * CL(m, ω; z)
α(m::VincettiMode{mT, cT, Val{false}}, ω; z=0) where {mT, cT} = zero(ω)
α(m::VincettiMode{mT, cT, <:Number}, ω; z=0) where {mT, cT} = m.loss * log(10)/10 * CL(m, ω; z)

# All other mode properties are identical to a MarcatiliMode. `modeinfo` is
# included so callers that key off `:kind`/`:n`/`:m` (e.g. Interface.jl's
# `needfull`/`_findmode`/`needpol_modes`) see the wrapped mode's values
# instead of the generic empty-Dict `AbstractMode` fallback.
for fun in (:field, :N, :dimlimits, :modeinfo)
    @eval ($fun)(m::VincettiMode, args...; kwargs...) = ($fun)(m.m, args...; kwargs...)
end

function CL(λ, Rco, t, r_ext, N; cladn, Nterms=8)
    # eq. (6) of [2]
    # confinement loss in dB/m
    F = normfreq(λ, t, cladn)
    pvs = p_ν_sum(F, t, r_ext, cladn, Nterms)
    clm = CLmin(Rco, λ, t, r_ext, cladn)
    return clm*pvs
end

CL(m::VincettiMode, ω; z=0) = CL(wlfreq(ω), m.m.a, m.t, m.r_ext, m.Ntubes;
                                 cladn=real(m.cladn(ω; z)), Nterms=m.Nterms)

function FcHE(μ::Integer, ν::Integer, t, r_ext, n)
    # eq. (4) in [2]
    if ν == 1
        return (abs(0.21 + 0.175μ - 0.1/(μ-0.35)^2) * (t/r_ext)^(0.55+5e-3*sqrt(n^4-1))
                + 0.04*sqrt(μ)*t/r_ext)
    else
        return 0.3/n^0.3 * (2/ν)^1.2*abs(μ-0.8)*(t/r_ext) + ν - 1
    end
end

function FcEH(μ::Integer, ν::Integer, t, r_ext, n)
    # eq. (5) in [2]
    if ν == 1
        return (0.73 + 0.57*(μ^0.8 + 1.5)/4 - 0.04/(μ-0.35))*(t/r_ext)^(0.5 - (n-1)/(10*(μ+0.5)^0.1))
    else
        tmp = (11.5/(ν^1.2*(7.75-ν))*(0.34+μ/4*(n/1.2)^1.15)/(μ+0.2/n)^0.15
               * (t/r_ext)^(0.75 + 0.06/n^1.15 + 0.1*sqrt(1.44/n)*(ν-2)))
        return tmp + ν -1
    end
end

function CLmin(Rco, λ, t, r_ext, n)
    # eq. (3) of [1]
    3e-4 * λ^4.5/Rco^4 * (1-t/r_ext)^-12 * sqrt(n^2-1)/(t*sqrt(r_ext)) * exp(2λ/(r_ext*(n^2-1)))
end

γloss = 3e-3
L(F) = γloss^2/(γloss^2 + F^2) # eq. (2) of [2]

A(μ) = 2e3 * exp(-0.05*abs(μ-1)^2.6) # eq. (3) of [2]

function p_ν(F, ν, t, r_ext, n, Nterms=8)
    # eq. (7) of [2]
    out = A(1) * (L(F-FcHE(1, ν, t, r_ext, n)) + L(F - FcEH(1, ν, t, r_ext, n)))
    for μ in 2:Nterms
        out += A(μ) * (L(F-FcHE(μ, ν, t, r_ext, n)) + L(F - FcEH(μ, ν, t, r_ext, n)))
    end
    out
end

function p_ν_sum(F, t, r_ext, n, Nterms=8)
    out = p_ν(F, 1, t, r_ext, n, Nterms)
    for ν in 2:Nterms
        out += p_ν(F, ν, t, r_ext, n, Nterms)
    end
    out
end

normfreq(λ, t, n, nco=1) = 2t/λ*sqrt(n^2-nco) # eq. (2) of [1]

function Rco_eff(λ, Rco, t, r_ext, N, n)
    # eq. (10) in [3]
    δ = getδ(Rco, r_ext, N)
    F = normfreq(λ, t, n)
    t1 = 1.027 + 1e-3*(F + 2/F^4)
    # corrected a probable typo here: |
    #                                 V
    t2 = sqrt(Rco^2 + N/π*3/64* r_ext^2 *(1+(3+20λ/Rco)*δ/r_ext))
    # in the paper there is no square, but this would be dimensionally incorrect
    # and eq. (9) has the square
    return t1*t2
end

Rco_eff(m::VincettiMode, ω; z=0) = Rco_eff(wlfreq(ω), m.m.a, m.t, m.r_ext, m.Ntubes, real(m.cladn(ω; z)))

"""
    getδ(Rco, r_ext, N)

Calculate the gap between resonators in a single-ring antiresonant PCF with inscribed core
radius `Rco` for `N` resonators with external radius `r_ext`.
"""
getδ(Rco, r_ext, N) = 2*(sin(π/N)*(Rco + r_ext) - r_ext) # from eq. (1) in [1]

"""
    getRco(r_ext, N, δ)

Calculate the inscribed core radius of a single-ring antiresonant PCF with `N` resonators
with external radius `r_ext` and a gap between resonators of `δ`. 
"""
function getRco(r_ext, N, δ)
    # eq. (1) of [1]
    k = 1 + δ/2r_ext
    Rco = r_ext * (k/sin(π/N) - 1)
    return Rco
end

"""
    getr_ext(Rco, N, δ)

Calculate the external radius of the resonators for a single-ring antiresonant PCF with
core radius `Rco`, `N` resonators and a gap between resonators of `δ`.
"""
getr_ext(Rco, N, δ) = (δ/2 - Rco*sin(π/N))/(sin(π/N) - 1)


γdisp = 3e-2
Li(F, F_0) = (F_0^2 - F^2)/((F^2 - F_0^2)^2 + (γdisp*F)^2) # eq. (2) in [3]

function νsum(F, t, r_ext, n, Nterms=8)
    # eq. (6) of [3]
    # here only μ=1 is considered
    # ν=1 term:
    out = Li(F, FcHE(1, 1, t, r_ext, n)) + Li(F, FcEH(1, 1, t, r_ext, n))
    for ν in 2:Nterms
        out += Li(F, FcHE(1, ν, t, r_ext, n)) + Li(F, FcEH(1, ν, t, r_ext, n))
    end
    return out * A(1)
end

function Δneff(λ, Rco, t, r_ext, n, Nterms=8)
    # eq. (6) of [3]
    F = normfreq(λ, t, n)
    ρ = 1 - t/r_ext
    return 4.5e-7/ρ^4 * (λ/Rco)^2 * νsum(F, t, r_ext, n, Nterms)
end

Δneff(m::VincettiMode, ω; z=0) = Δneff(wlfreq(ω), m.m.a, m.t, m.r_ext, real(m.cladn(ω; z)), m.Nterms)

function neff_real(m::VincettiMode, ω; z=0)
    # eq. (21) of [3]
    ng = m.m.coren(ω; z) # gas index
    return (ng
            - 1/2 * (m.m.unm*c/(ω*ng*Rco_eff(m, ω; z)))^2
            + Δneff(m, ω; z))
end

Aeff(m::VincettiMode, ω; z=0) = 0.48/8π * (m.m.unm*wlfreq(ω))^2/(m.m.coren(ω; z)-neff_real(m, ω))

end # module