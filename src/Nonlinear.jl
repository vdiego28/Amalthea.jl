module Nonlinear
import Luna
import Luna.PhysData: ε_0, e_ratio
import Luna: Maths, Utils, Config
import FFTW
import LinearAlgebra: mul!, ldiv!

# ─── Rust FFI helpers for Raman ──────────────────────────────────────────────
#
# The time-domain Raman solver in luna-rust computes the SDO ADE response in
# O(Nt) via an exponential integrator, replacing the O(Nt log Nt) FFT convolution
# when LUNA_USE_RUST_RAMAN=1.  Eligibility check (CombinedRamanResponse with all-SDO
# components and density-independent τ2) lives in Interface.jl where Raman types are
# visible; this module only provides the ccall wrappers, which require a module-level
# const for the library path (Julia's ccall lowering constraint — see ec76fd2).

function _libluna_rust_path()
    libname = if Sys.iswindows()
        "luna_rust.dll"
    elseif Sys.isapple()
        "libluna_rust.dylib"
    else
        "libluna_rust.so"
    end
    joinpath(Utils.lunadir(), "luna-rust", "target", "release", libname)
end

const _LIBLUNA_RUST = _libluna_rust_path()

"""
Mutable wrapper around a heap-allocated `TimeDomainRamanSolver` in the Rust shared
library.  A GC finalizer calls `free_raman_solver` when the handle is no longer
reachable, so the Rust heap allocation is always reclaimed.
"""
mutable struct RustRamanHandle
    ptr::Ptr{Cvoid}
    function RustRamanHandle(ptr::Ptr{Cvoid})
        h = new(ptr)
        finalizer(h) do self
            if self.ptr != C_NULL
                ccall((:free_raman_solver, _LIBLUNA_RUST),
                      Cvoid, (Ptr{Cvoid},), self.ptr)
                self.ptr = C_NULL
            end
        end
        return h
    end
end

"""
    _make_rust_raman_handle(omegas, gammas, couplings, dt) -> Union{Nothing, RustRamanHandle}

Build a Rust-side `TimeDomainRamanSolver` from already-extracted oscillator parameters.
Called from Interface.jl (which can see Raman types) after eligibility has been verified.
Returns `nothing` when the toggle is off, the lib is missing, or init fails.
"""
function _make_rust_raman_handle(omegas::Vector{Float64}, gammas::Vector{Float64},
                                  couplings::Vector{Float64}, dt::Float64)
    Config.backend_config().raman || return nothing
    if !isfile(_LIBLUNA_RUST)
        @warn "LUNA_USE_RUST_RAMAN=1 but Rust lib not found at $_LIBLUNA_RUST — " *
              "falling back to Julia. Build with `cargo build --release` in luna-rust/."
        return nothing
    end
    isempty(omegas) && return nothing
    ptr = GC.@preserve omegas gammas couplings begin
        ccall((:init_raman_solver, _LIBLUNA_RUST),
              Ptr{Cvoid},
              (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Csize_t, Float64),
              pointer(omegas), pointer(gammas), pointer(couplings), Csize_t(length(omegas)), dt)
    end
    if ptr == C_NULL
        @warn "init_raman_solver returned null — falling back to Julia Raman."
        return nothing
    end
    RustRamanHandle(ptr)
end

# ─────────────────────────────────────────────────────────────────────────────

function KerrScalar!(out, E, fac)
    @. out += fac*E^3
end

function KerrVector!(out, E, fac)
    for i = 1:size(E,1)
        Ex = E[i,1]
        Ey = E[i,2]
        Ex2 = Ex^2
        Ey2 = Ey^2
        out[i,1] += fac*(Ex2 + Ey2)*Ex
        out[i,2] += fac*(Ex2 + Ey2)*Ey
    end
end

"""
    kerr_γ3(resp)

Extract the γ3 nonlinear-Kerr coefficient captured by a plain
`Kerr_field`/`Kerr_env` closure (below), scanning `resp` — an iterable of
response objects, e.g. a `TransModeAvg`/`TransRadial`/`TransModal`/
`TransFree`'s `.resp` tuple — for the first entry whose closure-generated
struct has a field named with a `γ3` substring, and returning its value.
Continues scanning past a response whose extracted value is exactly `0.0`
(same "keep looking for a nonzero one" behavior as the code this replaces).
Returns `0.0` if no response has a `γ3`-named field at all (e.g. `resp`
contains only Raman/plasma responses).

Centralizes what was 4 duplicated inline reflective loops across
`RK45.jl`'s native-stepper construction paths (BACKLOG.md S4 item 3;
suggestion 8's "explicit accessor seams"). Note this does *not* distinguish
`Kerr_field`/`Kerr_env` (a single-field closure) from
`Kerr_field_nothg`/`Kerr_env_thg` (which also capture a `γ3` field, plus one
more) — that distinction is a separate eligibility question
(`RK45._is_plain_kerr_resp`), checked elsewhere before the extracted value
here is actually used.
"""
function kerr_γ3(resp)
    γ3 = 0.0
    for r in resp
        for fld in fieldnames(typeof(r))
            if occursin("γ3", string(fld))
                γ3 = getfield(r, fld)
                break
            end
        end
        γ3 != 0.0 && break
    end
    return γ3
end

"Kerr response for real field"
function Kerr_field(γ3)
    Kerr = let γ3 = γ3
        function Kerr(out, E, ρ)
            if size(E,2) == 1
                KerrScalar!(out, E, ρ*ε_0*γ3)
            else
                KerrVector!(out, E, ρ*ε_0*γ3)
            end
        end
    end
end

"Kerr response for real field but without THG"
function Kerr_field_nothg(γ3, n)
    E = Array{Float64}(undef, n)
    hilbert = Maths.plan_hilbert(E)
    Kerr = let γ3 = γ3, hilbert = hilbert
        function Kerr(out, E, ρ)
            out .+= ρ*3/4*ε_0*γ3.*abs2.(hilbert(E)).*E
        end
    end
end

function KerrScalarEnv!(out, E, fac)
    @. out += 3/4*fac*abs2(E)*E
end

function KerrVectorEnv!(out, E, fac)
    for i = 1:size(E,1)
        Ex = E[i,1]
        Ey = E[i,2]
        Ex2 = abs2(Ex)
        Ey2 = abs2(Ey)
        out[i,1] += 3/4*fac*((Ex2 + 2/3*Ey2)*Ex + 1/3*conj(Ex)*Ey^2)
        out[i,2] += 3/4*fac*((Ey2 + 2/3*Ex2)*Ey + 1/3*conj(Ey)*Ex^2)
    end
end

"Kerr response for envelope"
function Kerr_env(γ3)
    Kerr = let γ3 = γ3
        function Kerr(out, E, ρ)
            if size(E,2) == 1
                KerrScalarEnv!(out, E, ρ*ε_0*γ3)
            else
                KerrVectorEnv!(out, E, ρ*ε_0*γ3)
            end
        end
    end
end

"Kerr response for envelope but with THG"
# see Eq. 4, Genty et al., Opt. Express 15 5382 (2007)
function Kerr_env_thg(γ3, ω0, t)
    C = exp.(2im*ω0.*t)
    Kerr = let γ3 = γ3, C = C
        function Kerr(out, E, ρ)
            @. out += ρ*ε_0*γ3/4*(3*abs2(E) + C*E^2)*E
        end
    end
end

"Response type for cumtrapz-based plasma polarisation, adapted from:
M. Geissler, G. Tempea, A. Scrinzi, M. Schnürer, F. Krausz, and T. Brabec, Physical Review Letters 83, 2930 (1999)."
struct PlasmaCumtrapz{R, EType, tType}
    ratefunc::R # the ionization rate function
    ionpot::Float64 # the ionization potential (for calculation of ionization loss)
    rate::tType # buffer to hold the rate
    fraction::tType # buffer to hold the ionization fraction
    phase::EType # buffer to hold the plasma induced (mostly) phase modulation
    J::EType # buffer to hold the plasma current
    P::EType # buffer to hold the plasma polarisation
    δt::Float64 # the time step
    preionfrac::Float64 # the pre-ionisation fraction
end

"""
    PlasmaCumtrapz(t, E, ratefunc, ionpot)

Construct the Plasma polarisation response for a field on time grid `t`
with example electric field like `E`, an ionization rate callable
`ratefunc` and ionization potential `ionpot`.
"""
function PlasmaCumtrapz(t, E, ratefunc, ionpot; preionfrac=0.0)
    rate = similar(t)
    fraction = similar(t)
    phase = similar(E)
    J = similar(E)
    P = similar(E)
    !(0.0 <= preionfrac <= 1.0) && throw(DomainError(preionfrac, "preionfrac must be between 0 and 1"))
    if preionfrac > 0.0
        @warn("Using preionfrac > 0.0 is not a well founded physical model. Use only after careful consideration.")
    end
    return PlasmaCumtrapz(ratefunc, ionpot, rate, fraction, phase, J, P, t[2]-t[1], preionfrac)
end

"The plasma response for a scalar electric field"
function PlasmaScalar!(Plas::PlasmaCumtrapz, E)
    Plas.ratefunc(Plas.rate, E)
    Maths.cumtrapz!(Plas.fraction, Plas.rate, Plas.δt)
    @. Plas.fraction = Plas.preionfrac + 1 - exp(-Plas.fraction)
    @. Plas.phase = Plas.fraction * e_ratio * E
    Maths.cumtrapz!(Plas.J, Plas.phase, Plas.δt)
    for ii in eachindex(E)
        if abs(E[ii]) > 0
            Plas.J[ii] += Plas.ionpot * Plas.rate[ii] * (1-Plas.fraction[ii])/E[ii]
        end
    end
    Maths.cumtrapz!(Plas.P, Plas.J, Plas.δt)
end

"""
The plasma response for a vector electric field.

We take the magnitude of the electric field to calculate the ionization
rate and fraction, and then solve the plasma polarisation component-wise
for the vector field.

A similar approach was used in: C Tailliez et al 2020 New J. Phys. 22 103038.  
"""
function PlasmaVector!(Plas::PlasmaCumtrapz, E)
    Ex = E[:,1]
    Ey = E[:,2]
    Em = @. hypot.(Ex, Ey)
    Plas.ratefunc(Plas.rate, Em)
    Maths.cumtrapz!(Plas.fraction, Plas.rate, Plas.δt)
    @. Plas.fraction = Plas.preionfrac + 1 - exp(-Plas.fraction)
    @. Plas.phase = Plas.fraction * e_ratio * E
    Maths.cumtrapz!(Plas.J, Plas.phase, Plas.δt)
    for ii in eachindex(Em)
        if abs(Em[ii]) > 0
            pre = Plas.ionpot * Plas.rate[ii] * (1-Plas.fraction[ii])/Em[ii]^2
            Plas.J[ii,1] += pre*Ex[ii]
            Plas.J[ii,2] += pre*Ey[ii]
        end
    end
    Maths.cumtrapz!(Plas.P, Plas.J, Plas.δt)
end

"Handle plasma polarisation routing to `PlasmaVector` or `PlasmaScalar`."
function (Plas::PlasmaCumtrapz)(out, Et, ρ)
    if ndims(Et) > 1
        if size(Et, 2) == 1 # handle scalar case but within modal simulation
            PlasmaScalar!(Plas, reshape(Et, size(Et,1)))
            out .+= ρ .* reshape(Plas.P, size(Et))
        else
            PlasmaVector!(Plas, Et) # vector case
            out .+= ρ .* Plas.P
        end
    else
        PlasmaScalar!(Plas, Et) # straight scalar case
        out .+= ρ .* Plas.P
    end
end

"Raman polarisation response type"
abstract type RamanPolar end

"Raman polarisation response type for a carrier resolved field"
struct RamanPolarField{TR, Tt, Thv, Tω, Tv, FTt, HTt, RH} <: RamanPolar
    r::TR # Raman response
    h::Tt # doubled buffer to hold response + padding
    ht::Thv # buffer to hold time domain response
    hω::Tω # the frequency domain Raman response function
    Eω2::Tω # buffer to hold the Fourier transform of E^2
    Pω::Tω # buffer to hold the frequency domain polarisation
    E2::Tt # buffer to hold E^2
    E2v::Tv # view into first half of E2
    P::Tt # buffer to hold the time domain polarisation
    Pout::Tt # buffer to hold the output portion of the time domain polarisation
    FT::FTt # Fourier transform plan
    HT::HTt # Hilbert transform
    thg::Bool # do we include third harmonic generation
    dt::Float64 # time step for scaling
    rust_handle::RH # Nothing (Julia FFT path) or RustRamanHandle (Rust ADE path)
end

"Raman polarisation response type for an envelope"
struct RamanPolarEnv{TR, Tt, Thv, Tω, Tv, FTt, RH} <: RamanPolar
    r::TR # Raman response
    h::Tt # doubled buffer to hold response + padding
    ht::Thv # buffer to hold time domain response
    hω::Tω # the frequency domain Raman response function
    Eω2::Tω # buffer to hold the Fourier transform of E^2
    Pω::Tω # buffer to hold the frequency domain polarisation
    E2::Tω # buffer to hold E^2
    E2v::Tv # view into first half of E2
    P::Tω # buffer to hold the time domain polarisation
    Pout::Tω # buffer to hold the output portion of the time domain polarisation
    FT::FTt # Fourier transform plan
    dt::Float64 # time step for scaling
    rust_handle::RH # always Nothing in this slice (envelope Rust path is a follow-up)
end

"""
    RamanPolarField(t, ht; thg=true)

Construct Raman polarisation response for a field on time grid `t`
using response function `r`. If `thg=false` then exclude the third
harmonic generation component of the response.
"""
function RamanPolarField(t, r; thg=true, rust_handle=nothing)
    h = zeros(length(t)*2) # note double grid size, see explanation below
    ht = view(h, 1:length(t))
    Utils.loadFFTwisdom()
    FT = FFTW.plan_rfft(h, 1, flags=Luna.settings["fftw_flag"])
    inv(FT)
    Utils.saveFFTwisdom()
    hω = FT * h
    Eω2 = similar(hω)
    Pω = similar(hω)
    E2 = similar(h)
    E2v = view(E2, 1:length(t))
    P = similar(h)
    Pout = similar(t)
    HT = Maths.plan_hilbert(Pout)
    fill!(E2, 0.0)
    RamanPolarField(r, h, ht, hω, Eω2, Pω, E2, E2v, P, Pout, FT, HT, thg, t[2] - t[1], rust_handle)
end

"""
    RamanPolarEnv(t, ht)

Construct Raman polarisation response for an envelope on time grid `t`
using response function `r`.
"""
function RamanPolarEnv(t, r)
    h = zeros(length(t)*2) # note double grid size, see explanation below
    ht = view(h, 1:length(t))
    Utils.loadFFTwisdom()
    FT = FFTW.plan_fft(h, 1, flags=Luna.settings["fftw_flag"])
    inv(FT)
    Utils.saveFFTwisdom()
    hω = FT * h
    Eω2 = similar(hω)
    Pω = similar(hω)
    E2 = similar(hω)
    P = similar(hω)
    Pout = Array{ComplexF64,}(undef,size(t))
    E2v = view(E2, 1:length(t))
    fill!(E2, 0.0)
    RamanPolarEnv(r, h, ht, hω, Eω2, Pω, E2, E2v, P, Pout, FT, t[2] - t[1], nothing)
end

"Square the field or envelope"
function sqr!(R::RamanPolarField, E)
    if !R.thg
        # see documentation for factor of 1/2 here
        R.E2v .= 1/2 .* abs2.(R.HT(E))
    else
        R.E2v .= E.^2
    end
end

function sqr!(R::RamanPolarEnv, E)
    # see documentation for factor of 1/2 here
    R.E2v .= 1/2 .* abs2.(E)
end

"Calculate Raman polarisation for field/envelope Et"
function (R::RamanPolar)(out, Et, ρ)
    # get the field as a 1D Array
    n = size(Et, 1)
    if ndims(Et) > 1
        if size(Et, 2) == 1 # handle scalar case but within modal simulation
            E = reshape(Et, n)
        else
            # handle vector case
            error("vector Raman not yet implemented")
        end
    else
        E = Et # handle straight scalar case
    end

    # square the field or envelope in first half
    # corresponding to the field/envelope grid size
    sqr!(R, E)

    # Rust fast path: zero-copy O(Nt) time-domain ADE (carrier field only).
    # E2v is real Float64 for RamanPolarField; envelope always has rust_handle===nothing.
    # The ADE integrator resets state internally on each solve call (stateless from Julia).
    rust_done = false
    if R.rust_handle !== nothing
        rh = R.rust_handle
        e2_arr = R.E2  # preserve the backing array of E2v (GC.@preserve requires symbols)
        p_arr  = R.P
        ret = GC.@preserve e2_arr p_arr begin
            ccall((:raman_solve, _LIBLUNA_RUST), Cint,
                  (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64}, Csize_t),
                  rh.ptr, pointer(R.E2v), pointer(R.P), Csize_t(length(R.E2v)))
        end
        if ret == 0
            rust_done = true
        else
            @warn "Rust raman_solve returned $ret — falling back to Julia FFT path"
        end
    end

    if !rust_done
        # Julia FFT convolution path (default).
        # update frequency domain response function `hω`.
        # we fill only up to the first half of h (using the view ht)
        # i.e. only the part corresponding to the original time grid
        # note that the response function time 0 is put into the first element of the response array
        # this ensures that causality is maintained, and no artificial delay between the field and
        # the start of the response function occurs, at each convolution point.
        R.r(R.ht, ρ)
        R.hω .= R.FT * R.h

        # convolution by multiplication in frequency domain
        # The double grid gives us accurate full convolution between the full field grid
        # and full response function. It is unnecessary for highly damped responses, like
        # in glass. But for gases with very long decay times it prevents artefacts due to
        # truncation of the response function. There is likely a more efficient way. But
        # this is safe, until we come up with one.
        # we scale to correct for missing dt*dt*df from IFFT(FFT*FFT)
        # the ifft already scales by 1/n = dt*df, so we need an additional dt
        R.Eω2 .= R.FT * R.E2
        @. R.Pω = R.hω * R.Eω2 * R.dt
        R.P .= R.FT \ R.Pω
    end

    # calculate full polarisation, extracting only the valid
    # grid region, which is the first length(E) part.
    for i = 1:length(E)
        R.Pout[i] = ρ*E[i]*R.P[i]
    end

    # copy to output in dimensions requested
    if ndims(Et) > 1
        out .+= reshape(R.Pout, size(Et))
    else
        out .+= R.Pout
    end
end

end
