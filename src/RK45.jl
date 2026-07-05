module RK45
import Dates
import Logging
import Printf: @sprintf
import Luna.Utils: format_elapsed, lunadir
import FFTW
import Cubature
import ..Luna


#Get Butcher tableau etc from separate file (for convenience of changing if wanted)
include("dopri.jl")

function solve(f!, y0, t, dt, tmax;
               rtol=1e-6, atol=1e-10, safety=0.9, max_dt=Inf, min_dt=0, locextrap=true,
               norm=weaknorm,
               kwargs...)
    stepper = Stepper(f!, y0, t, dt,
                      rtol=rtol, atol=atol, safety=safety, max_dt=max_dt, min_dt=min_dt, locextrap=locextrap, norm=norm)
    return solve(stepper, tmax; kwargs...)
end

"""
One-time-per-session flag for the Phase 8 default-flip fallback warning —
avoids spamming a `@warn` on every `solve_precon` call in, e.g., a parameter
scan that repeatedly hits the same ineligible-for-native configuration.
"""
const _NATIVE_FALLBACK_WARNED = Ref(false)

"""
Records the concrete stepper type used by the most recent `solve_precon`
call (`RustNativeStepper`, `RustPreconStepper`, or `PreconStepper`). A test
hook only — `_NATIVE_FALLBACK_WARNED` alone can't answer "did *this* call use
the native path", since it's a one-time-per-session flag that stays `true`
forever after the first fallback anywhere in a test run (e.g. a deliberate
NativeIneligible test earlier in the same session). See
`test/test_native_default_workload.jl` (BACKLOG.md Phase C.2).
"""
const _LAST_STEPPER_TYPE = Ref{Any}(nothing)

function solve_precon(f!, linop, y0, t, dt, tmax;
                    rtol=1e-6, atol=1e-10, safety=0.9, max_dt=Inf, min_dt=0, locextrap=true, norm=weaknorm,
                    kwargs...)
    use_rust = get(ENV, "LUNA_USE_RUST_STEPPER", "0") == "1"
    # Phase 8: native is the default (LUNA_USE_RUST_NATIVE=0 opts back out).
    use_native = get(ENV, "LUNA_USE_RUST_NATIVE", "1") == "1"
    # Constant linop (Phases 1-6) or the narrow z-dependent case Phase 7 supports
    # (graded-core, constant-radius MarcatiliMode — see Capillary.jl / MATH.md
    # §3.5). Any other z-dependent linop (a bare `Function`, e.g. multimode or
    # radial/free-space/modal `nfun(ω;z)`) is NOT recognized here, so it falls
    # through to the non-native paths below exactly as before Phase 7 — no
    # behavior change for configurations outside the new narrow scope.
    native_ok = linop isa Array{ComplexF64} || linop isa Luna.Capillary.ZDepLinopMarcatili ||
                linop isa Luna.LinearOps.ZDepLinopFree ||
                linop isa Luna.Capillary.ZDepLinopModalTaper
    stepper = nothing
    if use_native && isfile(_LIBLUNA_RUST_RK45) && eltype(y0) == ComplexF64 && native_ok
        # Any scope restriction accumulated across Phases 1-7 (EnvGrid
        # variants, full=true modal, thg=false Raman, an ineligible extra
        # response, ...) throws `NativeIneligible` from deep inside the
        # constructor — caught here and treated as "fall back to the Julia
        # stepper for this call", not a crash, now that native is the
        # default rather than opt-in (Phase 8). A genuine bug (a nonzero FFI
        # return code, a shape/invariant mismatch, `init_native_sim`
        # returning `C_NULL`, ...) is a *different* exception type and
        # still propagates and crashes loudly, as before.
        try
            stepper = RustNativeStepper(f!, linop, y0, t, dt,
                          rtol=rtol, atol=atol, safety=safety, max_dt=max_dt, min_dt=min_dt, locextrap=locextrap, norm=norm,
                          flength=tmax)
        catch e
            e isa NativeIneligible || rethrow()
            if !_NATIVE_FALLBACK_WARNED[]
                @warn "LUNA_USE_RUST_NATIVE: falling back to the Julia stepper for a " *
                      "config outside the native port's current scope ($(e.msg)). This " *
                      "warning prints once per session; every other config still uses " *
                      "the resident Rust stepper. Set LUNA_USE_RUST_NATIVE=0 to disable " *
                      "the native path entirely."
                _NATIVE_FALLBACK_WARNED[] = true
            end
        end
    end
    if isnothing(stepper) && use_rust && isfile(_LIBLUNA_RUST_RK45) && eltype(y0) == ComplexF64
        stepper = RustPreconStepper(f!, linop, y0, t, dt,
                      rtol=rtol, atol=atol, safety=safety, max_dt=max_dt, min_dt=min_dt, locextrap=locextrap, norm=norm)
    elseif isnothing(stepper)
        stepper = PreconStepper(f!, linop, y0, t, dt,
                      rtol=rtol, atol=atol, safety=safety, max_dt=max_dt, min_dt=min_dt, locextrap=locextrap, norm=norm)
    end
    _LAST_STEPPER_TYPE[] = typeof(stepper)
    return solve(stepper, tmax; kwargs...)
end

"""
    _native_field_resync!(s)

No-op for every stepper except `RustNativeStepper` (defined alongside it,
below). `stepfun` (windowing: `Eω .*= grid.ωwin`, time-domain `twin`, ...) is
called every accepted step and mutates `s.yn` in place — for `PreconStepper`
that's the actual live state array, so the mutation carries forward into the
next step for free. For `RustNativeStepper`, `s.yn` is just a Julia-side
buffer that `native_step` *overwrites* at the top of every call from Rust's
own resident `field` (see `native.rs::native_step` — it does not read back
whatever Julia last wrote into the passed pointer). Left unsynced, every
`Luna.run`-driven simulation silently drops the windowing for the native
path entirely — invisible in every native-specific phase test (they all call
`solve()`/`step!()` directly, bypassing `stepfun`), but present in every real
simulation via `Interface.jl`/`Luna.run`. Fixed by pushing the
just-windowed `s.yn` back into Rust's resident field after every accepted
step, via `native_resync_field` — a lighter sibling of the construction-time
`set_field` FFI that updates only the resident `field` buffer and, crucially,
does NOT recompute the FSAL stage-0 RHS: Julia's own `PreconStepper` doesn't
re-evaluate the nonlinear RHS after windowing either (it keeps the
FSAL-carried last stage and only re-propagates it linearly into the new
interaction-picture frame — see `native_resync_field`'s doc comment in
`native.rs`), so recomputing k0 fresh here would introduce a new, undisclosed
divergence from Julia rather than fix the existing one.
"""
_native_field_resync!(s) = nothing

function solve(s, tmax; stepfun=donothing!, output=false, outputN=201,
                        status_period=1, repeat_limit=10)
    if output
        yout = Array{eltype(s.y)}(undef, (size(s.y)..., outputN))
        tout = range(s.t, stop=tmax, length=outputN)
        saved = 1
        yout[fill(:, ndims(s.y))..., 1] = s.y
    end

    steps = 0
    repeated = 0
    repeated_tot = 0

    Logging.@info "Starting propagation"
    start = Dates.now()
    tic = Dates.now()
    while s.tn <= tmax
        ok = step!(s)
        steps += 1
        if Dates.value(Dates.now()-tic) > 1000*status_period
            speed = s.tn/(Dates.value(Dates.now()-start)/1000)
            eta_in_s = (tmax-s.tn)/(speed)
            if eta_in_s > 356400
                Logging.@info @sprintf("Progress: %.2f %%, ETA: XX:XX:XX, stepsize %.2e, err %.2f, repeated %d/%d steps",
                s.tn/tmax*100, s.dt, s.err, repeated_tot, steps)
            else
                eta_in_ms = Dates.Millisecond(ceil(eta_in_s*1000))
                etad = Dates.DateTime(Dates.UTInstant(eta_in_ms))
                Logging.@info @sprintf("Progress: %.2f %%, ETA: %s, stepsize %.2e, err %.2f, repeated %d/%d steps",
                    s.tn/tmax*100, Dates.format(etad, "HH:MM:SS"), s.dt, s.err, repeated_tot, steps)
            end
            flush(stderr)
            tic = Dates.now()
        end
        if ok
            if output
                while (saved<outputN) && tout[saved+1] < s.tn
                    ti = tout[saved+1]
                    yout[fill(:, ndims(s.y))..., saved+1] .= interpolate(s, ti)
                    saved += 1
                end
            end
            stepfun(s.yn, s.tn, s.dtn, t -> interpolate(s, t))
            _native_field_resync!(s)
            repeated = 0
        else
            repeated += 1
            repeated_tot += 1
            if repeated > repeat_limit
                error("Reached limit for step repetition ($repeat_limit)")
            end
        end
    end
    totaltime = Dates.now()-start
    dtstring = format_elapsed(totaltime)
    Logging.@info @sprintf("Propagation finished in %s, %d steps",
                           dtstring, steps)

    if output
        return collect(tout), yout, steps
    else      
        return nothing
    end
end


mutable struct Stepper{T<:AbstractArray, F, nT}
    f!::F  # RHS function
    y::T  # Solution at current t
    yn::T  # Solution at t+dt
    yi::T  # Interpolant array (see interpolate())
    yerr::T  # solution error estimate (from embedded RK)
    ks::NTuple{7, T}  # k values (intermediate solutions for Runge-Kutta method)
    t::Float64  # current time (propagation variable)
    tn::Float64  # next time
    dt::Float64  # time step
    dtn::Float64  # time step for next step
    rtol::Float64  # relative tolerance on error
    atol::Float64  # absolute tolerance on error
    safety::Float64  # safety factor for stepsize control
    max_dt::Float64  # maximum value for dt (default Inf)
    min_dt::Float64  # minimum value for dt (default 0)
    locextrap::Bool  # true if using local extrapolation
    ok::Bool  # true if current step was successful
    err::Float64  # error metric to be compared to tol
    errlast::Float64  # error of the most recent successful step
    norm::nT # function to calculate error metric, defaults to RK45.weaknorm
end

function Stepper(f!, y0, t, dt;
                 rtol=1e-6, atol=1e-10, safety=0.9, max_dt=Inf, min_dt=0,
                 locextrap=true, norm=weaknorm)
    k1 = similar(y0)
    f!(k1, y0, t)
    ks = (k1, similar(k1), similar(k1), similar(k1), similar(k1), similar(k1), similar(k1))
    yerr = similar(y0)
    return Stepper(f!, copy(y0), copy(y0), similar(y0), yerr, ks,
        float(t), float(t), float(dt), float(dt),
        float(rtol), float(atol), float(safety), float(max_dt), float(min_dt),
        locextrap, false, 0.0, 0.0, norm)
end

mutable struct PreconStepper{T<:AbstractArray, F, P, nT}
    fbar!::F  # RHS callable
    prop!::P # linear propagator callable
    y::T  # Solution at current t
    yn::T  # Solution at t+dt
    yi::T  # Interpolant array (see interpolate())
    yerr::T  # solution error estimate (from embedded RK)
    ks::NTuple{7, T}  # k values (intermediate solutions for Runge-Kutta method)
    t::Float64  # current time (propagation variable)
    tn::Float64  # next time
    dt::Float64  # time step
    dtn::Float64  # time step for next step
    rtol::Float64  # relative tolerance on error
    atol::Float64  # absolute tolerance on error
    safety::Float64  # safety factor for stepsize control
    max_dt::Float64  # maximum value for dt (default Inf)
    min_dt::Float64  # minimum value for dt (default 0)
    locextrap::Bool  # true if using local extrapolation
    ok::Bool  # true if current step was successful
    err::Float64  # error metric to be compared to tol
    errlast::Float64  # error of the most recent successful step
    norm::nT  # function to calculate error metric, defaults to RK45.weaknorm
end

function PreconStepper(f!, linop, y0, t, dt;
                       rtol=1e-6, atol=1e-10, safety=0.9, max_dt=Inf, min_dt=0,
                       locextrap=true, norm=weaknorm)
    prop! = make_prop!(linop, y0)
    fbar! = make_fbar!(f!, prop!, y0)
    k1 = similar(y0)
    fbar!(k1, y0, t, t)
    ks = (k1, similar(k1), similar(k1), similar(k1), similar(k1), similar(k1), similar(k1))
    yerr = similar(y0)

    return PreconStepper(fbar!, prop!, copy(y0), copy(y0), similar(y0), yerr, ks,
        float(t), float(t), float(dt), float(dt), float(rtol), float(atol), float(safety),
        float(max_dt), float(min_dt), locextrap, false, 0.0, 0.0, norm)
end

function step!(s)
    evaluate!(s)

    if s.locextrap
        s.yn .= s.y
        for jj = 1:7
            b5[jj] == 0 || (s.yn .+= s.dt*b5[jj].*s.ks[jj])
        end
    end
    
    fill!(s.yerr, 0)
    for ii = 1:7
        errest[ii] == 0 || (@. s.yerr += s.dt*s.ks[ii]*errest[ii])
    end
    s.err = s.norm(s.yerr, s.y, s.yn, s.rtol, s.atol)
    s.ok = s.err <= 1
    stepcontrolPI!(s)
    if s.ok
        s.tn = s.t + s.dt
        s.ks[1] .= s.ks[end]
    else
        s.yn .= s.y
    end
    prop!_maybe(s) # propagate to new time to pass correct solution to stepfun
    return s.ok
end

function evaluate!(s::Stepper)
    # Set new time and stepsize values -- this happens at the beginning because
    # the interpolant still requires the old values after the step has finished
    s.dt = s.dtn
    s.t = s.tn
    s.y .= s.yn
    for ii = 1:6
        s.yn .= s.y
        for jj = 1:ii
            B[ii][jj] == 0 || (s.yn .+= s.dt*B[ii][jj].*s.ks[jj])
        end
        s.f!(s.ks[ii+1], s.yn, s.t+nodes[ii]*s.dt)
    end
end

function evaluate!(s::PreconStepper)
    # Set new time and stepsize values -- this happens at the beginning because
    # the interpolant still requires the old values after the step has finished
    s.y .= s.yn
    s.prop!(s.ks[1], s.t, s.tn)
    s.dt = s.dtn
    s.t = s.tn
    for ii = 1:6
        s.yn .= s.y
        for jj = 1:ii
            B[ii][jj] == 0 || (s.yn .+= s.dt*B[ii][jj].*s.ks[jj])
        end
        s.fbar!(s.ks[ii+1], s.yn, s.t, s.t+nodes[ii]*s.dt)
    end
end

prop!_maybe(s::PreconStepper) = s.prop!(s.yn, s.t, s.tn)
prop!_maybe(s) = nothing

"Interpolate solution, aka dense output."
function interpolate(s::Stepper, ti::Float64)
    if ti > s.tn
        error("Attempting to extrapolate!")
    end
    if ti == s.t
        return s.y
    elseif ti == s.tn
        return s.yn
    end
    σ = (ti - s.t)/s.dt
    σ2 = σ^2
    σ3 = σ2*σ
    σ4 = σ3*σ
    b = ntuple(ii -> σ*interpC[1,ii] + σ2*interpC[2,ii] + σ3*interpC[3,ii] + σ4*interpC[4,ii], Val(7))
    fill!(s.yi, 0)
    for ii = 1:7
         s.yi .+= s.ks[ii].*b[ii]
    end
    return @. s.y + s.dt.*s.yi
end

"Interpolate solution, aka dense output."
function interpolate(s::PreconStepper, ti::Float64)
    if ti > s.tn
        error("Attempting to extrapolate!")
    end
    if ti == s.t
        return s.y
    elseif ti == s.tn
        return s.yn
    end
    σ = (ti - s.t)/s.dt
    σ2 = σ^2
    σ3 = σ2*σ
    σ4 = σ3*σ
    b = ntuple(ii -> σ*interpC[1,ii] + σ2*interpC[2,ii] + σ3*interpC[3,ii] + σ4*interpC[4,ii], Val(7))
    fill!(s.yi, 0)
    for ii = 1:7
         s.yi .+= s.ks[ii].*b[ii]
    end
    out =  @. s.y + s.dt.*s.yi
    s.prop!(out, s.t, ti)
    return out
end

"Make propagator for the case of constant linear operator"
function make_prop!(linop::AbstractArray, y0)
    prop! = let linop=linop
        function prop!(y, t1, t2, bwd=false)
            if bwd
                @. y *= exp(linop*(t1-t2))
            else
                @. y *= exp(linop*(t2-t1))
            end
        end
    end
end

"Make propagator for the case of non-constant linear operator"
function make_prop!(linop!, y0)
    linop_int = similar(y0)
    lastt2 = [typemin(Float64)]
    function prop!(y, t1, t2, bwd=false)
        #= linop is always evaluated at later time, even for backward propagation
            therefore, linop is often evaluated at the same t2 twice in a row=#
        (lastt2[1] != t2) && linop!(linop_int, t2)
        lastt2[1] = t2
        dt = bwd ? (t1-t2) : (t2-t1)
        @. y *= exp(linop_int*dt)
    end
    return prop!
end

"Make closure for the pre-conditioned RHS function."
function make_fbar!(f!, prop!, y0)
    y = similar(y0)
    fbar! = let f! = f!, prop! = prop!, y=y
        function fbar!(out, ybar, t1, t2)
            y .= ybar
            prop!(y, t1, t2) # propagate to t2
            f!(out, y, t2) # evaluate RHS function
            prop!(out, t1, t2, true) # propagate back to t1
        end
    end
end

"Max-ish norm (from Dane Austin's code, no idea where he got it from)."
function maxnorm(yerr, y, yn, rtol, atol)
    maxerr = 0
    maxy = 0
    for ii in eachindex(yerr)
        maxerr = max(maxerr, abs(yerr[ii]))
        maxy = max(maxy, max(abs(y[ii]), abs(yn[ii])))
    end
    return maxerr/(atol + rtol*maxy)
end

"Alternative form of max-ish norm."
function maxnorm_ratio(yerr, y, yn, rtol, atol)
    m = 0
    for ii in eachindex(yerr)
        den = atol + rtol*max(abs(y[ii]), abs(yn[ii]))
        m = max(abs(yerr[ii])/den, m)
    end
    return m
end

"Semi-norm as used in DifferentialEquations.jl, see Hairer, Solving Ordinary Differential
Equations: Nonstiff Problems, eq. (4.11) (p.168 of the second revised edition)."
function normnorm(yerr, y, yn, rtol, atol)
    s = 0
    for ii in eachindex(yerr)
        s += abs2(yerr[ii]/(atol + rtol*max(abs(y[ii]), abs(yn[ii]))))
    end
    sqrt(s/length(yerr))
end

"'Weak' norm as used in fnfep."
function weaknorm(yerr, y, yn, rtol, atol)
    sy = 0
    syn = 0
    syerr = 0
    for ii in eachindex(yerr)
        sy += abs2(y[ii])
        syn += abs2(yn[ii])
        syerr += abs2(yerr[ii])
    end
    errwt = max(max(sqrt(sy), sqrt(syn)), atol)
    return sqrt(syerr)/rtol/errwt
end

"Simple proportional error controller, see e.g. Hairer eq. (4.13)."
function stepcontrolP!(s)
    if s.ok
        # if error is zero, there is no nonlinearity: increase step size by a lot
        s.dtn = s.err == 0 ? 1.5*s.dt : s.dt * min(5, s.safety*(s.err)^(-1/5))
    else
        if !isfinite(s.err) # check for NaN or Inf
            s.dtn = s.dt/2  # if we have one then we're in big trouble so halve the step size
        else
            s.dtn = s.dt * max(0.1, s.safety*(s.err)^(-1/5))
        end
    end
    steplims!(s)
end

"Proportional-integral error controller, aka Lund stabilisation.
See G. Söderlind and L. Wang, J. Comput. Appl. Math. 185, 225 (2006).
"
function stepcontrolPI!(s)
    β1 = 3/5 / 5
    β2 = -1/5 / 5
    ε = 0.8
    if s.ok
        s.errlast == 0 && (s.errlast = s.err) # if last error is zero, use current error instead
        if s.err == 0
            fac = 1.5 # zero error means no nonlinearity: increase step size by a lot 
        else
            fac = s.safety * (ε/s.err)^β1 * (ε/s.errlast)^β2
        end
        # (0.99 <= fac <= 1.01) && (fac = 1.0)
        s.dtn = fac * s.dt
        s.errlast = s.err
    else
        if !isfinite(s.err) # check for NaN or Inf
            s.dtn = s.dt/2  # if we have one then we're in big trouble so halve the step size
        else
            s.dtn = s.dt * max(0.1, s.safety*(s.err)^(-1/5))
        end
    end
    steplims!(s)
end

"Apply user-defined limits on step size."
function steplims!(s)
    if s.dtn > s.max_dt
        s.dtn = s.max_dt
    elseif s.dtn < s.min_dt
        s.dtn = s.min_dt
        s.ok = true
    end
end

function donothing!(y, z, dz, interpolant)
end

# ── Rust interaction-picture PreconStepper FFI (LUNA_USE_RUST_STEPPER=1) ───────

function _libluna_rust_path_rk45()
    libname = if Sys.iswindows(); "luna_rust.dll"
              elseif Sys.isapple(); "libluna_rust.dylib"
              else; "libluna_rust.so"; end
    joinpath(lunadir(), "luna-rust", "target", "release", libname)
end
const _LIBLUNA_RUST_RK45 = _libluna_rust_path_rk45()

"Result struct returned by precon_step_ffi — must match #[repr(C)] in Rust."
struct PreconStepResult
    ok::Cint
    dt::Float64
    t::Float64
    tn::Float64
    dtn::Float64
    err::Float64
    errlast::Float64
end

mutable struct RustPreconStepHandle
    ptr::Ptr{Cvoid}
    function RustPreconStepHandle(n::Int)
        ptr = ccall((:init_precon_step_ffi, _LIBLUNA_RUST_RK45), Ptr{Cvoid},
                    (Csize_t,), Csize_t(n))
        h = new(ptr)
        finalizer(h) do h
            p = h.ptr
            if p != C_NULL
                ccall((:free_precon_step_ffi, _LIBLUNA_RUST_RK45), Cvoid, (Ptr{Cvoid},), p)
                h.ptr = C_NULL
            end
        end
        h
    end
end

"Holds fbar! and prop! closures plus array shape for unsafe_wrap in C callbacks."
mutable struct RustStepperCallbacks
    fbar!::Any
    prop!::Any
    shape::Tuple
end

"Module-level C-callable wrapper for fbar!(k_out, y_in, t1, t2)."
function _fbar_cfun(t1::Float64, t2::Float64,
                    y_in::Ptr{ComplexF64}, k_out::Ptr{ComplexF64},
                    ::Csize_t, userdata::Ptr{Cvoid})::Cvoid
    cb = unsafe_pointer_to_objref(userdata)::RustStepperCallbacks
    y_sl = unsafe_wrap(Array, y_in,  cb.shape)
    k_sl = unsafe_wrap(Array, k_out, cb.shape)
    cb.fbar!(k_sl, y_sl, t1, t2)
    return nothing
end

"Module-level C-callable wrapper for prop!(y, t1, t2)."
function _prop_cfun(t1::Float64, t2::Float64,
                    y_inout::Ptr{ComplexF64},
                    ::Csize_t, userdata::Ptr{Cvoid})::Cvoid
    cb = unsafe_pointer_to_objref(userdata)::RustStepperCallbacks
    y_sl = unsafe_wrap(Array, y_inout, cb.shape)
    cb.prop!(y_sl, t1, t2)
    return nothing
end

# Populated in __init__ so the pointer is valid in the running session (not the precompile image).
const _FBAR_CPTR = Ref{Ptr{Cvoid}}(C_NULL)
const _PROP_CPTR = Ref{Ptr{Cvoid}}(C_NULL)

function __init__()
    _FBAR_CPTR[] = @cfunction(_fbar_cfun, Cvoid,
        (Float64, Float64, Ptr{ComplexF64}, Ptr{ComplexF64}, Csize_t, Ptr{Cvoid}))
    _PROP_CPTR[] = @cfunction(_prop_cfun, Cvoid,
        (Float64, Float64, Ptr{ComplexF64}, Csize_t, Ptr{Cvoid}))
end

mutable struct RustPreconStepper{T<:AbstractArray, F, P, nT}
    fbar!::F
    prop!::P
    y::T
    yn::T
    yi::T
    yerr::T
    ks::NTuple{7, T}
    t::Float64
    tn::Float64
    dt::Float64
    dtn::Float64
    rtol::Float64
    atol::Float64
    safety::Float64
    max_dt::Float64
    min_dt::Float64
    locextrap::Bool
    ok::Bool
    err::Float64
    errlast::Float64
    norm::nT
    _handle::RustPreconStepHandle
    _callbacks::RustStepperCallbacks
    _k_ptrs::Vector{Ptr{ComplexF64}}
end

function RustPreconStepper(f!, linop, y0, t, dt;
                            rtol=1e-6, atol=1e-10, safety=0.9, max_dt=Inf, min_dt=0,
                            locextrap=true, norm=weaknorm)
    prop!  = make_prop!(linop, y0)
    fbar!  = make_fbar!(f!, prop!, y0)
    k1     = similar(y0)
    fbar!(k1, y0, t, t)
    ks     = (k1, similar(k1), similar(k1), similar(k1), similar(k1), similar(k1), similar(k1))
    n      = length(y0)
    handle = RustPreconStepHandle(n)
    handle.ptr == C_NULL && error("init_precon_step_ffi failed (n=$n)")
    callbacks = RustStepperCallbacks(fbar!, prop!, size(y0))
    k_ptrs = Ptr{ComplexF64}[pointer(ks[i]) for i in 1:7]
    RustPreconStepper(
        fbar!, prop!, copy(y0), copy(y0), similar(y0), similar(y0), ks,
        float(t), float(t), float(dt), float(dt),
        float(rtol), float(atol), float(safety), float(max_dt), float(min_dt),
        locextrap, false, 0.0, 0.0, norm,
        handle, callbacks, k_ptrs
    )
end

function step!(s::RustPreconStepper)
    result_ref = Ref(PreconStepResult(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    callbacks = s._callbacks
    GC.@preserve s callbacks begin
        cb_ptr = pointer_from_objref(callbacks)
        for i in 1:7
            s._k_ptrs[i] = pointer(s.ks[i])
        end
        rc = ccall((:precon_step_ffi, _LIBLUNA_RUST_RK45), Cint,
            (Ptr{Cvoid},           # handle
             Ptr{ComplexF64},      # y
             Ptr{ComplexF64},      # yn
             Ptr{Ptr{ComplexF64}}, # k_ptrs[7]
             Csize_t,              # n
             Float64, Float64,     # t_old, t_new
             Float64,              # dtn
             Float64, Float64,     # rtol, atol
             Float64, Float64, Float64,  # safety, max_dt, min_dt
             Float64,              # errlast_in
             Cint,                 # locextrap
             Ptr{Cvoid},           # fbar_fn
             Ptr{Cvoid},           # prop_fn
             Ptr{Cvoid},           # userdata
             Ptr{PreconStepResult} # result
            ),
            s._handle.ptr,
            s.y, s.yn,
            s._k_ptrs,
            Csize_t(length(s.y)),
            s.t, s.tn, s.dtn,
            s.rtol, s.atol,
            s.safety, s.max_dt, s.min_dt,
            s.errlast,
            Cint(s.locextrap),
            _FBAR_CPTR[], _PROP_CPTR[],
            cb_ptr,
            result_ref
        )
        rc == 0 || error("precon_step_ffi returned error code $rc")
    end
    res = result_ref[]
    s.ok      = res.ok != 0
    s.dt      = res.dt
    s.t       = res.t
    s.tn      = res.tn
    s.dtn     = res.dtn
    s.err     = res.err
    s.errlast = res.errlast
    return s.ok
end

function interpolate(s::RustPreconStepper, ti::Float64)
    if ti > s.tn
        error("Attempting to extrapolate!")
    end
    if ti == s.t
        return s.y
    elseif ti == s.tn
        return s.yn
    end
    σ = (ti - s.t)/s.dt
    σ2 = σ^2; σ3 = σ2*σ; σ4 = σ3*σ
    b = ntuple(ii -> σ*interpC[1,ii] + σ2*interpC[2,ii] + σ3*interpC[3,ii] + σ4*interpC[4,ii], Val(7))
    fill!(s.yi, 0)
    for ii = 1:7
        s.yi .+= s.ks[ii].*b[ii]
    end
    out = @. s.y + s.dt.*s.yi
    s.prop!(out, s.t, ti)
    return out
end

# ─── Rust native interaction-picture Stepper FFI (LUNA_USE_RUST_NATIVE=1) ───────

"""
    NativeIneligible(msg)

Thrown by `RustNativeStepper`'s constructor when the requested geometry/
config is outside the native port's current scope (any of the per-phase
narrowing restrictions documented in `docs/native-port/MATH.md` — EnvGrid
variants, `full=true` modal, `thg=false` Raman, an unsupported/ineligible
extra response such as plasma or Raman on a Kerr-only geometry, ...).

This is a **distinct exception type from a bare `error()`** specifically so
`solve_precon` (Phase 8, default-flip) can catch *this* and silently fall
back to the Julia stepper, while letting a genuine bug (an FFI call
returning a nonzero error code, a shape/invariant mismatch, `init_native_sim`
returning `C_NULL`, ...) propagate and crash loudly as before. Before Phase
8, native was opt-in only, so a scope-restriction `error()` reaching the user
meant they had deliberately turned on `LUNA_USE_RUST_NATIVE` for an
unsupported config — a crash with an instructive message was the right
behavior. Now that native is the default, the exact same situation must be
reachable by any ordinary user just running a simulation, so it can no
longer be a crash: it must fall back quietly (with one warning) instead.
"""
struct NativeIneligible <: Exception
    msg::String
end
Base.showerror(io::IO, e::NativeIneligible) = print(io, "NativeIneligible: ", e.msg)

"""
    _check_density_zindependent(densityfun, flength)

Guard against silently freezing a z-varying `densityfun` at `z=0`. The
high-level `prop_capillary`/`prop_gnlse` interface can't hit this — a z-
varying fill always produces either a function-typed linop (rejected above,
see the `f!` check) or a `ZDepLinopMarcatili` (handled by the dedicated
z-dependent path, which re-evaluates `densityfun(z)` every stage). But the
low-level API allows pairing a constant `Array{ComplexF64}` linop with a
z-varying `densityfun` directly — a combination Julia's own `TransModeAvg`
handles correctly (it calls `densityfun(z)` fresh every stage) but which the
non-z-dep native paths below bake into a `z=0` constant. Sample at three
points (0, `flength/2`, `flength` when finite; otherwise 0, 1, 2 as a cheap
non-constness probe, safe because a genuinely constant `densityfun` ignores
its argument) and refuse eligibility on any mismatch.
"""
function _check_density_zindependent(densityfun, flength)
    probes = isfinite(flength) ? (0.0, flength/2, flength) : (0.0, 1.0, 2.0)
    vals = densityfun.(probes)
    d0 = vals[1]
    all(v -> isapprox(v, d0; rtol=1e-10), vals) ||
        throw(NativeIneligible("densityfun(z) is not z-independent (density at " *
              "z=$probes → $vals) but this native RHS path bakes density(0) into a " *
              "constant kerr_fac/plasma/Raman coefficient — only the dedicated " *
              "z-dependent linop path (ZDepLinopMarcatili) supports a z-varying density."))
end

struct NativeStepResult
    ok::Cint
    dt::Float64
    t::Float64
    tn::Float64
    dtn::Float64
    err::Float64
    errlast::Float64
end

mutable struct RustNativeSimHandle
    ptr::Ptr{Cvoid}
    function RustNativeSimHandle(linop::Array{ComplexF64})
        n = length(linop)
        ptr = ccall((:init_native_sim, _LIBLUNA_RUST_RK45), Ptr{Cvoid},
                    (Ptr{ComplexF64}, Csize_t), pointer(linop), Csize_t(n))
        h = new(ptr)
        finalizer(h) do h
            p = h.ptr
            if p != C_NULL
                ccall((:free_native_sim, _LIBLUNA_RUST_RK45), Cvoid, (Ptr{Cvoid},), p)
                h.ptr = C_NULL
            end
        end
        h
    end
    # Phase 7: z-dependent linop — `linop` is a `Capillary.ZDepLinopMarcatili`
    # wrapper, not an `Array{ComplexF64}`, so there is no fixed array to seed
    # with. Untyped (not `Luna.Capillary.ZDepLinopMarcatili`) because RK45.jl
    # is `include`d before Capillary.jl in Luna.jl — the caller
    # (`RustNativeStepper`) already gates this constructor on a runtime
    # `isa` check, so no eager type resolution is needed here; dispatch on
    # 2-arg vs 1-arg is unambiguous regardless. `init_native_sim` only uses
    # the seed to size `n` and populate the initial `s.linop`;
    # `ensure_linop_at` overwrites it before it is ever read (the first
    # `native_step` call), so a zero seed is exact, not an approximation.
    function RustNativeSimHandle(linop, n::Int)
        seed = zeros(ComplexF64, n)
        ptr = ccall((:init_native_sim, _LIBLUNA_RUST_RK45), Ptr{Cvoid},
                    (Ptr{ComplexF64}, Csize_t), pointer(seed), Csize_t(n))
        h = new(ptr)
        finalizer(h) do h
            p = h.ptr
            if p != C_NULL
                ccall((:free_native_sim, _LIBLUNA_RUST_RK45), Cvoid, (Ptr{Cvoid},), p)
                h.ptr = C_NULL
            end
        end
        h
    end
end

mutable struct RustNativeStepper{T<:AbstractArray}
    y::T
    yn::T
    t::Float64
    tn::Float64
    dt::Float64
    dtn::Float64
    rtol::Float64
    atol::Float64
    safety::Float64
    max_dt::Float64
    min_dt::Float64
    locextrap::Bool
    ok::Bool
    err::Float64
    errlast::Float64
    _handle::RustNativeSimHandle
end

function RustNativeStepper(linop, y0, t, dt; kwargs...)
    RustNativeStepper(nothing, linop, y0, t, dt; kwargs...)
end

function RustNativeStepper(f!, linop, y0, t, dt;
                            rtol=1e-6, atol=1e-10, safety=0.9, max_dt=Inf, min_dt=0,
                            locextrap=true, norm=weaknorm, flength=Inf)
    n = length(y0)
    is_zdep_mode_avg = linop isa Luna.Capillary.ZDepLinopMarcatili
    is_zdep_free_linop = linop isa Luna.LinearOps.ZDepLinopFree
    is_zdep_modal_taper = linop isa Luna.Capillary.ZDepLinopModalTaper
    if is_zdep_mode_avg || is_zdep_free_linop || is_zdep_modal_taper
        isfinite(flength) || throw(NativeIneligible("z-dependent linop requires a " *
                  "finite `flength` (propagation length) to build the resident z-LUTs " *
                  "— got flength=$flength."))
        handle = RustNativeSimHandle(linop, n)
    else
        handle = RustNativeSimHandle(linop)
    end

    handle.ptr == C_NULL && error("init_native_sim failed")

    is_mode_avg = f! isa Luna.NonlinearRHS.TransModeAvg
    is_radial = f! isa Luna.NonlinearRHS.TransRadial
    is_modal = f! isa Luna.NonlinearRHS.TransModal
    is_free = f! isa Luna.NonlinearRHS.TransFree

    # `f! === nothing` is the deliberate bare-stepper case (Phase 0's own
    # tests construct it directly via the single-arg convenience method
    # below to exercise the resident FFT/RK machinery with no RHS at all).
    # Any *other* unrecognized `f!` (e.g. a raw ad-hoc closure, as RK45.jl's
    # own low-level unit tests use to test the Julia solver plumbing
    # directly) must not silently fall through: none of the
    # `native_set_*_params` calls below would ever run for it, so the
    # native RHS would end up configured with zero nonlinearity — silently
    # wrong physics, not a crash. Reject it up front instead.
    isnothing(f!) || is_mode_avg || is_radial || is_modal || is_free ||
        throw(NativeIneligible("f! is not one of TransModeAvg/TransRadial/" *
              "TransModal/TransFree — the native resident stepper only supports " *
              "Luna's own NonlinearRHS pipeline types."))

    if is_mode_avg || is_radial || is_modal || is_free
        # Gas mixtures (`MarcatiliMode(a, (gas1,gas2,...), (p1,p2,...))`) give
        # `densityfun(z)` a per-species Vector return type and a nested
        # tuple-of-tuples `resp` (one response tuple per component) — neither
        # shape is handled by the native RHS setup below (it expects a
        # scalar density and a flat response tuple), and nothing upstream
        # rejects it. Left unguarded this silently computes a Vector
        # `kerr_fac`/`beta` that only fails much later at the FFI boundary
        # with an opaque `MethodError`. Reject it here instead, with a
        # message that says what's actually unsupported.
        f!.densityfun(0.0) isa Real || throw(NativeIneligible("densityfun(z) returned " *
              "a non-scalar value — gas mixtures are not yet supported by the native path."))

        grid = f!.grid
        is_real_grid = grid isa Luna.Grid.RealGrid   # EnvGrid uses c2c FFTs
        n_time = length(grid.t)
        n_time_over = length(grid.to)
    else
        is_real_grid = true
        n_time = n
        n_time_over = 0
    end

    # Set FFTW plans; is_real=1 for RealGrid (r2c), 0 for EnvGrid (c2c).
    # FFTW_ESTIMATE = 1 << 6 = 64
    lib_path = FFTW.FFTW_jll.libfftw3
    rc = ccall((:native_set_fftw_plans, _LIBLUNA_RUST_RK45), Cint,
          (Ptr{Cvoid}, Cstring, Csize_t, Csize_t, Cint, Cuint),
          handle.ptr, lib_path, n_time, n_time_over, is_real_grid ? 1 : 0, 64)
    rc == 0 || error("native_set_fftw_plans failed")

    # Set parameters if mode-averaged
    if is_mode_avg
        norm_func = f!.norm!
        pre = getfield(norm_func, :pre)

        if hasfield(typeof(norm_func), :βfun!)
            bfun! = getfield(norm_func, :βfun!)
            beta = zeros(Float64, length(pre))
            bfun!(beta, 0.0)
        else
            beta = ones(Float64, length(pre))
        end

        towin = f!.grid.towin
        owin = f!.grid.ωwin
        sidx = UInt8.(f!.grid.sidx)

        # Find Kerr response and extract γ3
        γ3 = 0.0
        for r in f!.resp
            for fld in fieldnames(typeof(r))
                if occursin("γ3", string(fld))
                    γ3 = getfield(r, fld)
                    break
                end
            end
            γ3 != 0.0 && break
        end

        is_zdep_mode_avg || _check_density_zindependent(f!.densityfun, flength)
        density = f!.densityfun(0.0)
        kerr_fac = density * Luna.PhysData.ε_0 * γ3
        nlscale = Luna.NonlinearRHS.nlscale
        sqrt_aeff = sqrt(f!.aeff(0.0))

        rc = ccall((:native_set_mode_avg_params, _LIBLUNA_RUST_RK45), Cint,
            (Ptr{Cvoid}, Csize_t, Csize_t,
             Ptr{Float64}, Ptr{Float64}, Ptr{UInt8},
             Ptr{Float64}, Ptr{Float64}, Ptr{Float64},
             Float64, Float64, Float64),
            handle.ptr, n_time, n_time_over,
            towin, owin, sidx,
            real.(pre), imag.(pre), beta,
            kerr_fac, nlscale, sqrt_aeff)
        rc == 0 || error("native_set_mode_avg_params failed: $rc")

        # Wire plasma if present and a Rust ionization handle is available.
        # Since Phase C (BACKLOG.md), `IonRatePPTAccel` builds this handle
        # whenever the Rust library is present and EITHER
        # `LUNA_USE_RUST_IONISATION=1` OR the native stepper itself is
        # enabled (`LUNA_USE_RUST_NATIVE` defaults to `"1"` since Phase 8) —
        # decoupling it from the opt-in kernel toggle so the fork's default
        # field-resolved workload (`plasma = !envelope`) actually runs
        # natively out of the box (REVIEW.md §3.2). If the handle is still
        # missing here (library absent, or the user explicitly forced both
        # toggles off), the WHOLE construction must fail (NativeIneligible →
        # solve_precon falls back to the Julia stepper) rather than silently
        # continuing without plasma — continuing native-minus-plasma would be
        # wrong physics with only a log line to notice it, which became
        # unacceptable once native became the default (Phase 8) rather than
        # opt-in.
        for r in f!.resp
            if r isa Luna.Nonlinear.PlasmaCumtrapz
                irf = r.ratefunc
                if irf isa Luna.Ionisation.IonRatePPTAccel &&
                        !isnothing(irf.rust_handle) &&
                        irf.rust_handle.ptr != C_NULL
                    rc = ccall((:native_set_plasma_params, _LIBLUNA_RUST_RK45), Cint,
                        (Ptr{Cvoid}, Ptr{Cvoid}, Float64, Float64, Float64, Float64),
                        handle.ptr, irf.rust_handle.ptr,
                        r.ionpot, Luna.PhysData.e_ratio, r.preionfrac, r.δt)
                    rc == 0 || throw(NativeIneligible("native_set_plasma_params returned $rc"))
                else
                    throw(NativeIneligible("plasma detected but no Rust ionisation handle is " *
                          "available (library missing, or both LUNA_USE_RUST_IONISATION and " *
                          "LUNA_USE_RUST_NATIVE were explicitly disabled), or ratefunc is not " *
                          "an IonRatePPTAccel — this config is not eligible for the native path."))
                end
                break  # only one plasma response expected
            end
        end

        # Wire Raman if present and eligible for the resident ADE path. The
        # resident path needs the raw oscillator arrays (not just a handle
        # pointer), so eligibility is re-checked here — same criteria as
        # `Interface._make_rust_raman_handle_from_response` (the existing
        # `LUNA_USE_RUST_RAMAN` FFI wiring): all-SDO `CombinedRamanResponse`
        # with density-independent τ2. Scope: `thg=true` only (E² intensity,
        # no Hilbert transform) — see docs/native-port/MATH.md §5.3. As with
        # plasma above, an ineligible Raman response must fail the whole
        # construction (NativeIneligible), not silently continue without it.
        for r in f!.resp
            if r isa Luna.Nonlinear.RamanPolarField
                if !r.thg
                    throw(NativeIneligible("RamanPolarField has thg=false — native Raman " *
                          "path only supports thg=true (E² intensity)."))
                end
                rr = r.r
                eligible = rr isa Luna.Raman.CombinedRamanResponse &&
                    all(ri isa Luna.Raman.RamanRespSingleDampedOscillator for ri in rr.Rs) &&
                    all(abs(ri.τ2ρ(1.0) - ri.τ2ρ(2.0)) < 1e-10 * abs(ri.τ2ρ(1.0)) for ri in rr.Rs)
                eligible || throw(NativeIneligible("Raman response present but not eligible " *
                      "for the native path (needs all-SDO CombinedRamanResponse with " *
                      "density-independent τ2)."))
                omegas    = Float64[ri.Ω for ri in rr.Rs]
                gammas    = Float64[1.0 / ri.τ2ρ(1.0) for ri in rr.Rs]
                couplings = Float64[ri.K for ri in rr.Rs]
                raman_density = f!.densityfun(0.0)
                rc = ccall((:native_set_raman_params, _LIBLUNA_RUST_RK45), Cint,
                    (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Csize_t, Float64, Float64),
                    handle.ptr, omegas, gammas, couplings, Csize_t(length(omegas)),
                    r.dt, raman_density)
                rc == 0 || throw(NativeIneligible("native_set_raman_params returned $rc"))
                break  # only one Raman response expected
            end
        end

        # Any response type not recognized by the three checks above (Kerr
        # via the γ3 field scan, `PlasmaCumtrapz`, `RamanPolarField`) must
        # fail the whole construction too — e.g. `RamanPolarEnv` (envelope
        # Raman, the response `Interface.makeresponse` attaches for
        # EnvGrid/GNLSE configs) matches none of those `isa` checks, so
        # without this it would silently slip through with no wiring and no
        # error: exactly the silent-wrong-physics gap this file's
        # `NativeIneligible` conversion (Phase 8) exists to close. Found via
        # `test_gnlse.jl`'s "Soliton shift" test once native became the
        # default — that test's self-frequency-shift numbers depend
        # entirely on Raman, so silently dropping it produced a completely
        # different (not just numerically close) result.
        is_kerr_resp(r) = any(occursin("γ3", string(fld)) for fld in fieldnames(typeof(r)))
        for r in f!.resp
            is_kerr_resp(r) || r isa Luna.Nonlinear.PlasmaCumtrapz ||
                r isa Luna.Nonlinear.RamanPolarField ||
                throw(NativeIneligible("response type $(typeof(r)) is not wired for " *
                      "the native mode-averaged path."))
        end
    end

    # Set parameters for the z-dependent linop (Phase 7) — mode-averaged,
    # graded-core constant-radius MarcatiliMode only. See MATH.md §3.5 and
    # docs/native-port/BETA1_ANALYTIC.md. dens(pressure) is TRANSFERRED
    # exactly (PhysData.densityspline's own (x,y,D)), not re-fit — re-fitting
    # a fresh spline through sampled dens values doesn't converge (real-gas
    # density via CoolProp, composed with dspl already being a spline
    # itself, isn't smooth enough at the scale a refit needs). β1(z) is a
    # closed form in dens(z) (4 constants precomputed in
    # `Capillary.make_linop`'s ZDepLinopMarcatili specialization) — not a
    # LUT: an earlier version of this design LUT'd β1 against density, which
    # could only ever chase Julia's own adaptive-FD noise floor
    # (`Modes.dispersion`) rather than the true derivative.
    if is_zdep_mode_avg
        w = linop
        w.L == flength || error("RustNativeStepper: z-dependent linop's gradient length " *
                  "($(w.L)) does not match the propagation length flength=$flength passed " *
                  "to solve_precon — these should always be the same quantity " *
                  "(Capillary.gradient's `L` vs `grid.zmax`/`tmax`). Refusing to guess " *
                  "which is correct.")
        model_code = w.model == :full ? Cuint(0) : Cuint(1)
        loss_code = w.loss ? Cuint(1) : Cuint(0)
        # density(z)·ε₀·γ3 = kerr_fac(z): γ3 must be re-scaled by the
        # z-dependent density every RK stage (TransModeAvg calls
        # `densityfun(z)` fresh, unlike the constant-medium phases which
        # bake `density(0)` into `kerr_fac` once) — see `ensure_linop_at`.
        eps0_gamma3 = Luna.PhysData.ε_0 * γ3

        rc = ccall((:native_set_zdep_mode_avg_params, _LIBLUNA_RUST_RK45), Cint,
            (Ptr{Cvoid}, Float64, Float64, Float64,
             Csize_t, Ptr{Float64}, Ptr{Float64}, Ptr{Float64},
             Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64},
             Cuint, Cuint, Float64,
             Float64, Float64, Float64, Float64, Float64, Float64, Float64),
            handle.ptr, w.L, w.p0, w.p1,
            Csize_t(length(w.dspl_x)), w.dspl_x, w.dspl_y, w.dspl_d,
            w.gamma, w.nwg_re, w.nwg_im, w.omega,
            model_code, loss_code, eps0_gamma3,
            w.ω0, w.γ0, w.dγ0, w.nwg0_re, w.nwg0_im, w.dnwg0_re, w.dnwg0_im)
        rc == 0 || error("native_set_zdep_mode_avg_params failed: $rc")
    end

    # Set parameters if radial (TransRadial) — Phase 3 gate: scalar Kerr only,
    # RealGrid or EnvGrid (EnvGrid added Phase D.1, BACKLOG.md — `rhs_radial_env`
    # in native.rs, dispatched on `sim.is_real` exactly like mode-averaged's
    # real/env split). Phase D.2 (BACKLOG.md) adds plasma alongside Kerr
    # (`apply_plasma_radial` in native.rs, RealGrid only — Julia has no
    # EnvGrid plasma either). See docs/native-port/MATH.md §3.2 for the
    # design (precomputed normalization array M, resident QdhtFfiHandle reused
    # directly); `M`'s length (`n_spec`) and the FFI's internal buffer sizing
    # both already generalize to either grid type without further changes here.
    if is_radial
        HT = f!.QDHT
        n_r = HT.N
        n % n_r == 0 || error("RustNativeStepper: field length $n not divisible by QDHT.N=$n_r")

        T_mat = Matrix{Float64}(HT.T)
        size(T_mat) == (n_r, n_r) || error("RustNativeStepper: QDHT.T shape mismatch")
        scale_fwd = Float64(HT.scaleRK)
        scale_inv = 1.0 / scale_fwd

        towin = f!.grid.towin

        # Find Kerr response and extract γ3 (same pattern as mode-averaged above).
        γ3 = 0.0
        for r in f!.resp
            for fld in fieldnames(typeof(r))
                if occursin("γ3", string(fld))
                    γ3 = getfield(r, fld)
                    break
                end
            end
            γ3 != 0.0 && break
        end
        # Phase D.2 allows Kerr + a plasma response; Phase D.4 (BACKLOG.md)
        # adds Raman (`RamanPolarField`, RealGrid `thg=true` only — same
        # eligibility criteria as the mode-averaged Raman wiring below: an
        # all-SDO `CombinedRamanResponse` with density-independent τ2). Any
        # other response type (e.g. `RamanPolarEnv`, EnvGrid's envelope
        # Raman) remains out of scope.
        is_kerr_resp_radial(r) = any(occursin("γ3", string(fld)) for fld in fieldnames(typeof(r)))
        for r in f!.resp
            is_kerr_resp_radial(r) || r isa Luna.Nonlinear.PlasmaCumtrapz ||
                r isa Luna.Nonlinear.RamanPolarField ||
                throw(NativeIneligible("radial: only Kerr, plasma, and/or Raman " *
                      "(RealGrid, thg=true) responses are supported by the native path " *
                      "(Phase D.4 gate)."))
        end
        γ3 != 0.0 ||
            throw(NativeIneligible("radial: no Kerr response found (γ3=0) — the native " *
                  "radial path requires a Kerr response."))

        _check_density_zindependent(f!.densityfun, flength)
        density = f!.densityfun(0.0)
        kerr_fac = density * Luna.PhysData.ε_0 * γ3

        # Precompute M[iω,ir] = ωwin[iω]·(-i·ω[iω]) / (2·normfun(0.0)[iω,ir]).
        # Folds norm_radial + ωwin into one array. Valid only for a z-invariant
        # normfun (const_norm_radial, constant-medium gas cell) — see MATH.md §3.2.
        normarr = f!.normfun(0.0)
        ω = f!.grid.ω
        ωwin = f!.grid.ωwin
        M = (ωwin .* (-im .* ω)) ./ (2 .* normarr)

        rc = ccall((:native_set_radial_params, _LIBLUNA_RUST_RK45), Cint,
            (Ptr{Cvoid}, Csize_t, Csize_t, Csize_t,
             Ptr{Float64}, Float64, Float64,
             Ptr{Float64}, Float64,
             Ptr{Float64}, Ptr{Float64}),
            handle.ptr, n_time, n_time_over, n_r,
            T_mat, scale_fwd, scale_inv,
            towin, kerr_fac,
            real.(M), imag.(M))
        rc == 0 || error("native_set_radial_params failed: $rc")

        # Wire plasma if present and a Rust ionisation handle is available —
        # same decoupled-from-LUNA_USE_RUST_IONISATION eligibility as the
        # mode-averaged wiring above (Phase C, BACKLOG.md). Must fail the
        # whole construction (NativeIneligible), not silently continue
        # without plasma, for the same silent-wrong-physics reason.
        for r in f!.resp
            if r isa Luna.Nonlinear.PlasmaCumtrapz
                irf = r.ratefunc
                if irf isa Luna.Ionisation.IonRatePPTAccel &&
                        !isnothing(irf.rust_handle) &&
                        irf.rust_handle.ptr != C_NULL
                    rc = ccall((:native_set_plasma_params, _LIBLUNA_RUST_RK45), Cint,
                        (Ptr{Cvoid}, Ptr{Cvoid}, Float64, Float64, Float64, Float64),
                        handle.ptr, irf.rust_handle.ptr,
                        r.ionpot, Luna.PhysData.e_ratio, r.preionfrac, r.δt)
                    rc == 0 || throw(NativeIneligible("native_set_plasma_params returned $rc"))
                else
                    throw(NativeIneligible("plasma detected but no Rust ionisation handle is " *
                          "available (library missing, or both LUNA_USE_RUST_IONISATION and " *
                          "LUNA_USE_RUST_NATIVE were explicitly disabled), or ratefunc is not " *
                          "an IonRatePPTAccel — this config is not eligible for the native path."))
                end
                break  # only one plasma response expected
            end
        end

        # Wire Raman if present and eligible — Phase D.4 (BACKLOG.md). Same
        # eligibility criteria and FFI call as the mode-averaged wiring
        # above; `native_set_raman_params` sizes its scratch buffers as
        # `n_time_over*n_r` when `s.is_radial` (already set by
        # `native_set_radial_params`, called earlier in this block).
        for r in f!.resp
            if r isa Luna.Nonlinear.RamanPolarField
                if !r.thg
                    throw(NativeIneligible("RamanPolarField has thg=false — native Raman " *
                          "path only supports thg=true (E² intensity)."))
                end
                rr = r.r
                eligible = rr isa Luna.Raman.CombinedRamanResponse &&
                    all(ri isa Luna.Raman.RamanRespSingleDampedOscillator for ri in rr.Rs) &&
                    all(abs(ri.τ2ρ(1.0) - ri.τ2ρ(2.0)) < 1e-10 * abs(ri.τ2ρ(1.0)) for ri in rr.Rs)
                eligible || throw(NativeIneligible("Raman response present but not eligible " *
                      "for the native path (needs all-SDO CombinedRamanResponse with " *
                      "density-independent τ2)."))
                omegas    = Float64[ri.Ω for ri in rr.Rs]
                gammas    = Float64[1.0 / ri.τ2ρ(1.0) for ri in rr.Rs]
                couplings = Float64[ri.K for ri in rr.Rs]
                raman_density = f!.densityfun(0.0)
                rc = ccall((:native_set_raman_params, _LIBLUNA_RUST_RK45), Cint,
                    (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Csize_t, Float64, Float64),
                    handle.ptr, omegas, gammas, couplings, Csize_t(length(omegas)),
                    r.dt, raman_density)
                rc == 0 || throw(NativeIneligible("native_set_raman_params returned $rc"))
                break  # only one Raman response expected
            end
        end
    end

    # Set parameters if modal (TransModal) — Phase 5 gate: RealGrid,
    # `full=false` only (the radial modal integral — this is exactly what
    # `Interface.needfull` selects for `HE, n=1` mode collections, i.e. the
    # common case, not an artificial restriction), constant-radius Marcatili
    # `kind=:HE, n=1` modes only, Kerr-only. See MATH.md §3.3 for the design
    # (libcubature reuse, closed-form N(m), the numerically-probed norm!).
    if is_modal
        is_real_grid || throw(NativeIneligible("EnvGrid modal not yet supported " *
                               "(Phase 5 gate is RealGrid-only)"))
        f!.full && throw(NativeIneligible("full=true (2-D modal integral) not yet " *
                          "supported (Phase 5 gate is full=false)"))
        isnothing(f!.Emω_noise) || throw(NativeIneligible("modified shot-noise " *
                          "(Emω_noise) not yet supported for the native modal path"))

        modes = f!.ts.ms
        n_modes = length(modes)
        npol = f!.ts.npol
        npol == 1 || throw(NativeIneligible("native modal path only supports " *
                  "npol=1 (single polarisation component). The KerrVector! " *
                  "(npol=2, circular/elliptical polarisation) code path is " *
                  "implemented in native.rs but not yet verified against the " *
                  "Julia oracle (Phase 5 gate — see MATH.md §3.3)."))

        all(m -> m isa Luna.Capillary.MarcatiliMode, modes) ||
            throw(NativeIneligible("native modal path only supports MarcatiliMode " *
                  "(Phase 5 gate)"))
        all(m -> m.kind in (:HE, :TE, :TM), modes) ||
            throw(NativeIneligible("native modal path only supports kind ∈ " *
                  "(:HE, :TE, :TM) Marcatili modes"))
        # Phase E.2: tapered radius, via the dedicated `ZDepLinopModalTaper`
        # wrapper (`Capillary.make_linop`'s specialized method) — any other
        # Function-valued `a` (not wrapped, e.g. per-mode differing tapers)
        # remains out of scope.
        is_zdep_modal_taper = linop isa Luna.Capillary.ZDepLinopModalTaper
        all(m -> m.a isa Number, modes) || is_zdep_modal_taper ||
            throw(NativeIneligible("native modal path only supports constant-radius " *
                  "modes, or a shared tapered radius via Capillary.make_linop's " *
                  "ZDepLinopModalTaper wrapper (Phase E.2)"))

        if is_zdep_modal_taper
            isfinite(flength) || throw(NativeIneligible("z-dependent modal linop " *
                      "requires a finite `flength` to build the resident z-LUT — " *
                      "got flength=$flength."))
            a = Float64(modes[1].a(0.0))
        else
            a = Float64(modes[1].a)
            all(m -> Float64(m.a) == a, modes) ||
                throw(NativeIneligible("all modes must share the same core radius"))
        end

        unm = Float64[m.unm for m in modes]
        invsqrtn = Float64[1.0 / sqrt(Luna.Modes.N(m, z=0.0)) for m in modes]
        # `field(m, (r,0))` (Capillary.jl) factored as `J_order(x)·(angle_x,
        # angle_y)` — Phase E.1 generalizes the Phase 5 `HE,n=1`-only
        # `J0·(sinϕ,cosϕ)` special case (recovered here at n=1) to any
        # `:HE`/`:TE`/`:TM` mode order.
        order = Int32[m.kind == :HE ? m.n - 1 : 1 for m in modes]
        angle_x = Float64[m.kind == :HE ? sin(m.n * m.ϕ) : (m.kind == :TM ? 1.0 : 0.0) for m in modes]
        angle_y = Float64[m.kind == :HE ? cos(m.n * m.ϕ) : (m.kind == :TM ? 0.0 : 1.0) for m in modes]

        pol_select = npol == 2 ? UInt8[0, 1] : UInt8[f!.ts.indices == 1 ? 0 : 1]

        towin = f!.grid.towin

        # Find Kerr response and extract γ3 (same pattern as mode-averaged/radial above).
        γ3 = 0.0
        for r in f!.resp
            for fld in fieldnames(typeof(r))
                if occursin("γ3", string(fld))
                    γ3 = getfield(r, fld)
                    break
                end
            end
            γ3 != 0.0 && break
        end
        # Phase 5 gate is Kerr-only; Phase D.4 (BACKLOG.md) adds Raman
        # (RealGrid, thg=true, npol=1 — same per-node scalar-field scope as
        # Kerr here) alongside it. Any other response type remains ineligible.
        is_kerr_resp_modal(r) = any(occursin("γ3", string(fld)) for fld in fieldnames(typeof(r)))
        for r in f!.resp
            is_kerr_resp_modal(r) || r isa Luna.Nonlinear.RamanPolarField ||
                throw(NativeIneligible("modal: only Kerr and/or Raman (RealGrid, " *
                      "thg=true) responses are supported by the native path " *
                      "(Phase D.4 gate)."))
        end
        γ3 != 0.0 ||
            throw(NativeIneligible("modal: no Kerr response found (γ3=0) — the " *
                  "native modal path requires a Kerr response."))

        _check_density_zindependent(f!.densityfun, flength)
        density = f!.densityfun(0.0)
        kerr_fac = density * Luna.PhysData.ε_0 * γ3

        # Extract exactly what `ωwin .* norm!` computes by numerically probing
        # the closure (robust to norm_modal's shock/no-shock branch — avoids
        # re-deriving grid.ω/shock logic, see MATH.md §3.3).
        nlfac = ComplexF64.(f!.grid.ωwin)
        f!.norm!(nlfac)

        lib_path = Cubature.Cubature_jll.libcubature

        rc = ccall((:native_set_modal_params, _LIBLUNA_RUST_RK45), Cint,
            (Ptr{Cvoid}, Csize_t, Csize_t, Csize_t, Csize_t,
             Float64, Ptr{Float64}, Ptr{Float64}, Ptr{Int32}, Ptr{Float64}, Ptr{Float64},
             Ptr{UInt8}, Ptr{Float64}, Float64, Ptr{Float64}, Ptr{Float64},
             Cstring, Float64, Float64, Csize_t),
            handle.ptr, n_time, n_time_over, n_modes, npol,
            a, unm, invsqrtn, order, angle_x, angle_y, pol_select,
            towin, kerr_fac, real.(nlfac), imag.(nlfac),
            lib_path, f!.rtol, f!.atol, Csize_t(f!.mfcn))
        rc == 0 || error("native_set_modal_params failed: $rc")

        # Wire the z-dependent linop (tapered radius) — Phase E.2. Must come
        # after `native_set_modal_params` (reads back its `modal_inv_sqrt_n`
        # z=0 baseline for the `1/√N(m,z) ∝ 1/a(z)` rescale).
        if is_zdep_modal_taper
            w = linop
            n_a = length(w.a_x)
            rc = ccall((:native_set_modal_zdep_params, _LIBLUNA_RUST_RK45), Cint,
                (Ptr{Cvoid}, Float64, Float64,
                 Csize_t, Ptr{Float64}, Ptr{Float64}, Ptr{Float64},
                 Ptr{Float64}, Ptr{UInt8}, UInt8, UInt8,
                 Ptr{Float64}, Ptr{Float64}, Ptr{Float64},
                 Float64, Csize_t,
                 Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}),
                handle.ptr, w.flength, a,
                Csize_t(n_a), w.a_x, w.a_y, w.a_d,
                w.omega, UInt8.(w.sidx), w.model == :full ? UInt8(0) : UInt8(1), UInt8(w.loss),
                vec(w.eco), vec(w.vn_re), vec(w.vn_im),
                w.ω0, Csize_t(w.ref_mode),
                w.eco0, w.deco0, w.v0_re, w.v0_im, w.dv0_re, w.dv0_im)
            rc == 0 || throw(NativeIneligible("native_set_modal_zdep_params returned $rc"))
        end

        # Wire Raman if present and eligible — Phase D.4 (BACKLOG.md). Same
        # eligibility criteria and FFI call as mode-averaged/radial above;
        # `native_set_raman_params` sizes its scratch buffers as
        # `n_time_over` (the `else` branch — modal is neither mode-averaged
        # nor radial, and `rhs_modal_pointcalc` reuses one shared buffer
        # sequentially across quadrature nodes, exactly like mode-averaged's
        # single per-call buffer — see native.rs's `apply_raman_radial` doc
        # for why the same solver instance is safe to reuse this way).
        for r in f!.resp
            if r isa Luna.Nonlinear.RamanPolarField
                if !r.thg
                    throw(NativeIneligible("RamanPolarField has thg=false — native Raman " *
                          "path only supports thg=true (E² intensity)."))
                end
                rr = r.r
                eligible = rr isa Luna.Raman.CombinedRamanResponse &&
                    all(ri isa Luna.Raman.RamanRespSingleDampedOscillator for ri in rr.Rs) &&
                    all(abs(ri.τ2ρ(1.0) - ri.τ2ρ(2.0)) < 1e-10 * abs(ri.τ2ρ(1.0)) for ri in rr.Rs)
                eligible || throw(NativeIneligible("Raman response present but not eligible " *
                      "for the native path (needs all-SDO CombinedRamanResponse with " *
                      "density-independent τ2)."))
                omegas    = Float64[ri.Ω for ri in rr.Rs]
                gammas    = Float64[1.0 / ri.τ2ρ(1.0) for ri in rr.Rs]
                couplings = Float64[ri.K for ri in rr.Rs]
                raman_density = f!.densityfun(0.0)
                rc = ccall((:native_set_raman_params, _LIBLUNA_RUST_RK45), Cint,
                    (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Csize_t, Float64, Float64),
                    handle.ptr, omegas, gammas, couplings, Csize_t(length(omegas)),
                    r.dt, raman_density)
                rc == 0 || throw(NativeIneligible("native_set_raman_params returned $rc"))
                break  # only one Raman response expected
            end
        end
    end

    # Set parameters if free-space (TransFree) — Phase 6 gate: RealGrid,
    # const_norm_free (z-invariant), scalar Kerr only. Phase D.3 (BACKLOG.md)
    # adds EnvGrid (c2c 3-D FFTW plan, `ComplexFft3d`) — `native_set_free_params`
    # branches on `sim.is_real` (set by `native_set_fftw_plans`, called earlier)
    # exactly like the radial Phase D.1 EnvGrid extension. Reuses the FFTW
    # library handle native_set_fftw_plans already loaded (no new dlopen) —
    # only a new 3-D plan is created. Phase D.5 (BACKLOG.md) drops the
    # z-invariant restriction for a two-point pressure-gradient gas cell —
    # `normfun isa NonlinearRHS.ZDepNormFree` (and `linop isa
    # LinearOps.ZDepLinopFree`, checked by `native_ok` above) — wiring an
    # additional `native_set_free_zdep_params` call. See MATH.md §3.4.
    if is_free
        isnothing(f!.Et_noise) || throw(NativeIneligible("modified shot-noise " *
                          "(Et_noise) not yet supported for the native free-space path"))

        n_y = length(f!.xygrid.y)
        n_x = length(f!.xygrid.x)

        towin = f!.grid.towin

        γ3 = 0.0
        for r in f!.resp
            for fld in fieldnames(typeof(r))
                if occursin("γ3", string(fld))
                    γ3 = getfield(r, fld)
                    break
                end
            end
            γ3 != 0.0 && break
        end
        length(f!.resp) == 1 && γ3 != 0.0 ||
            throw(NativeIneligible("free-space: only single-response Kerr-only " *
                  "is supported by the native path (Phase 6 gate); extra responses " *
                  "(plasma, Raman, ...), or a lone non-Kerr response, are not wired " *
                  "for this geometry."))

        is_zdep_free = f!.normfun isa Luna.NonlinearRHS.ZDepNormFree
        if is_zdep_free
            linop isa Luna.LinearOps.ZDepLinopFree ||
                throw(NativeIneligible("free-space: a z-dependent normfun (ZDepNormFree) " *
                      "requires a matching z-dependent linop (ZDepLinopFree) — got " *
                      "$(typeof(linop))."))
        else
            _check_density_zindependent(f!.densityfun, flength)
        end
        density = f!.densityfun(0.0)
        kerr_fac = density * Luna.PhysData.ε_0 * γ3

        # Precompute M[iω,iky,ikx] = ωwin[iω]·(-i·ω[iω]) / (2·normfun(0.0)[iω,iky,ikx]).
        # Folds norm_free + ωwin into one array. For the z-dependent case this
        # is just the z=0 seed value — `native_set_free_zdep_params` below
        # (via `ensure_free_norm_at`) overwrites it before first use, since
        # `zdep_free_last_z` starts at NaN.
        normarr = f!.normfun(0.0)
        ω = f!.grid.ω
        ωwin = f!.grid.ωwin
        M = (ωwin .* (-im .* ω)) ./ (2 .* normarr)

        # FFTW_ESTIMATE = 1 << 6 = 64 — must match native_set_fftw_plans' flag.
        rc = ccall((:native_set_free_params, _LIBLUNA_RUST_RK45), Cint,
            (Ptr{Cvoid}, Csize_t, Csize_t, Csize_t, Csize_t, Cuint,
             Ptr{Float64}, Float64,
             Ptr{Float64}, Ptr{Float64}),
            handle.ptr, n_time, n_time_over, n_y, n_x, Cuint(64),
            towin, kerr_fac,
            real.(M), imag.(M))
        rc == 0 || error("native_set_free_params failed: $rc")

        if is_zdep_free
            w = linop
            eps0_gamma3 = Luna.PhysData.ε_0 * γ3
            rc = ccall((:native_set_free_zdep_params, _LIBLUNA_RUST_RK45), Cint,
                (Ptr{Cvoid}, Float64, Float64, Float64,
                 Csize_t, Ptr{Float64}, Ptr{Float64}, Ptr{Float64},
                 Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{UInt8},
                 Float64, Float64, Float64, Float64),
                handle.ptr, w.L, w.p0, w.p1,
                Csize_t(length(w.dspl_x)), w.dspl_x, w.dspl_y, w.dspl_d,
                w.gamma, w.omega, ωwin, w.kperp2, UInt8.(w.sidx),
                eps0_gamma3, w.ω0, w.γ0, w.dγ0)
            rc == 0 || throw(NativeIneligible("native_set_free_zdep_params returned $rc"))
        end
    end

    # Copy initial field to Rust
    ccall((:set_field, _LIBLUNA_RUST_RK45), Cint,
          (Ptr{Cvoid}, Ptr{ComplexF64}, Csize_t),
          handle.ptr, pointer(y0), Csize_t(n))

    RustNativeStepper(
        copy(y0), copy(y0), float(t), float(t), float(dt), float(dt),
        float(rtol), float(atol), float(safety), float(max_dt), float(min_dt),
        locextrap, false, 0.0, 0.0, handle
    )
end

function step!(s::RustNativeStepper)
    # native_step overwrites s.yn in place; snapshot the pre-call value (the
    # field at the start of this interval, s.t) so interpolate() has a valid
    # start-of-interval field once the step is accepted.
    y_before = copy(s.yn)

    result_ref = Ref(NativeStepResult(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    rc = ccall((:native_step, _LIBLUNA_RUST_RK45), Cint,
        (Ptr{Cvoid}, Ptr{ComplexF64}, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Cint, Ptr{NativeStepResult}),
        s._handle.ptr, pointer(s.yn), s.t, s.tn, s.dtn, s.rtol, s.atol, s.safety, s.max_dt, s.min_dt, s.errlast, Cint(s.locextrap), result_ref
    )
    rc == 0 || error("native_step returned error code $rc")

    res = result_ref[]
    s.ok = res.ok != 0
    s.dt = res.dt
    s.t = res.t
    s.tn = res.tn
    s.dtn = res.dtn
    s.err = res.err
    s.errlast = res.errlast

    if s.ok
        s.y .= y_before
    end

    return s.ok
end

function _native_field_resync!(s::RustNativeStepper)
    rc = ccall((:native_resync_field, _LIBLUNA_RUST_RK45), Cint,
          (Ptr{Cvoid}, Ptr{ComplexF64}, Csize_t),
          s._handle.ptr, pointer(s.yn), Csize_t(length(s.yn)))
    rc == 0 || error("native_resync_field (post-stepfun resync) failed: $rc")
end

function interpolate(s::RustNativeStepper, ti::Float64)
    if ti > s.tn
        error("Attempting to extrapolate!")
    end
    if ti == s.t
        return s.y
    elseif ti == s.tn
        return s.yn
    end
    # Same quartic dense output as PreconStepper/RustPreconStepper (interpC,
    # all 7 RK stages), not linear — the propagation itself already matches
    # Julia to ~1e-15, but the earlier linear-only interpolation between
    # accepted steps was measurably lower-order than Julia's, which showed
    # up as a ~1-2% error in any densely-sampled output (MemoryOutput,
    # i.e. essentially every general-purpose test) despite the underlying
    # solve being correct (see PORT_LOG Phase 8). The 7 stages live in
    # Rust's resident `ks` buffer (never transferred per-step, unlike
    # PreconStepper where Julia owns them already) — fetched here via
    # `get_ks_stage`, one call per stage, only on the (rare, relative to
    # step!) dense-output path.
    n = length(s.yn)
    σ = (ti - s.t) / s.dt
    σ2 = σ^2; σ3 = σ2*σ; σ4 = σ3*σ
    b = ntuple(ii -> σ*interpC[1,ii] + σ2*interpC[2,ii] + σ3*interpC[3,ii] + σ4*interpC[4,ii], Val(7))
    # `similar`/`zero`, not `zeros(ComplexF64, n)` — `RustNativeStepper{T}` is
    # generic over `T<:AbstractArray`, and modal/multi-mode geometries use a
    # `Matrix{ComplexF64}` field (n_ω x n_modes), not a flat vector; a flat
    # `Vector` buffer here would silently broadcast-mismatch against `s.y`.
    yi = zero(s.yn)
    kbuf = similar(s.yn)
    for ii = 1:7
        rc = ccall((:get_ks_stage, _LIBLUNA_RUST_RK45), Cint,
              (Ptr{Cvoid}, Csize_t, Ptr{ComplexF64}, Csize_t),
              s._handle.ptr, Csize_t(ii - 1), kbuf, Csize_t(n))
        rc == 0 || error("get_ks_stage failed: $rc")
        yi .+= kbuf .* b[ii]
    end
    out = @. s.y + s.dt * yi
    rc = ccall((:native_apply_prop, _LIBLUNA_RUST_RK45), Cint,
          (Ptr{Cvoid}, Ptr{ComplexF64}, Csize_t, Float64, Float64),
          s._handle.ptr, out, Csize_t(n), s.t, ti)
    rc == 0 || error("native_apply_prop failed: $rc")
    return out
end

end