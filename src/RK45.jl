module RK45
import Dates
import Logging
import Printf: @sprintf
import Luna.Utils: format_elapsed, lunadir
import FFTW
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

function solve_precon(f!, linop, y0, t, dt, tmax;
                    rtol=1e-6, atol=1e-10, safety=0.9, max_dt=Inf, min_dt=0, locextrap=true, norm=weaknorm,
                    kwargs...)
    use_rust = get(ENV, "LUNA_USE_RUST_STEPPER", "0") == "1"
    use_native = get(ENV, "LUNA_USE_RUST_NATIVE", "0") == "1"
    if use_native && isfile(_LIBLUNA_RUST_RK45) && eltype(y0) == ComplexF64 && linop isa Array{ComplexF64}
        stepper = RustNativeStepper(f!, linop, y0, t, dt,
                      rtol=rtol, atol=atol, safety=safety, max_dt=max_dt, min_dt=min_dt, locextrap=locextrap, norm=norm)
    elseif use_rust && isfile(_LIBLUNA_RUST_RK45) && eltype(y0) == ComplexF64
        stepper = RustPreconStepper(f!, linop, y0, t, dt,
                      rtol=rtol, atol=atol, safety=safety, max_dt=max_dt, min_dt=min_dt, locextrap=locextrap, norm=norm)
    else
        stepper = PreconStepper(f!, linop, y0, t, dt,
                      rtol=rtol, atol=atol, safety=safety, max_dt=max_dt, min_dt=min_dt, locextrap=locextrap, norm=norm)
    end
    return solve(stepper, tmax; kwargs...)
end

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
                            locextrap=true, norm=weaknorm)
    handle = RustNativeSimHandle(linop)

    handle.ptr == C_NULL && error("init_native_sim failed")
    
    n = length(y0)
    is_mode_avg = f! isa Luna.NonlinearRHS.TransModeAvg
    is_radial = f! isa Luna.NonlinearRHS.TransRadial

    if is_mode_avg || is_radial
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
        # Requires LUNA_USE_RUST_IONISATION=1; if not set, plasma is silently
        # skipped by the native path (wrong physics — warn the user).
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
                    rc == 0 || @warn "native_set_plasma_params returned $rc; plasma uses Julia fallback"
                else
                    @warn "LUNA_USE_RUST_NATIVE: plasma detected but LUNA_USE_RUST_IONISATION not set " *
                          "or not IonRatePPTAccel — native path will skip plasma (incorrect physics). " *
                          "Set LUNA_USE_RUST_IONISATION=1 to enable native plasma."
                end
                break  # only one plasma response expected
            end
        end
    end

    # Set parameters if radial (TransRadial) — Phase 3 gate: RealGrid + scalar
    # Kerr only. See docs/native-port/MATH.md §3.2 for the design (precomputed
    # normalization array M, resident QdhtFfiHandle reused directly).
    if is_radial
        is_real_grid || error("RustNativeStepper: EnvGrid radial not yet supported " *
                               "(Phase 3 gate is RealGrid-only)")

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
        length(f!.resp) == 1 ||
            @warn "RustNativeStepper (radial): only single-response (Kerr-only) is " *
                  "verified by the native path; extra responses (plasma, Raman, ...) " *
                  "are ignored — physics will be wrong if any are present."

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

function interpolate(s::RustNativeStepper, ti::Float64)
    if ti > s.tn
        error("Attempting to extrapolate!")
    end
    if ti == s.t
        return s.y
    elseif ti == s.tn
        return s.yn
    end
    # Linear interpolation in the interaction picture (both y and yn are in IP).
    # Full DOPRI5 dense output would require exporting k-stages from Rust via FFI.
    σ = (ti - s.t) / (s.tn - s.t)
    return @. s.y + σ * (s.yn - s.y)
end

end