module RK45
import Dates
import Logging
import Printf: @sprintf
import Amalthea.Utils: format_elapsed, lunadir
import FFTW
import Cubature
import ..Amalthea


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
`test/test_native_default_workload.jl` (docs/dev/BACKLOG.md Phase C.2).
"""
const _LAST_STEPPER_TYPE = Ref{Any}(nothing)

"""
docs/dev/BACKLOG.md S1 item 1: on-disk FFTW planner-wisdom persistence for the native
resident stepper is opt-in, default OFF. Default-off avoids two problems: (a)
paying disk I/O + FFTW's planner-lock on every `RustNativeStepper`
construction (relevant for parameter scans that build many short-lived
steppers), and (b) importing wisdom before planning make plan selection —
and therefore the adaptive step-size path, via the RK45 controller's
near-cancellation sensitivity (see the Phase 2 gotcha in CLAUDE.md) — vary
with whatever the on-disk file happens to contain from a prior run. Set
`AMALTHEA_NATIVE_FFTW_WISDOM=1` to restore persistence (import + export) for
workloads where the accumulated wisdom is worth the tradeoff. See
`docs/dev/native-port/PLANS.md §1` for the full analysis.
"""
_native_wisdom_enabled() = Amalthea.Config.backend_config().native_wisdom

function solve_precon(f!, linop, y0, t, dt, tmax;
                    rtol=1e-6, atol=1e-10, safety=0.9, max_dt=Inf, min_dt=0, locextrap=true, norm=weaknorm,
                    kwargs...)
    cfg = Amalthea.Config.backend_config()
    use_rust = cfg.stepper
    # Phase 8: native is the default (AMALTHEA_USE_RUST_NATIVE=0 opts back out).
    use_native = cfg.native
    # Constant linop (Phases 1-6) or the narrow z-dependent case Phase 7 supports
    # (graded-core, constant-radius MarcatiliMode — see Capillary.jl / MATH.md
    # §3.5). Any other z-dependent linop (a bare `Function`, e.g. multimode or
    # radial/free-space/modal `nfun(ω;z)`) is NOT recognized here, so it falls
    # through to the non-native paths below exactly as before Phase 7 — no
    # behavior change for configurations outside the new narrow scope.
    native_ok = linop isa Array{ComplexF64} || linop isa Amalthea.Capillary.ZDepLinopMarcatili ||
                linop isa Amalthea.LinearOps.ZDepLinopFree ||
                linop isa Amalthea.Capillary.ZDepLinopModalTaper
    stepper = nothing
    if use_native && isfile(_LIBAMALTHEA_RK45) && eltype(y0) == ComplexF64 && native_ok
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
                @warn "AMALTHEA_USE_RUST_NATIVE: falling back to the Julia stepper for a " *
                      "config outside the native port's current scope ($(e.msg)). This " *
                      "warning prints once per session; every other config still uses " *
                      "the resident Rust stepper. Set AMALTHEA_USE_RUST_NATIVE=0 to disable " *
                      "the native path entirely."
                _NATIVE_FALLBACK_WARNED[] = true
            end
        end
    end
    if isnothing(stepper) && use_rust && isfile(_LIBAMALTHEA_RK45) && eltype(y0) == ComplexF64
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
`Amalthea.run`-driven simulation silently drops the windowing for the native
path entirely — invisible in every native-specific phase test (they all call
`solve()`/`step!()` directly, bypassing `stepfun`), but present in every real
simulation via `Interface.jl`/`Amalthea.run`. Fixed by pushing the
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

"""
    _is_plain_kerr_resp(r)

True iff `r` is `Nonlinear.Kerr_field(γ3)` or `Nonlinear.Kerr_env(γ3)` — the
two Kerr closures native.rs actually implements (`KerrScalar!`/`KerrVector!`
for RealGrid, `KerrScalarEnv!`/`KerrVectorEnv!` for EnvGrid, dispatched
purely on `sim.is_real`). A bare "does some field contain γ3" scan is not
enough: `Kerr_field_nothg(γ3, n)` and `Kerr_env_thg(γ3, ω0, t)` also capture
a field named γ3, but flip the THG treatment relative to what native.rs
always applies for a given grid type (native.rs cannot tell them apart —
there is no THG flag in the FFI). Passing either through the naive scan
used to construct a `RustNativeStepper` that silently ran with the WRONG
THG treatment instead of falling back to Julia. Distinguish them by field
count: `Kerr_field`/`Kerr_env` capture only `γ3`; the `_nothg`/`_thg`
variants also capture a Hilbert-transform plan or oscillating-phase buffer
(one extra field).
"""
function _is_plain_kerr_resp(r)
    flds = fieldnames(typeof(r))
    length(flds) == 1 && occursin("γ3", string(flds[1]))
end

# ── Rust interaction-picture PreconStepper FFI (AMALTHEA_USE_RUST_STEPPER=1) ───────

function _libamalthea_path_rk45()
    libname = if Sys.iswindows(); "amalthea.dll"
              elseif Sys.isapple(); "libamalthea.dylib"
              else; "libamalthea.so"; end
    joinpath(lunadir(), "amalthea", "target", "release", libname)
end
const _LIBAMALTHEA_RK45 = _libamalthea_path_rk45()

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
        ptr = ccall((:init_precon_step_ffi, _LIBAMALTHEA_RK45), Ptr{Cvoid},
                    (Csize_t,), Csize_t(n))
        h = new(ptr)
        finalizer(h) do h
            p = h.ptr
            if p != C_NULL
                ccall((:free_precon_step_ffi, _LIBAMALTHEA_RK45), Cvoid, (Ptr{Cvoid},), p)
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
        rc = ccall((:precon_step_ffi, _LIBAMALTHEA_RK45), Cint,
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
        check_ffi(rc, "precon_step_ffi")
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

# ─── Rust native interaction-picture Stepper FFI (AMALTHEA_USE_RUST_NATIVE=1) ───────

"""
    NativeIneligible(msg)

Thrown by `RustNativeStepper`'s constructor when the requested geometry/
config is outside the native port's current scope (any of the per-phase
narrowing restrictions documented in `docs/dev/native-port/MATH.md` — EnvGrid
variants, `full=true` modal, `thg=false` Raman, an unsupported/ineligible
extra response such as plasma or Raman on a Kerr-only geometry, ...).

This is a **distinct exception type from a bare `error()`** specifically so
`solve_precon` (Phase 8, default-flip) can catch *this* and silently fall
back to the Julia stepper, while letting a genuine bug (an FFI call
returning a nonzero error code, a shape/invariant mismatch, `init_native_sim`
returning `C_NULL`, ...) propagate and crash loudly as before. Before Phase
8, native was opt-in only, so a scope-restriction `error()` reaching the user
meant they had deliberately turned on `AMALTHEA_USE_RUST_NATIVE` for an
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
    check_ffi(rc, what; ineligible=false)

Check a Rust FFI call's `Cint` return code, `0` meaning success (the
universal convention across every `native_*`/`precon_step_ffi`/
`get_ks_stage` entry point). On a nonzero `rc`: `ineligible=true` throws
[`NativeIneligible`](@ref) (the calling config legitimately falls back to
the Julia stepper — used where a Rust-side failure genuinely means "this
input combination isn't wired", e.g. a plasma/Raman setter rejecting a
combination Julia's own eligibility checks couldn't already rule out); the
default `ineligible=false` raises a hard `error` (an invariant violation
that should never happen if the surrounding Julia code is correct — a null
pointer, a length mismatch, a stale handle — and must not be silently
swallowed as ineligibility, see `NativeIneligible`'s own docstring).

Centralizes what was 27 duplicated `rc == 0 || error(...)`/
`rc == 0 || throw(NativeIneligible(...))` call sites (docs/dev/BACKLOG.md S4 item 2;
suggestion 7's "FFI error enum"). The Rust side's return codes are still a
plain `0 = success` convention today — the ineligible-vs-bug classification
is inherently a call-site policy decision (the same numeric nonzero code
means "ineligible" at one call site and "bug" at another, depending on what
Julia has already validated before making the call), so it lives here as an
explicit keyword rather than being encoded in the numeric code itself.
"""
function check_ffi(rc, what; ineligible::Bool=false)
    rc == 0 && return nothing
    if ineligible
        throw(NativeIneligible("$what returned $rc"))
    else
        error("$what failed: $rc")
    end
end

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
    function RustNativeSimHandle(linop::Array{ComplexF64}; use_gpu::Bool=false)
        n = length(linop)
        ptr = if use_gpu
            ccall((:init_cuda_native_sim, _LIBAMALTHEA_RK45), Ptr{Cvoid},
                  (Ptr{ComplexF64}, Csize_t), pointer(linop), Csize_t(n))
        else
            ccall((:init_native_sim, _LIBAMALTHEA_RK45), Ptr{Cvoid},
                  (Ptr{ComplexF64}, Csize_t), pointer(linop), Csize_t(n))
        end
        h = new(ptr)
        finalizer(h) do h
            p = h.ptr
            if p != C_NULL
                ccall((:free_native_sim, _LIBAMALTHEA_RK45), Cvoid, (Ptr{Cvoid},), p)
                h.ptr = C_NULL
            end
        end
        h
    end
    # Phase 7: z-dependent linop — `linop` is a `Capillary.ZDepLinopMarcatili`
    # wrapper, not an `Array{ComplexF64}`, so there is no fixed array to seed
    # with. Untyped (not `Amalthea.Capillary.ZDepLinopMarcatili`) because RK45.jl
    # is `include`d before Capillary.jl in Amalthea.jl — the caller
    # (`RustNativeStepper`) already gates this constructor on a runtime
    # `isa` check, so no eager type resolution is needed here; dispatch on
    # 2-arg vs 1-arg is unambiguous regardless. `init_native_sim` only uses
    # the seed to size `n` and populate the initial `s.linop`;
    # `ensure_linop_at` overwrites it before it is ever read (the first
    # `native_step` call), so a zero seed is exact, not an approximation.
    function RustNativeSimHandle(linop, n::Int)
        seed = zeros(ComplexF64, n)
        ptr = ccall((:init_native_sim, _LIBAMALTHEA_RK45), Ptr{Cvoid},
                    (Ptr{ComplexF64}, Csize_t), pointer(seed), Csize_t(n))
        h = new(ptr)
        finalizer(h) do h
            p = h.ptr
            if p != C_NULL
                ccall((:free_native_sim, _LIBAMALTHEA_RK45), Cvoid, (Ptr{Cvoid},), p)
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
    # GC-safety: `_handle`'s Rust-side `NativeSim` stores *raw pointers* into
    # a small number of Julia-owned, Rust-allocated objects that are NOT
    # copied at setup time (unlike e.g. the Raman/dispersion oscillator
    # data, which Rust copies into its own `Vec`s once and never touches
    # the original Julia pointer again) — specifically the PPT/ADK
    # ionization-rate handle (`plasma_ion_ptr`/`plasma_adk_ptr` in
    # `native.rs`), re-dereferenced on every `native_step` call for the
    # stepper's entire lifetime. Those handles (`RustIonizationHandle`/
    # `RustAdkHandle` in `Ionisation.jl`) carry a GC finalizer that frees
    # the underlying Rust memory. Before this field existed, nothing kept
    # them Julia-reachable past construction (`RustNativeStepper` never
    # stored `f!`/`irf`), so Julia's GC — which runs concurrently with a
    # blocking `ccall`, having no visibility into native Rust threads still
    # holding the raw pointer — could finalize (free) the handle mid-`solve`,
    # causing a use-after-free. Confirmed as the root cause of docs/dev/BACKLOG.md S2
    # Phase 3's reverted heap-corruption bug: reproduced the crash (a rayon
    # worker segfaulting inside `PptIonizationRate::rate`'s `Vec` access),
    # confirmed `evaluate()`'s binary search/`get_unchecked` is correct and
    # thread-safe in isolation, and traced the actual defect to this missing
    # GC root — threading only widened the race window (longer-lived worker
    # threads, more concurrent allocation pressure), it didn't cause the
    # unsoundness. Every Julia object referenced here must simply stay
    # alive; nothing reads through `_gc_roots` directly.
    _gc_roots::Vector{Any}
end

function RustNativeStepper(linop, y0, t, dt; kwargs...)
    RustNativeStepper(nothing, linop, y0, t, dt; kwargs...)
end

"""
    _gpu_kernel_supports(f!, linop)

True iff the experimental GPU-resident stepper (`CudaNativeSim`,
`amalthea/src/cuda_native.rs`) can handle this exact config: mode-averaged
(`TransModeAvg`), RealGrid, a constant (non-z-dependent) linop, a scalar
(non-mixture) density, no shot noise, exactly one plain Kerr response, and
at most one plasma response using PPT ionisation (`IonRatePPTAccel` —
`CudaNativeSim::set_plasma_params_adk` still returns -1, ADK is not
implemented on the GPU path yet; docs/dev/BACKLOG.md S3 item 2). No other response
kind (e.g. Raman) is implemented on `CudaNativeSim` — every `set_*_params`
beyond `set_mode_avg_params`/`set_plasma_params` returns -1.

Pure config-shape check: does not read the `cuda_native`/`gpu_dispatch`
toggles or the problem size — see [`_gpu_native_eligible`](@ref) for the full
dispatch decision.
"""
function _gpu_kernel_supports(f!, linop)
    f! isa Amalthea.NonlinearRHS.TransModeAvg || return false
    linop isa Array{ComplexF64} || return false
    f!.grid isa Amalthea.Grid.RealGrid || return false
    f!.densityfun(0.0) isa Real || return false
    isnothing(f!.Et_noise) || return false
    kerr_resps = filter(_is_plain_kerr_resp, f!.resp)
    plasma_resps = filter(r -> r isa Amalthea.Nonlinear.PlasmaCumtrapz, f!.resp)
    length(kerr_resps) == 1 || return false
    length(plasma_resps) + length(kerr_resps) == length(f!.resp) || return false
    length(plasma_resps) <= 1 || return false
    isempty(plasma_resps) || plasma_resps[1].ratefunc isa Amalthea.Ionisation.IonRatePPTAccel || return false
    true
end

"""
    _GPU_KERR_ONLY_N_THRESHOLD

Minimum problem size (`length(y0)`, i.e. Nω) at which the GPU-resident
stepper measurably outperforms the CPU-resident one for a Kerr-only
mode-averaged config — measured, not guessed (docs/dev/BACKLOG.md S3 item 3),
on real hardware (RTX 5060 Ti, 2026-07-16). Below this the GPU path is
dominated by CUDA kernel-launch/sync overhead (`cuda_native.rs::step`'s
`launch_checked` synchronizes after every one of ~dozens of per-stage kernel
launches); above it, cuFFT throughput wins. Measured sweep (mode-avg,
RealGrid, Kerr-only, `plasma=false`, fixed-step `native_step`, 10-iteration
average after a 3-iteration warmup):

| n (=length(Eω)) | CPU µs/step | GPU µs/step | GPU speedup |
|---:|---:|---:|---:|
| 2,049   |    427  |  1,014 | 0.42x |
| 4,097   |    899  |    877 | 1.03x |
| 8,193   |  1,849  |  1,412 | 1.31x |
| 16,385  |  4,152  |    802 | 5.17x |
| 32,769  |  9,684  |  1,188 | 8.15x |
| 65,537  | 20,689  |  1,471 | 14.07x |
| 131,073 | 47,514  |  3,440 | 13.81x |
| 262,145 | 139,084 |  5,168 | 26.91x |

16384 sits just above the measured breakeven (n=8,193, 1.31x) with margin,
rather than at the exact crossover, since GPU time isn't a clean monotonic
function of n at these small sizes (compare n=2,049 vs. n=16,385 above —
launch/sync overhead noise, not throughput, dominates down there).

**This threshold does NOT apply when a plasma response is present.** The
same sweep repeated with `plasma=true` (PPT, `gas=:Ar`) showed the GPU path
consistently 20-30x *slower* than CPU at every size up to n=131,073,
*worsening* with n rather than improving —
`plasma_fraction_kernel`/`plasma_current_kernel`/`plasma_polarization_kernel`
are documented single-GPU-thread sequential scans (BACKLOG.md S3 item 2/4/6,
a deliberate V1 tradeoff), which don't benefit from n the way cuFFT does. No
crossover was found in the tested range, so `:auto` dispatch excludes any
plasma-bearing config entirely rather than guessing an unsupported threshold.
"""
const _GPU_KERR_ONLY_N_THRESHOLD = 16384

"""
    _gpu_native_eligible(f!, linop, n)

Full GPU dispatch decision for `RustNativeStepper`: `true` iff
[`_gpu_kernel_supports`](@ref)`(f!, linop)` holds AND the current
`AMALTHEA_NATIVE_GPU` dispatch policy
(`Amalthea.Config.backend_config().gpu_dispatch`, one of `:off`/`:on`/`:auto`,
default `:auto`) selects GPU for this exact config:

- `:off` — never (forces CPU even when `AMALTHEA_USE_RUST_CUDA_NATIVE=1`).
- `:on` — always, whenever the kernel supports it (the old unconditional
  behavior — useful to force GPU on a small config, e.g. to reproduce a
  specific benchmark/test run regardless of the measured threshold).
- `:auto` (default) — GPU only for a plasma-free config with
  `n >= _GPU_KERR_ONLY_N_THRESHOLD`; see that constant's docstring for the
  measured data this is based on.

Also requires the `AMALTHEA_USE_RUST_CUDA_NATIVE=1` master opt-in
(`cuda_native` field) regardless of `gpu_dispatch` — Rust's
`init_cuda_native_sim` re-checks that same env var independently, so this
function is purely to avoid attempting GPU init for an ineligible config
where it would return a confusing null pointer instead of the real reason.
"""
function _gpu_native_eligible(f!, linop, n::Integer)
    cfg = Amalthea.Config.backend_config()
    cfg.cuda_native || return false
    _gpu_kernel_supports(f!, linop) || return false
    cfg.gpu_dispatch === :off && return false
    cfg.gpu_dispatch === :on && return true
    # :auto — plasma-free and large enough (see _GPU_KERR_ONLY_N_THRESHOLD)
    any(r -> r isa Amalthea.Nonlinear.PlasmaCumtrapz, f!.resp) && return false
    n >= _GPU_KERR_ONLY_N_THRESHOLD
end

function RustNativeStepper(f!, linop, y0, t, dt;
                            rtol=1e-6, atol=1e-10, safety=0.9, max_dt=Inf, min_dt=0,
                            locextrap=true, norm=weaknorm, flength=Inf,
                            native_threads::Integer=Threads.nthreads())
    n = length(y0)
    is_zdep_mode_avg = linop isa Amalthea.Capillary.ZDepLinopMarcatili
    is_zdep_free_linop = linop isa Amalthea.LinearOps.ZDepLinopFree
    is_zdep_modal_taper = linop isa Amalthea.Capillary.ZDepLinopModalTaper
    if is_zdep_mode_avg || is_zdep_free_linop || is_zdep_modal_taper
        isfinite(flength) || throw(NativeIneligible("z-dependent linop requires a " *
                  "finite `flength` (propagation length) to build the resident z-LUTs " *
                  "— got flength=$flength."))
        handle = RustNativeSimHandle(linop, n)
    else
        handle = RustNativeSimHandle(linop; use_gpu=_gpu_native_eligible(f!, linop, n))
    end

    handle.ptr == C_NULL && error("init_native_sim failed")

    # See `RustNativeStepper`'s `_gc_roots` field doc for why this exists:
    # collects strong references to every Julia object whose Rust-side
    # pointer `_handle` stores persistently (re-dereferenced on later
    # `native_step` calls), so Julia's GC can't finalize them mid-solve.
    gc_roots = Any[]

    is_mode_avg = f! isa Amalthea.NonlinearRHS.TransModeAvg
    is_radial = f! isa Amalthea.NonlinearRHS.TransRadial
    is_modal = f! isa Amalthea.NonlinearRHS.TransModal
    is_free = f! isa Amalthea.NonlinearRHS.TransFree

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
              "Amalthea's own NonlinearRHS pipeline types."))

    if is_mode_avg || is_radial || is_modal || is_free
        # Gas mixtures (`MarcatiliMode(a, (gas1,gas2,...), (p1,p2,...))`) give
        # `densityfun(z)` a per-species Vector return type and a nested
        # tuple-of-tuples `resp` (one response tuple per component). Since
        # Phase F item 4 / Phase I item 4 (docs/dev/BACKLOG.md), all four
        # native RHS geometries (mode-averaged, radial, modal, free-space)
        # handle this — Kerr's linearity in density·γ3 collapses a
        # per-species mixture to one scalar `kerr_fac` regardless of
        # geometry, since the only place density enters any of their RHS
        # kernels is the same generic `NonlinearRHS.Et_to_Pt!`, which
        # already dispatches per-species for an `AbstractVector` density —
        # see each `is_*` block below for the per-geometry discriminating
        # check and mixture branch. Anything that is not a scalar or a
        # per-species `Vector{<:Real}` would otherwise silently compute
        # garbage `kerr_fac`/`beta` that only fails much later at the FFI
        # boundary with an opaque `MethodError` — reject it here instead,
        # with a message that says what's actually unsupported.
        dens0 = f!.densityfun(0.0)
        dens0 isa Real || dens0 isa AbstractVector{<:Real} ||
            throw(NativeIneligible("densityfun(z) returned $(typeof(dens0)) — the " *
                  "native path only understands a scalar density or a per-species " *
                  "Vector{<:Real} (gas mixture)."))

        grid = f!.grid
        is_real_grid = grid isa Amalthea.Grid.RealGrid   # EnvGrid uses c2c FFTs
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

    # docs/dev/BACKLOG.md S1 item 1: best-effort planner wisdom, dedicated to the
    # native path (a separate file from Utils.loadFFTwisdom's own
    # `FFTWcache_*threads`, not shared — avoids any concurrent-access
    # interaction with Julia's own `mkpidlock`-guarded load/save). A missing
    # file is a no-op on the Rust side (see `set_fftw_plans`'s doc), so no
    # existence check is needed here; `native_wisdom_path` is only `nothing`
    # if `Utils.cachedir()` itself throws (kept purely best-effort — this
    # must never block native construction).
    # Empty string (not C_NULL — keeps the ccall argument type-stably
    # `Cstring`) if `cachedir()` throws; `CStr::from_ptr("").to_str()` on the
    # Rust side is `Ok("")`, and opening a file at `""` just fails the
    # best-effort import, same as a missing file.
    # Persistence is opt-in (`_native_wisdom_enabled`, default off — see its
    # docstring above); when off, pass "" so Rust does not import anything.
    native_wisdom_path::String = if _native_wisdom_enabled()
        try
            joinpath(Amalthea.Utils.cachedir(), "native_fftw_wisdom_$(Amalthea.Utils.FFTWthreads())threads")
        catch
            ""
        end
    else
        ""
    end

    rc = ccall((:native_set_fftw_plans, _LIBAMALTHEA_RK45), Cint,
          (Ptr{Cvoid}, Cstring, Csize_t, Csize_t, Cint, Cuint, Cstring),
          handle.ptr, lib_path, n_time, n_time_over, is_real_grid ? 1 : 0, 64,
          native_wisdom_path)
    check_ffi(rc, "native_set_fftw_plans")

    # docs/dev/BACKLOG.md S2 item 1: rayon thread count for later-phase parallel RHS
    # seams. Defaults to `Threads.nthreads()` (Julia's own thread count) —
    # automatic, not something users set separately — but can be overridden
    # via `native_threads` (a Rust-side-only rayon thread count, independent
    # of Julia's own threading model) for equivalence testing without
    # restarting Julia. `n<=1` is sequential on the Rust side (the pre-S2
    # default), so single-threaded Julia sessions see no behavior change.
    rc = ccall((:native_set_threads, _LIBAMALTHEA_RK45), Cint,
          (Ptr{Cvoid}, Csize_t), handle.ptr, native_threads)
    check_ffi(rc, "native_set_threads")

    # docs/dev/BACKLOG.md S5.2: deterministic mode. `AMALTHEA_NATIVE_DETERMINISTIC=1`
    # forces the radial-geometry QDHT to skip the BLAS-3 `dgemm` path
    # (`AMALTHEA_QDHT_BLAS`) even if it's enabled, using the row-parallel Rayon
    # fallback instead. The native path is already run-to-run deterministic
    # on one machine by default — every parallel seam here (the QDHT
    # fallback, the older per-kernel `AMALTHEA_USE_RUST_QDHT` batch loops) is
    # embarrassingly parallel with no cross-thread reduction, and native
    # FFTW plans only ever use `FFTW_ESTIMATE`. The real lever this flag
    # has: `BLAS_API` (amalthea's `blas.rs`) is a process-global
    # `OnceLock`, populated only by the per-kernel
    # `AMALTHEA_USE_RUST_QDHT`+`AMALTHEA_QDHT_BLAS` path — once that happens
    # anywhere in the process, every *later* native-path QDHT call becomes
    # eligible for BLAS too. `deterministic=true` makes that eligibility
    # invariant to whether some other part of the process touched BLAS,
    # and to which BLAS library/thread-count is linked — not a claim that
    # a fixed BLAS state is otherwise repeat-run unstable at these problem
    # sizes. Default off (matches every prior behavior exactly).
    # **Not guaranteed:** bit-identical results across different
    # machines/CPU targets — the crate is built with `target-cpu=native`,
    # so a different build host takes a different SIMD/libm path.
    rc = ccall((:native_set_deterministic, _LIBAMALTHEA_RK45), Cint,
          (Ptr{Cvoid}, Cint), handle.ptr, Amalthea.Config.backend_config().deterministic ? 1 : 0)
    check_ffi(rc, "native_set_deterministic")

    # Set parameters if mode-averaged
    if is_mode_avg
        norm_func = f!.norm!
        pre = Amalthea.NonlinearRHS.norm_pre(norm_func)
        bfun! = Amalthea.NonlinearRHS.norm_βfun(norm_func)

        if !isnothing(bfun!)
            beta = zeros(Float64, length(pre))
            bfun!(beta, 0.0)
        else
            beta = ones(Float64, length(pre))
        end

        towin = f!.grid.towin
        owin = f!.grid.ωwin
        sidx = UInt8.(f!.grid.sidx)

        is_zdep_mode_avg || _check_density_zindependent(f!.densityfun, flength)
        density = f!.densityfun(0.0)
        is_mixture = density isa AbstractVector

        if is_mixture
            # Gas mixtures (Phase F item 4): `densityfun(z)` returns one
            # density per species and `f!.resp` is a tuple of per-species
            # response tuples (Amalthea's own low-level mixture convention —
            # see test_mixtures.jl). Kerr is the only nonlinearity handled
            # here: `NonlinearRHS.Et_to_Pt!`'s `AbstractVector`-density
            # branch calls each species' response tuple with that species'
            # own density and accumulates into the same `Pt` buffer, so for
            # a plain Kerr response (linear in density·γ3) the whole sum
            # collapses into one scalar — no per-species vector is needed on
            # the Rust side at all: kerr_fac = Σᵢ densityᵢ·ε₀·γ3ᵢ. Plasma and
            # Raman mixtures are not a simple linear sum like this, so they
            # are rejected below rather than silently ignored.
            is_zdep_mode_avg && throw(NativeIneligible("gas mixtures are not supported " *
                  "together with a z-dependent (pressure-gradient) linop."))
            length(f!.resp) == length(density) ||
                throw(NativeIneligible("mixture densityfun(z) returned $(length(density)) " *
                      "species but f!.resp has $(length(f!.resp)) entries — shape mismatch."))
            kerr_fac = 0.0
            for (ρi, respi) in zip(density, f!.resp)
                length(respi) == 1 && _is_plain_kerr_resp(respi[1]) ||
                    throw(NativeIneligible("mixture species response $(respi) is not a " *
                          "single plain Kerr response — only Kerr is wired for the native " *
                          "mixture path (Phase F item 4); plasma/Raman mixtures are out of " *
                          "scope."))
                γ3i = Amalthea.Nonlinear.kerr_γ3((respi[1],))
                kerr_fac += ρi * Amalthea.PhysData.ε_0 * γ3i
            end
        else
            γ3 = Amalthea.Nonlinear.kerr_γ3(f!.resp)
            kerr_fac = density * Amalthea.PhysData.ε_0 * γ3
        end

        nlscale = Amalthea.NonlinearRHS.nlscale
        sqrt_aeff = sqrt(f!.aeff(0.0))

        rc = ccall((:native_set_mode_avg_params, _LIBAMALTHEA_RK45), Cint,
            (Ptr{Cvoid}, Csize_t, Csize_t,
             Ptr{Float64}, Ptr{Float64}, Ptr{UInt8},
             Ptr{Float64}, Ptr{Float64}, Ptr{Float64},
             Float64, Float64, Float64),
            handle.ptr, n_time, n_time_over,
            towin, owin, sidx,
            real.(pre), imag.(pre), beta,
            kerr_fac, nlscale, sqrt_aeff)
        check_ffi(rc, "native_set_mode_avg_params")

        # Phase I item 1: wire modified shot-noise. `f!.Et_noise` is a constant
        # buffer of length n_time_over (Float64 for RealGrid, ComplexF64 for
        # EnvGrid), precomputed once at TransModeAvg construction time
        # (NonlinearRHS.jl:534-535); the Rust RHS adds it (rescaled by 1/sc
        # each call) to `eto`/`eto_cplx`, matching `Et_nl = Eto + Et_noise/sc`.
        if !isnothing(f!.Et_noise)
            if is_real_grid
                rc = ccall((:native_set_mode_avg_noise, _LIBAMALTHEA_RK45), Cint,
                    (Ptr{Cvoid}, Ptr{Float64}, Csize_t),
                    handle.ptr, f!.Et_noise, length(f!.Et_noise))
                check_ffi(rc, "native_set_mode_avg_noise")
            else
                rc = ccall((:native_set_mode_avg_noise_cplx, _LIBAMALTHEA_RK45), Cint,
                    (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64}, Csize_t),
                    handle.ptr, real.(f!.Et_noise), imag.(f!.Et_noise), length(f!.Et_noise))
                check_ffi(rc, "native_set_mode_avg_noise_cplx")
            end
        end

        # Wire plasma if present and a Rust ionization handle is available.
        # Since Phase C (docs/dev/BACKLOG.md), `IonRatePPTAccel` builds this handle
        # whenever the Rust library is present and EITHER
        # `AMALTHEA_USE_RUST_IONISATION=1` OR the native stepper itself is
        # enabled (`AMALTHEA_USE_RUST_NATIVE` defaults to `"1"` since Phase 8) —
        # decoupling it from the opt-in kernel toggle so the fork's default
        # field-resolved workload (`plasma = !envelope`) actually runs
        # natively out of the box (REVIEW.md §3.2). If the handle is still
        # missing here (library absent, or the user explicitly forced both
        # toggles off), the WHOLE construction must fail (NativeIneligible →
        # solve_precon falls back to the Julia stepper) rather than silently
        # continuing without plasma — continuing native-minus-plasma would be
        # wrong physics with only a log line to notice it, which became
        # unacceptable once native became the default (Phase 8) rather than
        # opt-in. Mixtures are Kerr-only by construction (validated above),
        # so none of this applies to them.
        for r in (is_mixture ? () : f!.resp)
            if r isa Amalthea.Nonlinear.PlasmaCumtrapz
                irf = r.ratefunc
                if irf isa Amalthea.Ionisation.IonRatePPTAccel &&
                        !isnothing(irf.rust_handle) &&
                        irf.rust_handle.ptr != C_NULL
                    rc = ccall((:native_set_plasma_params, _LIBAMALTHEA_RK45), Cint,
                        (Ptr{Cvoid}, Ptr{Cvoid}, Float64, Float64, Float64, Float64, Float64),
                        handle.ptr, irf.rust_handle.ptr,
                        r.ionpot, Amalthea.PhysData.e_ratio, r.preionfrac, r.δt, density)
                    check_ffi(rc, "native_set_plasma_params"; ineligible=true)
                    push!(gc_roots, irf.rust_handle)
                elseif irf isa Amalthea.Ionisation.IonRateADK &&
                        !isnothing(irf.rust_handle) &&
                        irf.rust_handle.ptr != C_NULL
                    # Phase I item 3: ADK is closed-form, no LUT — same
                    # eligibility gate, a different (parameter-only) FFI setter.
                    rc = ccall((:native_set_plasma_params_adk, _LIBAMALTHEA_RK45), Cint,
                        (Ptr{Cvoid}, Ptr{Cvoid}, Float64, Float64, Float64, Float64, Float64),
                        handle.ptr, irf.rust_handle.ptr,
                        r.ionpot, Amalthea.PhysData.e_ratio, r.preionfrac, r.δt, density)
                    check_ffi(rc, "native_set_plasma_params_adk"; ineligible=true)
                    push!(gc_roots, irf.rust_handle)
                else
                    throw(NativeIneligible("plasma detected but no Rust ionisation handle is " *
                          "available (library missing, or both AMALTHEA_USE_RUST_IONISATION and " *
                          "AMALTHEA_USE_RUST_NATIVE were explicitly disabled), or ratefunc is not " *
                          "an IonRatePPTAccel/IonRateADK — this config is not eligible for the " *
                          "native path."))
                end
                break  # only one plasma response expected
            end
        end

        # Wire Raman if present and eligible for the resident ADE path. The
        # resident path needs the raw oscillator arrays (not just a handle
        # pointer), so eligibility is re-checked here. Since Phase F.2:
        # `Raman.flatten_sdo_oscillators` expands any `RamanRespRotationalNonRigid`
        # into its per-J SDOs (all sharing one τ2ρ), so a rotational (or
        # rotational+vibrational) response reaches the solver too — the FFI
        # already accepted n_osc>1, only the Julia-side extraction was
        # SDO-only. `gammas` is evaluated at the sim's actual constant
        # density (already computed below as `raman_density` for the FFI
        # call), not a density=1.0 placeholder, so a density-dependent-τ2
        # gas (e.g. H2's vibrational line) is now eligible too — correct as
        # long as density doesn't vary with z, which holds here since this
        # branch requires either a non-zdep config (density fixed for the
        # whole propagation) or, for a zdep config, is no better/worse than
        # this same latent frozen-at-z=0 approximation already existed for
        # any Raman response before Phase F.2 (z-dependent Raman is not
        # wired at all — out of scope, see docs/dev/BACKLOG.md Phase F item 3).
        # `RamanPolarEnv` (mode-averaged EnvGrid — `rhs_mode_avg_env`'s new
        # Raman step) is accepted alongside `RamanPolarField`; envelope
        # Raman's intensity (`sqr!(R::RamanPolarEnv,E) = 1/2·|E|²`,
        # Nonlinear.jl:351-354) has no `thg` concept, so `thg` is passed as
        # always-true (native.rs never builds the unused Hilbert plan for
        # it). Both `thg=true`/`thg=false` `RamanPolarField` (Phase F.1) are
        # still supported. An ineligible Raman response must fail the whole
        # construction (NativeIneligible), not silently continue without it.
        # Mixtures are Kerr-only by construction (validated above).
        for r in (is_mixture ? () : f!.resp)
            if r isa Amalthea.Nonlinear.RamanPolarField || r isa Amalthea.Nonlinear.RamanPolarEnv
                rr = r.r
                if r isa Amalthea.Nonlinear.RamanPolarEnv && rr isa Amalthea.Raman.RamanRespIntermediateBroadening
                    # Phase I item 2: `ramanmodel=:SiO2` (only reachable via
                    # `prop_gnlse`, mode-averaged EnvGrid, RamanPolarEnv) —
                    # a Gaussian-damped multi-line response with no finite-SDO
                    # decomposition (see docs/dev/BACKLOG.md), so it needs the resident
                    # FFT-convolution kernel instead of the ADE solver below.
                    raman_density = f!.densityfun(0.0)
                    rc = ccall((:native_set_raman_fft_params, _LIBAMALTHEA_RK45), Cint,
                        (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Csize_t,
                         Float64, Float64, Csize_t, Float64),
                        handle.ptr, rr.ωi, rr.Ai, rr.Γi, rr.γi, Csize_t(length(rr.ωi)),
                        rr.scale, r.dt, Csize_t(length(rr.t)), raman_density)
                    check_ffi(rc, "native_set_raman_fft_params"; ineligible=true)
                    break  # only one Raman response expected
                end
                Rs = rr isa Amalthea.Raman.CombinedRamanResponse ?
                    Amalthea.Raman.flatten_sdo_oscillators(rr) : nothing
                isnothing(Rs) && throw(NativeIneligible("Raman response present but not " *
                      "eligible for the native path (needs a CombinedRamanResponse whose " *
                      "components are all RamanRespSingleDampedOscillator and/or " *
                      "RamanRespRotationalNonRigid — e.g. not an intermediate-broadening " *
                      "response)."))
                raman_density = f!.densityfun(0.0)
                omegas    = Float64[ri.Ω for ri in Rs]
                gammas    = Float64[1.0 / ri.τ2ρ(raman_density) for ri in Rs]
                couplings = Float64[ri.K for ri in Rs]
                thg_flag = r isa Amalthea.Nonlinear.RamanPolarField ? Cint(r.thg) : Cint(1)
                rc = ccall((:native_set_raman_params, _LIBAMALTHEA_RK45), Cint,
                    (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Csize_t, Float64, Float64, Cint),
                    handle.ptr, omegas, gammas, couplings, Csize_t(length(omegas)),
                    r.dt, raman_density, thg_flag)
                check_ffi(rc, "native_set_raman_params"; ineligible=true)
                break  # only one Raman response expected
            end
        end

        # Any response type not recognized by the checks above (Kerr via the
        # γ3 field scan, `PlasmaCumtrapz`, `RamanPolarField`/`RamanPolarEnv`
        # — the latter native since Phase F.2) must fail the whole
        # construction too, so an unrecognized response can't silently slip
        # through with no wiring and no error: exactly the silent-wrong-
        # physics gap this file's `NativeIneligible` conversion (Phase 8)
        # exists to close. Found via `test_gnlse.jl`'s "Soliton shift" test
        # once native became the default — that test's self-frequency-shift
        # numbers depend entirely on Raman, so silently dropping it produced
        # a completely different (not just numerically close) result.
        # Mixtures are validated above (each species checked individually);
        # this catch-all only applies to the single-species case.
        is_kerr_resp(r) = _is_plain_kerr_resp(r)
        for r in (is_mixture ? () : f!.resp)
            is_kerr_resp(r) || r isa Amalthea.Nonlinear.PlasmaCumtrapz ||
                r isa Amalthea.Nonlinear.RamanPolarField || r isa Amalthea.Nonlinear.RamanPolarEnv ||
                throw(NativeIneligible("response type $(typeof(r)) is not wired for " *
                      "the native mode-averaged path (this rejects Kerr_field_nothg/" *
                      "Kerr_env_thg too — see `_is_plain_kerr_resp`)."))
        end
    end

    # Set parameters for the z-dependent linop (Phase 7) — mode-averaged,
    # graded-core constant-radius MarcatiliMode only. See MATH.md §3.5 and
    # docs/dev/native-port/BETA1_ANALYTIC.md. dens(pressure) is TRANSFERRED
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
        w.Z[end] == flength || error("RustNativeStepper: z-dependent linop's gradient " *
                  "length ($(w.Z[end])) does not match the propagation length " *
                  "flength=$flength passed to solve_precon — these should always be the " *
                  "same quantity (Capillary.gradient's last breakpoint vs `grid.zmax`/" *
                  "`tmax`). Refusing to guess which is correct.")
        model_code = w.model == :full ? Cuint(0) : Cuint(1)
        loss_code = w.loss ? Cuint(1) : Cuint(0)
        # density(z)·ε₀·γ3 = kerr_fac(z): γ3 must be re-scaled by the
        # z-dependent density every RK stage (TransModeAvg calls
        # `densityfun(z)` fresh, unlike the constant-medium phases which
        # bake `density(0)` into `kerr_fac` once) — see `ensure_linop_at`.
        eps0_gamma3 = Amalthea.PhysData.ε_0 * γ3

        rc = ccall((:native_set_zdep_mode_avg_params, _LIBAMALTHEA_RK45), Cint,
            (Ptr{Cvoid}, Csize_t, Ptr{Float64}, Ptr{Float64},
             Csize_t, Ptr{Float64}, Ptr{Float64}, Ptr{Float64},
             Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64},
             Cuint, Cuint, Float64,
             Float64, Float64, Float64, Float64, Float64, Float64, Float64),
            handle.ptr, Csize_t(length(w.Z)), w.Z, w.P,
            Csize_t(length(w.dspl_x)), w.dspl_x, w.dspl_y, w.dspl_d,
            w.gamma, w.nwg_re, w.nwg_im, w.omega,
            model_code, loss_code, eps0_gamma3,
            w.ω0, w.γ0, w.dγ0, w.nwg0_re, w.nwg0_im, w.dnwg0_re, w.dnwg0_im)
        check_ffi(rc, "native_set_zdep_mode_avg_params")
    end

    # Set parameters if radial (TransRadial) — Phase 3 gate: scalar Kerr only,
    # RealGrid or EnvGrid (EnvGrid added Phase D.1, docs/dev/BACKLOG.md — `rhs_radial_env`
    # in native.rs, dispatched on `sim.is_real` exactly like mode-averaged's
    # real/env split). Phase D.2 (docs/dev/BACKLOG.md) adds plasma alongside Kerr
    # (`apply_plasma_radial` in native.rs, RealGrid only — Julia has no
    # EnvGrid plasma either). Phase D.4 adds Raman (`RamanPolarField`,
    # RealGrid); the radial-env-Raman follow-up adds `RamanPolarEnv`
    # (EnvGrid, `apply_raman_radial_env` in native.rs) — closes
    # NATIVE_SUPPORT_MATRIX.md's last radial-Raman gap. See
    # docs/dev/native-port/MATH.md §3.2 for the design (precomputed
    # normalization array M, resident QdhtFfiHandle reused directly); `M`'s
    # length (`n_spec`) and the FFI's internal buffer sizing both already
    # generalize to either grid type without further changes here.
    if is_radial
        HT = f!.QDHT
        n_r = HT.N
        n % n_r == 0 || error("RustNativeStepper: field length $n not divisible by QDHT.N=$n_r")

        T_mat = Matrix{Float64}(HT.T)
        size(T_mat) == (n_r, n_r) || error("RustNativeStepper: QDHT.T shape mismatch")
        scale_fwd = Float64(HT.scaleRK)
        scale_inv = 1.0 / scale_fwd

        towin = f!.grid.towin

        _check_density_zindependent(f!.densityfun, flength)
        density = f!.densityfun(0.0)
        is_mixture = density isa AbstractVector

        if is_mixture
            # docs/dev/BACKLOG.md Phase I item 4: same reasoning as the mode-averaged
            # mixture branch above — Kerr is linear in density·γ3, so it
            # collapses to one scalar kerr_fac regardless of geometry;
            # plasma/Raman mixtures are not a simple sum, so they're
            # rejected here rather than silently ignored (radial's plasma/
            # Raman wiring below assumes a single scalar density).
            length(f!.resp) == length(density) ||
                throw(NativeIneligible("radial mixture: densityfun(z) returned " *
                      "$(length(density)) species but f!.resp has $(length(f!.resp)) " *
                      "entries — shape mismatch."))
            kerr_fac = 0.0
            for (ρi, respi) in zip(density, f!.resp)
                length(respi) == 1 && _is_plain_kerr_resp(respi[1]) ||
                    throw(NativeIneligible("radial mixture species response $(respi) is " *
                          "not a single plain Kerr response — only Kerr is wired for the " *
                          "native radial mixture path (Phase I item 4); plasma/Raman " *
                          "mixtures are out of scope."))
                γ3i = Amalthea.Nonlinear.kerr_γ3((respi[1],))
                kerr_fac += ρi * Amalthea.PhysData.ε_0 * γ3i
            end
        else
            γ3 = Amalthea.Nonlinear.kerr_γ3(f!.resp)
            # Phase D.2 allows Kerr + a plasma response; Phase D.4 (docs/dev/BACKLOG.md)
            # adds Raman (`RamanPolarField`, RealGrid, either `thg` value since
            # Phase F.1 — same eligibility criteria as the mode-averaged Raman
            # wiring below: an all-SDO `CombinedRamanResponse` with
            # density-independent τ2). `RamanPolarEnv` (EnvGrid's envelope
            # Raman) closes the last radial-Raman gap in
            # NATIVE_SUPPORT_MATRIX.md's Radial table — `rhs_radial_env`
            # (native.rs) now has an envelope Raman step (`apply_raman_radial_env`)
            # mirroring `rhs_mode_avg_env`'s mode-averaged envelope Raman
            # exactly (same `sqr!(R::RamanPolarEnv,E)=1/2·|E|²` formula, no
            # `thg` concept — see that function's docstring). Intermediate-
            # broadening (`:SiO2`) Raman is NOT covered here — that needs the
            # separate FFT-convolution kernel (`native_set_raman_fft_params`),
            # which is not wired for radial geometry; it is explicitly
            # rejected below, not silently mishandled.
            is_kerr_resp_radial(r) = _is_plain_kerr_resp(r)
            for r in f!.resp
                is_kerr_resp_radial(r) || r isa Amalthea.Nonlinear.PlasmaCumtrapz ||
                    r isa Amalthea.Nonlinear.RamanPolarField ||
                    r isa Amalthea.Nonlinear.RamanPolarEnv ||
                    throw(NativeIneligible("radial: only Kerr, plasma, and/or Raman " *
                          "(RealGrid or EnvGrid) responses are supported by the native path " *
                          "(Phase D.4 / radial-env-Raman gate) — this also rejects " *
                          "Kerr_field_nothg/Kerr_env_thg, see `_is_plain_kerr_resp`."))
            end
            γ3 != 0.0 ||
                throw(NativeIneligible("radial: no Kerr response found (γ3=0) — the native " *
                      "radial path requires a Kerr response."))

            kerr_fac = density * Amalthea.PhysData.ε_0 * γ3
        end

        # Precompute M[iω,ir] = ωwin[iω]·(-i·ω[iω]) / (2·normfun(0.0)[iω,ir]).
        # Folds norm_radial + ωwin into one array. Valid only for a z-invariant
        # normfun (const_norm_radial, constant-medium gas cell) — see MATH.md §3.2.
        normarr = f!.normfun(0.0)
        ω = f!.grid.ω
        ωwin = f!.grid.ωwin
        M = (ωwin .* (-im .* ω)) ./ (2 .* normarr)

        rc = ccall((:native_set_radial_params, _LIBAMALTHEA_RK45), Cint,
            (Ptr{Cvoid}, Csize_t, Csize_t, Csize_t,
             Ptr{Float64}, Float64, Float64,
             Ptr{Float64}, Float64,
             Ptr{Float64}, Ptr{Float64}),
            handle.ptr, n_time, n_time_over, n_r,
            T_mat, scale_fwd, scale_inv,
            towin, kerr_fac,
            real.(M), imag.(M))
        check_ffi(rc, "native_set_radial_params")

        # Phase I item 1: wire modified shot-noise. `f!.Et_noise` is
        # column-major `(n_time_over, n_r)` (Float64 for RealGrid, ComplexF64
        # for EnvGrid), matching Rust's `radial_eto`/`radial_eto_c` layout
        # exactly (NonlinearRHS.jl:660-662).
        if !isnothing(f!.Et_noise)
            if is_real_grid
                rc = ccall((:native_set_radial_noise, _LIBAMALTHEA_RK45), Cint,
                    (Ptr{Cvoid}, Ptr{Float64}, Csize_t),
                    handle.ptr, f!.Et_noise, length(f!.Et_noise))
                check_ffi(rc, "native_set_radial_noise")
            else
                rc = ccall((:native_set_radial_noise_cplx, _LIBAMALTHEA_RK45), Cint,
                    (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64}, Csize_t),
                    handle.ptr, real.(f!.Et_noise), imag.(f!.Et_noise), length(f!.Et_noise))
                check_ffi(rc, "native_set_radial_noise_cplx")
            end
        end

        # Wire plasma if present and a Rust ionisation handle is available —
        # same decoupled-from-AMALTHEA_USE_RUST_IONISATION eligibility as the
        # mode-averaged wiring above (Phase C, docs/dev/BACKLOG.md). Must fail the
        # whole construction (NativeIneligible), not silently continue
        # without plasma, for the same silent-wrong-physics reason.
        for r in f!.resp
            if r isa Amalthea.Nonlinear.PlasmaCumtrapz
                irf = r.ratefunc
                if irf isa Amalthea.Ionisation.IonRatePPTAccel &&
                        !isnothing(irf.rust_handle) &&
                        irf.rust_handle.ptr != C_NULL
                    rc = ccall((:native_set_plasma_params, _LIBAMALTHEA_RK45), Cint,
                        (Ptr{Cvoid}, Ptr{Cvoid}, Float64, Float64, Float64, Float64, Float64),
                        handle.ptr, irf.rust_handle.ptr,
                        r.ionpot, Amalthea.PhysData.e_ratio, r.preionfrac, r.δt, density)
                    check_ffi(rc, "native_set_plasma_params"; ineligible=true)
                    push!(gc_roots, irf.rust_handle)
                elseif irf isa Amalthea.Ionisation.IonRateADK &&
                        !isnothing(irf.rust_handle) &&
                        irf.rust_handle.ptr != C_NULL
                    # Phase I item 3: ADK is closed-form, no LUT — same
                    # eligibility gate, a different (parameter-only) FFI setter.
                    rc = ccall((:native_set_plasma_params_adk, _LIBAMALTHEA_RK45), Cint,
                        (Ptr{Cvoid}, Ptr{Cvoid}, Float64, Float64, Float64, Float64, Float64),
                        handle.ptr, irf.rust_handle.ptr,
                        r.ionpot, Amalthea.PhysData.e_ratio, r.preionfrac, r.δt, density)
                    check_ffi(rc, "native_set_plasma_params_adk"; ineligible=true)
                    push!(gc_roots, irf.rust_handle)
                else
                    throw(NativeIneligible("plasma detected but no Rust ionisation handle is " *
                          "available (library missing, or both AMALTHEA_USE_RUST_IONISATION and " *
                          "AMALTHEA_USE_RUST_NATIVE were explicitly disabled), or ratefunc is not " *
                          "an IonRatePPTAccel/IonRateADK — this config is not eligible for the " *
                          "native path."))
                end
                break  # only one plasma response expected
            end
        end

        # Wire Raman if present and eligible — Phase D.4 (docs/dev/BACKLOG.md), both
        # `thg` values since Phase F.1, rotational multi-oscillator and
        # density-dependent τ2 since Phase F.2 (see the mode-averaged
        # wiring above for the full rationale — identical here). Same FFI
        # call as the mode-averaged wiring; `native_set_raman_params` sizes
        # its scratch buffers as `n_time_over*n_r` when `s.is_radial`
        # (already set by `native_set_radial_params`, called earlier in this
        # block). `RamanPolarEnv` (EnvGrid) accepted alongside
        # `RamanPolarField` (RealGrid) — closes the "Radial EnvGrid Raman"
        # gap in NATIVE_SUPPORT_MATRIX.md's Radial table. Envelope Raman's
        # intensity (`sqr!(R::RamanPolarEnv,E) = 1/2·|E|²`, Nonlinear.jl:387-389)
        # has no `thg` concept, so `thg` is passed as always-true, exactly
        # mirroring the mode-averaged wiring's `thg_flag` convention above
        # (`native.rs` never builds the unused Hilbert plan for it —
        # `apply_raman_radial_env` has no Hilbert path at all). Intermediate-
        # broadening (`:SiO2`, `RamanRespIntermediateBroadening`) is NOT
        # wired for radial geometry (unlike mode-averaged EnvGrid's
        # `native_set_raman_fft_params` path) — rejected explicitly below
        # rather than falling through to the generic "not a
        # CombinedRamanResponse" message, so the failure reason is clear.
        for r in f!.resp
            if r isa Amalthea.Nonlinear.RamanPolarField || r isa Amalthea.Nonlinear.RamanPolarEnv
                rr = r.r
                if r isa Amalthea.Nonlinear.RamanPolarEnv &&
                        rr isa Amalthea.Raman.RamanRespIntermediateBroadening
                    throw(NativeIneligible("radial: RamanRespIntermediateBroadening " *
                          "(:SiO2) is not wired for radial geometry — only " *
                          "CombinedRamanResponse (SDO/rotational) Raman is supported " *
                          "for the native radial path."))
                end
                Rs = rr isa Amalthea.Raman.CombinedRamanResponse ?
                    Amalthea.Raman.flatten_sdo_oscillators(rr) : nothing
                isnothing(Rs) && throw(NativeIneligible("Raman response present but not " *
                      "eligible for the native path (needs a CombinedRamanResponse whose " *
                      "components are all RamanRespSingleDampedOscillator and/or " *
                      "RamanRespRotationalNonRigid)."))
                raman_density = f!.densityfun(0.0)
                omegas    = Float64[ri.Ω for ri in Rs]
                gammas    = Float64[1.0 / ri.τ2ρ(raman_density) for ri in Rs]
                couplings = Float64[ri.K for ri in Rs]
                thg_flag = r isa Amalthea.Nonlinear.RamanPolarField ? Cint(r.thg) : Cint(1)
                rc = ccall((:native_set_raman_params, _LIBAMALTHEA_RK45), Cint,
                    (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Csize_t, Float64, Float64, Cint),
                    handle.ptr, omegas, gammas, couplings, Csize_t(length(omegas)),
                    r.dt, raman_density, thg_flag)
                check_ffi(rc, "native_set_raman_params"; ineligible=true)
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
        # Phase E.4 item 5: EnvGrid modal. native.rs's `rhs_modal_pointcalc`
        # now branches on `sim.is_real` (already set generically above, via
        # `native_set_fftw_plans`) between the RealGrid r2c path and a c2c
        # path using `KerrScalarEnv!`/`KerrVectorEnv!`
        # (src/Nonlinear.jl:120-133) — no longer RealGrid-only.
        # Phase E.3: `full=true` (genuine 2-D `(r,θ)` cubature, `hcubature_v`)
        # is now also supported, alongside the original `full=false` (radial
        # only, `pcubature_v`) — see native.rs's `mode_angle_xy`/`rhs_modal`.
        isnothing(f!.Emω_noise) || throw(NativeIneligible("modified shot-noise " *
                          "(Emω_noise) not yet supported for the native modal path"))

        modes = f!.ts.ms
        n_modes = length(modes)
        npol = f!.ts.npol
        npol in (1, 2) || throw(NativeIneligible("native modal path only supports " *
                  "npol ∈ (1,2) (Modes.ToSpace.npol)."))

        all(m -> m isa Amalthea.Capillary.MarcatiliMode, modes) ||
            throw(NativeIneligible("native modal path only supports MarcatiliMode " *
                  "(Phase 5 gate)"))
        all(m -> m.kind in (:HE, :TE, :TM), modes) ||
            throw(NativeIneligible("native modal path only supports kind ∈ " *
                  "(:HE, :TE, :TM) Marcatili modes"))
        # Phase E.2: tapered radius, via the dedicated `ZDepLinopModalTaper`
        # wrapper (`Capillary.make_linop`'s specialized method) — any other
        # Function-valued `a` (not wrapped, e.g. per-mode differing tapers)
        # remains out of scope.
        is_zdep_modal_taper = linop isa Amalthea.Capillary.ZDepLinopModalTaper
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
        invsqrtn = Float64[1.0 / sqrt(Amalthea.Modes.N(m, z=0.0)) for m in modes]
        # `field(m, (r,θ))` (Capillary.jl) factored as `J_order(x)·(ax,ay)`,
        # with `(ax,ay)` a per-kind function of `θ` (`mode_angle_xy` in
        # native.rs) — Phase E.1 generalized the Phase 5 `HE,n=1`-only `J0`
        # scope to any `:HE`/`:TE`/`:TM` mode order (fixed `θ=0`); Phase E.3
        # further generalizes to genuine `θ`-dependence (`full=true`).
        order = Int32[m.kind == :HE ? m.n - 1 : 1 for m in modes]
        kind_code = UInt8[m.kind == :HE ? 0 : (m.kind == :TE ? 1 : 2) for m in modes]
        phi = Float64[m.ϕ for m in modes]

        pol_select = npol == 2 ? UInt8[0, 1] : UInt8[f!.ts.indices == 1 ? 0 : 1]

        towin = f!.grid.towin

        # docs/dev/BACKLOG.md Phase I item 4 (modal): discriminating check —
        # does anything in TransModal's per-mode projection introduce a
        # per-species density dependence at the point kerr_fac is computed?
        # No: `Erω_to_Prω!` calls `Et_to_Pt!(t.Pr, t.Er, t.resp, t.density)`
        # at every quadrature node (NonlinearRHS.jl), the SAME generic
        # `Et_to_Pt!` that already has an `AbstractVector`-density overload
        # dispatching per-species — identical mechanism to mode-averaged/
        # radial. The two modal-specific quantities that also enter the
        # native RHS, `invsqrtn = 1/√N(m,z=0)` (Modes.N — mode-area
        # normalization, a pure function of `unm`/radius/kind, no gas/
        # density argument at all, Capillary.jl:285-288) and `nlfac`
        # (probed from `norm_modal`, a pure function of grid.ω/ω0, no
        # density argument, NonlinearRHS.jl:476-481) are both density-
        # independent. So Kerr's linearity in density·γ3 collapses a
        # per-species mixture to one scalar kerr_fac exactly like the
        # mode-averaged/radial cases — no different physics, just a
        # different RHS shape wrapped around the same `Et_to_Pt!` call.
        _check_density_zindependent(f!.densityfun, flength)
        density = f!.densityfun(0.0)
        is_mixture = density isa AbstractVector

        if is_mixture
            # Kerr-only, mirroring the mode-averaged/radial mixture branches
            # exactly: plasma/Raman mixtures are not a simple linear sum, so
            # they are rejected rather than silently ignored.
            length(f!.resp) == length(density) ||
                throw(NativeIneligible("modal mixture: densityfun(z) returned " *
                      "$(length(density)) species but f!.resp has $(length(f!.resp)) " *
                      "entries — shape mismatch."))
            kerr_fac = 0.0
            for (ρi, respi) in zip(density, f!.resp)
                length(respi) == 1 && _is_plain_kerr_resp(respi[1]) ||
                    throw(NativeIneligible("modal mixture species response $(respi) is " *
                          "not a single plain Kerr response — only Kerr is wired for the " *
                          "native modal mixture path (Phase I item 4); plasma/Raman " *
                          "mixtures are out of scope."))
                γ3i = Amalthea.Nonlinear.kerr_γ3((respi[1],))
                kerr_fac += ρi * Amalthea.PhysData.ε_0 * γ3i
            end
        else
            γ3 = Amalthea.Nonlinear.kerr_γ3(f!.resp)
            # Phase 5 gate is Kerr-only; Phase D.4 (docs/dev/BACKLOG.md) adds Raman
            # (RealGrid, npol=1, either `thg` value since Phase F.1 — same
            # per-node scalar-field scope as Kerr here) alongside it. Any other
            # response type remains ineligible.
            is_kerr_resp_modal(r) = _is_plain_kerr_resp(r)
            for r in f!.resp
                is_kerr_resp_modal(r) || r isa Amalthea.Nonlinear.RamanPolarField ||
                    throw(NativeIneligible("modal: only Kerr and/or Raman (RealGrid) " *
                          "responses are supported by the native path " *
                          "(Phase D.4 gate) — this also rejects Kerr_field_nothg/" *
                          "Kerr_env_thg, see `_is_plain_kerr_resp`."))
                # native.rs's inline Raman ADE solve only ever touches modal
                # polarisation column 0 (`rhs_modal_pointcalc`'s "npol=1 scalar
                # field only" Raman block) — with npol=2 it would silently drop
                # the second (y) column's Raman contribution instead of raising.
                r isa Amalthea.Nonlinear.RamanPolarField && npol != 1 &&
                    throw(NativeIneligible("modal: Raman (RamanPolarField) is only " *
                          "supported natively for npol=1 — native.rs's inline " *
                          "solver does not yet handle a second polarisation column."))
                # native.rs's inline Raman ADE solve operates on real time-domain
                # buffers only (`modal_er`/`modal_pr`) — EnvGrid modal (Phase E.4
                # item 5) uses complex buffers (`modal_er_c`/`modal_pr_c`) with no
                # Raman wiring at all.
                r isa Amalthea.Nonlinear.RamanPolarField && !is_real_grid &&
                    throw(NativeIneligible("modal: Raman (RamanPolarField) is only " *
                          "supported natively for RealGrid — native.rs has no EnvGrid " *
                          "Raman path for modal."))
            end
            γ3 != 0.0 ||
                throw(NativeIneligible("modal: no Kerr response found (γ3=0) — the " *
                      "native modal path requires a Kerr response."))

            kerr_fac = density * Amalthea.PhysData.ε_0 * γ3
        end

        # Extract exactly what `ωwin .* norm!` computes by numerically probing
        # the closure (robust to norm_modal's shock/no-shock branch — avoids
        # re-deriving grid.ω/shock logic, see MATH.md §3.3).
        nlfac = ComplexF64.(f!.grid.ωwin)
        f!.norm!(nlfac)

        lib_path = Cubature.Cubature_jll.libcubature

        rc = ccall((:native_set_modal_params, _LIBAMALTHEA_RK45), Cint,
            (Ptr{Cvoid}, Csize_t, Csize_t, Csize_t, Csize_t,
             Float64, Ptr{Float64}, Ptr{Float64}, Ptr{Int32}, Ptr{UInt8}, Ptr{Float64}, UInt8,
             Ptr{UInt8}, Ptr{Float64}, Float64, Ptr{Float64}, Ptr{Float64},
             Cstring, Float64, Float64, Csize_t),
            handle.ptr, n_time, n_time_over, n_modes, npol,
            a, unm, invsqrtn, order, kind_code, phi, UInt8(f!.full), pol_select,
            towin, kerr_fac, real.(nlfac), imag.(nlfac),
            lib_path, f!.rtol, f!.atol, Csize_t(f!.mfcn))
        check_ffi(rc, "native_set_modal_params")

        # Wire the z-dependent linop (tapered radius) — Phase E.2. Must come
        # after `native_set_modal_params` (reads back its `modal_inv_sqrt_n`
        # z=0 baseline for the `1/√N(m,z) ∝ 1/a(z)` rescale).
        if is_zdep_modal_taper
            w = linop
            n_a = length(w.a_x)
            rc = ccall((:native_set_modal_zdep_params, _LIBAMALTHEA_RK45), Cint,
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
            check_ffi(rc, "native_set_modal_zdep_params"; ineligible=true)
        end

        # Wire Raman if present and eligible — Phase D.4 (docs/dev/BACKLOG.md), both
        # `thg` values since Phase F.1, rotational multi-oscillator and
        # density-dependent τ2 since Phase F.2 (see the mode-averaged wiring
        # above for the full rationale — identical here). Same FFI call as
        # mode-averaged/radial above; `native_set_raman_params` sizes its
        # scratch buffers as `n_time_over` (the `else` branch — modal is
        # neither mode-averaged nor radial, and `rhs_modal_pointcalc` reuses
        # one shared buffer sequentially across quadrature nodes, exactly
        # like mode-averaged's single per-call buffer — see native.rs's
        # `apply_raman_radial` doc for why the same solver instance is safe
        # to reuse this way). Mixtures are Kerr-only by construction
        # (validated above), so this loop is skipped entirely for them.
        for r in (is_mixture ? () : f!.resp)
            if r isa Amalthea.Nonlinear.RamanPolarField
                rr = r.r
                Rs = rr isa Amalthea.Raman.CombinedRamanResponse ?
                    Amalthea.Raman.flatten_sdo_oscillators(rr) : nothing
                isnothing(Rs) && throw(NativeIneligible("Raman response present but not " *
                      "eligible for the native path (needs a CombinedRamanResponse whose " *
                      "components are all RamanRespSingleDampedOscillator and/or " *
                      "RamanRespRotationalNonRigid)."))
                raman_density = f!.densityfun(0.0)
                omegas    = Float64[ri.Ω for ri in Rs]
                gammas    = Float64[1.0 / ri.τ2ρ(raman_density) for ri in Rs]
                couplings = Float64[ri.K for ri in Rs]
                rc = ccall((:native_set_raman_params, _LIBAMALTHEA_RK45), Cint,
                    (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Csize_t, Float64, Float64, Cint),
                    handle.ptr, omegas, gammas, couplings, Csize_t(length(omegas)),
                    r.dt, raman_density, Cint(r.thg))
                check_ffi(rc, "native_set_raman_params"; ineligible=true)
                break  # only one Raman response expected
            end
        end
    end

    # Set parameters if free-space (TransFree) — Phase 6 gate: RealGrid,
    # const_norm_free (z-invariant), scalar Kerr only. Phase D.3 (docs/dev/BACKLOG.md)
    # adds EnvGrid (c2c 3-D FFTW plan, `ComplexFft3d`) — `native_set_free_params`
    # branches on `sim.is_real` (set by `native_set_fftw_plans`, called earlier)
    # exactly like the radial Phase D.1 EnvGrid extension. Reuses the FFTW
    # library handle native_set_fftw_plans already loaded (no new dlopen) —
    # only a new 3-D plan is created. Phase D.5 (docs/dev/BACKLOG.md) drops the
    # z-invariant restriction for a two-point pressure-gradient gas cell —
    # `normfun isa NonlinearRHS.ZDepNormFree` (and `linop isa
    # LinearOps.ZDepLinopFree`, checked by `native_ok` above) — wiring an
    # additional `native_set_free_zdep_params` call. See MATH.md §3.4.
    # Phase I item 6 (docs/dev/BACKLOG.md) adds plasma and/or Raman (`RamanPolarField`)
    # alongside Kerr, RealGrid only — `TransFree`'s `Et_to_Pt!` dispatches each
    # response over `t.idcs` (`CartesianIndices((Ny,Nx))`), so both see a
    # scalar field per `(y,x)` column exactly like `TransRadial`'s Phase
    # D.2/D.4 wiring; `apply_plasma_free`/`apply_raman_free` in native.rs
    # mirror the radial per-column kernels verbatim over `n_y*n_x` columns.
    # EnvGrid free-space (`rhs_free_env`) has no plasma/Raman step (no EnvGrid
    # geometry gets native plasma anywhere in this port, and `RamanPolarEnv`
    # is only wired for mode-averaged EnvGrid), so EnvGrid keeps the original
    # Kerr-only restriction. Plasma/Raman + a z-dependent normfun together are
    # also out of scope (unverified interaction, no existing test coverage) —
    # rejected explicitly rather than silently combined.
    if is_free
        isnothing(f!.Et_noise) || throw(NativeIneligible("modified shot-noise " *
                          "(Et_noise) not yet supported for the native free-space path"))

        n_y = length(f!.xygrid.y)
        n_x = length(f!.xygrid.x)

        towin = f!.grid.towin

        is_zdep_free = f!.normfun isa Amalthea.NonlinearRHS.ZDepNormFree
        if is_zdep_free
            linop isa Amalthea.LinearOps.ZDepLinopFree ||
                throw(NativeIneligible("free-space: a z-dependent normfun (ZDepNormFree) " *
                      "requires a matching z-dependent linop (ZDepLinopFree) — got " *
                      "$(typeof(linop))."))
        else
            _check_density_zindependent(f!.densityfun, flength)
        end
        density = f!.densityfun(0.0)
        is_mixture = density isa AbstractVector

        # docs/dev/BACKLOG.md Phase I item 4 (free-space): discriminating check —
        # `(t::TransFree)(nl,Eωk,z)` calls
        # `Et_to_Pt!(t.Pto, t.Eto, t.resp, t.densityfun(z), t.idcs)`
        # (NonlinearRHS.jl:875-891), which recurses per-(y,x)-column
        # (`Et_to_Pt!(Pt,Et,responses,density,idcs)`, the 5-arg overload)
        # into the SAME generic 4-arg `Et_to_Pt!` that already has an
        # `AbstractVector`-density overload dispatching per-species —
        # identical mechanism to mode-averaged/radial/modal. The only other
        # per-frequency quantity, `M` (folding `normfun`+ωwin, computed once
        # below from `f!.normfun(0.0)`), comes from `const_norm_free`/
        # `norm_free`, both pure functions of the grid's (ω,kx,ky) and the
        # *linear* refractive index `nfun` supplied at construction — no
        # runtime density argument, already baked in before native
        # construction, exactly like radial's normalization array. So
        # Kerr's linearity in density·γ3 collapses a per-species mixture to
        # one scalar kerr_fac here too, for either RealGrid or EnvGrid.
        has_plasma_free = false
        has_raman_free = false
        if is_mixture
            # Kerr-only, mirroring the mode-averaged/radial/modal mixture
            # branches: plasma/Raman mixtures are not a simple linear sum,
            # so they are rejected rather than silently ignored. A
            # z-dependent normfun bakes a single-species `gamma_arr`/`densf`
            # (`norm_free_gradient`) that doesn't generalize to a per-species
            # sum either, so that combination is rejected too — same
            # reasoning as mode-averaged's z-dependent-linop rejection.
            is_zdep_free && throw(NativeIneligible("free-space: gas mixtures are not " *
                  "supported together with a z-dependent (pressure-gradient) normfun."))
            length(f!.resp) == length(density) ||
                throw(NativeIneligible("free-space mixture: densityfun(z) returned " *
                      "$(length(density)) species but f!.resp has $(length(f!.resp)) " *
                      "entries — shape mismatch."))
            kerr_fac = 0.0
            for (ρi, respi) in zip(density, f!.resp)
                length(respi) == 1 && _is_plain_kerr_resp(respi[1]) ||
                    throw(NativeIneligible("free-space mixture species response $(respi) " *
                          "is not a single plain Kerr response — only Kerr is wired for " *
                          "the native free-space mixture path (Phase I item 4); " *
                          "plasma/Raman mixtures are out of scope."))
                γ3i = Amalthea.Nonlinear.kerr_γ3((respi[1],))
                kerr_fac += ρi * Amalthea.PhysData.ε_0 * γ3i
            end
        else
            γ3 = Amalthea.Nonlinear.kerr_γ3(f!.resp)

            if is_real_grid
                for r in f!.resp
                    _is_plain_kerr_resp(r) || r isa Amalthea.Nonlinear.PlasmaCumtrapz ||
                        r isa Amalthea.Nonlinear.RamanPolarField ||
                        throw(NativeIneligible("free-space: only Kerr, plasma, and/or Raman " *
                              "(RealGrid, RamanPolarField) responses are supported by the " *
                              "native path (Phase I item 6 gate) — this also rejects " *
                              "Kerr_field_nothg, see `_is_plain_kerr_resp`."))
                end
            else
                length(f!.resp) == 1 && γ3 != 0.0 && _is_plain_kerr_resp(f!.resp[1]) ||
                    throw(NativeIneligible("free-space (EnvGrid): only single-response " *
                          "Kerr-only is supported by the native path — plasma/Raman are " *
                          "RealGrid-only (Phase I item 6), and a lone non-Kerr response or " *
                          "Kerr_env_thg (see `_is_plain_kerr_resp`) are not wired either."))
            end
            γ3 != 0.0 ||
                throw(NativeIneligible("free-space: no Kerr response found (γ3=0) — the " *
                      "native free-space path requires a Kerr response."))

            has_plasma_free = is_real_grid && any(r -> r isa Amalthea.Nonlinear.PlasmaCumtrapz, f!.resp)
            has_raman_free  = is_real_grid && any(r -> r isa Amalthea.Nonlinear.RamanPolarField, f!.resp)

            if is_zdep_free
                has_plasma_free && throw(NativeIneligible("free-space: plasma combined with a " *
                      "z-dependent normfun is not supported by the native path."))
                has_raman_free && throw(NativeIneligible("free-space: Raman combined with a " *
                      "z-dependent normfun is not supported by the native path."))
            end

            kerr_fac = density * Amalthea.PhysData.ε_0 * γ3
        end

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
        rc = ccall((:native_set_free_params, _LIBAMALTHEA_RK45), Cint,
            (Ptr{Cvoid}, Csize_t, Csize_t, Csize_t, Csize_t, Cuint,
             Ptr{Float64}, Float64,
             Ptr{Float64}, Ptr{Float64}),
            handle.ptr, n_time, n_time_over, n_y, n_x, Cuint(64),
            towin, kerr_fac,
            real.(M), imag.(M))
        check_ffi(rc, "native_set_free_params")

        if is_zdep_free
            w = linop
            eps0_gamma3 = Amalthea.PhysData.ε_0 * γ3
            rc = ccall((:native_set_free_zdep_params, _LIBAMALTHEA_RK45), Cint,
                (Ptr{Cvoid}, Float64, Float64, Float64,
                 Csize_t, Ptr{Float64}, Ptr{Float64}, Ptr{Float64},
                 Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{UInt8},
                 Float64, Float64, Float64, Float64),
                handle.ptr, w.L, w.p0, w.p1,
                Csize_t(length(w.dspl_x)), w.dspl_x, w.dspl_y, w.dspl_d,
                w.gamma, w.omega, ωwin, w.kperp2, UInt8.(w.sidx),
                eps0_gamma3, w.ω0, w.γ0, w.dγ0)
            check_ffi(rc, "native_set_free_zdep_params"; ineligible=true)
        end

        # Wire plasma if present and a Rust ionisation handle is available —
        # same decoupled eligibility as the mode-averaged/radial wiring above.
        # `native_set_plasma_params`/`_adk` size their scratch buffers as
        # `n_time_over*n_y*n_x` when `s.is_free` (already set by
        # `native_set_free_params`, called earlier in this block).
        if has_plasma_free
            for r in f!.resp
                if r isa Amalthea.Nonlinear.PlasmaCumtrapz
                    irf = r.ratefunc
                    if irf isa Amalthea.Ionisation.IonRatePPTAccel &&
                            !isnothing(irf.rust_handle) &&
                            irf.rust_handle.ptr != C_NULL
                        rc = ccall((:native_set_plasma_params, _LIBAMALTHEA_RK45), Cint,
                            (Ptr{Cvoid}, Ptr{Cvoid}, Float64, Float64, Float64, Float64, Float64),
                            handle.ptr, irf.rust_handle.ptr,
                            r.ionpot, Amalthea.PhysData.e_ratio, r.preionfrac, r.δt, density)
                        check_ffi(rc, "native_set_plasma_params"; ineligible=true)
                        push!(gc_roots, irf.rust_handle)
                    elseif irf isa Amalthea.Ionisation.IonRateADK &&
                            !isnothing(irf.rust_handle) &&
                            irf.rust_handle.ptr != C_NULL
                        rc = ccall((:native_set_plasma_params_adk, _LIBAMALTHEA_RK45), Cint,
                            (Ptr{Cvoid}, Ptr{Cvoid}, Float64, Float64, Float64, Float64, Float64),
                            handle.ptr, irf.rust_handle.ptr,
                            r.ionpot, Amalthea.PhysData.e_ratio, r.preionfrac, r.δt, density)
                        check_ffi(rc, "native_set_plasma_params_adk"; ineligible=true)
                        push!(gc_roots, irf.rust_handle)
                    else
                        throw(NativeIneligible("plasma detected but no Rust ionisation handle is " *
                              "available (library missing, or both AMALTHEA_USE_RUST_IONISATION and " *
                              "AMALTHEA_USE_RUST_NATIVE were explicitly disabled), or ratefunc is not " *
                              "an IonRatePPTAccel/IonRateADK — this config is not eligible for the " *
                              "native path."))
                    end
                    break  # only one plasma response expected
                end
            end
        end

        # Wire Raman if present and eligible — same criteria as the
        # mode-averaged/radial wiring above (all-SDO/rotational
        # `CombinedRamanResponse`, density-independent τ2). Same FFI call;
        # `native_set_raman_params` sizes its scratch buffers as
        # `n_time_over*n_y*n_x` when `s.is_free`.
        if has_raman_free
            for r in f!.resp
                if r isa Amalthea.Nonlinear.RamanPolarField
                    rr = r.r
                    Rs = rr isa Amalthea.Raman.CombinedRamanResponse ?
                        Amalthea.Raman.flatten_sdo_oscillators(rr) : nothing
                    isnothing(Rs) && throw(NativeIneligible("Raman response present but not " *
                          "eligible for the native path (needs a CombinedRamanResponse whose " *
                          "components are all RamanRespSingleDampedOscillator and/or " *
                          "RamanRespRotationalNonRigid)."))
                    raman_density = f!.densityfun(0.0)
                    omegas    = Float64[ri.Ω for ri in Rs]
                    gammas    = Float64[1.0 / ri.τ2ρ(raman_density) for ri in Rs]
                    couplings = Float64[ri.K for ri in Rs]
                    rc = ccall((:native_set_raman_params, _LIBAMALTHEA_RK45), Cint,
                        (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Csize_t, Float64, Float64, Cint),
                        handle.ptr, omegas, gammas, couplings, Csize_t(length(omegas)),
                        r.dt, raman_density, Cint(r.thg))
                    check_ffi(rc, "native_set_raman_params"; ineligible=true)
                    break  # only one Raman response expected
                end
            end
        end
    end

    # Copy initial field to Rust
    ccall((:set_field, _LIBAMALTHEA_RK45), Cint,
          (Ptr{Cvoid}, Ptr{ComplexF64}, Csize_t),
          handle.ptr, pointer(y0), Csize_t(n))

    # docs/dev/BACKLOG.md S1 item 1: save accumulated planner wisdom (every plan this
    # construction created, across all the `native_set_*_params` calls
    # above) for next time — mirrors Julia's own load-before/save-after
    # `Utils.loadFFTwisdom`/`saveFFTwisdom` pattern. Best-effort: `mkpath`
    # and the export call are both allowed to fail silently (this must never
    # throw or block returning a working stepper).
    if _native_wisdom_enabled() && !isempty(native_wisdom_path)
        try
            isdir(dirname(native_wisdom_path)) || mkpath(dirname(native_wisdom_path))
            ccall((:native_export_fftw_wisdom, _LIBAMALTHEA_RK45), Cint,
                  (Ptr{Cvoid}, Cstring), handle.ptr, native_wisdom_path)
        catch
        end
    end

    RustNativeStepper(
        copy(y0), copy(y0), float(t), float(t), float(dt), float(dt),
        float(rtol), float(atol), float(safety), float(max_dt), float(min_dt),
        locextrap, false, 0.0, 0.0, handle, gc_roots
    )
end

function step!(s::RustNativeStepper)
    # native_step overwrites s.yn in place; snapshot the pre-call value (the
    # field at the start of this interval, s.t) so interpolate() has a valid
    # start-of-interval field once the step is accepted.
    y_before = copy(s.yn)

    result_ref = Ref(NativeStepResult(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    rc = ccall((:native_step, _LIBAMALTHEA_RK45), Cint,
        (Ptr{Cvoid}, Ptr{ComplexF64}, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Cint, Ptr{NativeStepResult}),
        s._handle.ptr, pointer(s.yn), s.t, s.tn, s.dtn, s.rtol, s.atol, s.safety, s.max_dt, s.min_dt, s.errlast, Cint(s.locextrap), result_ref
    )
    check_ffi(rc, "native_step")

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
    rc = ccall((:native_resync_field, _LIBAMALTHEA_RK45), Cint,
          (Ptr{Cvoid}, Ptr{ComplexF64}, Csize_t),
          s._handle.ptr, pointer(s.yn), Csize_t(length(s.yn)))
    check_ffi(rc, "native_resync_field (post-stepfun resync)")
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
        rc = ccall((:get_ks_stage, _LIBAMALTHEA_RK45), Cint,
              (Ptr{Cvoid}, Csize_t, Ptr{ComplexF64}, Csize_t),
              s._handle.ptr, Csize_t(ii - 1), kbuf, Csize_t(n))
        check_ffi(rc, "get_ks_stage")
        yi .+= kbuf .* b[ii]
    end
    out = @. s.y + s.dt * yi
    rc = ccall((:native_apply_prop, _LIBAMALTHEA_RK45), Cint,
          (Ptr{Cvoid}, Ptr{ComplexF64}, Csize_t, Float64, Float64),
          s._handle.ptr, out, Csize_t(n), s.t, ti)
    check_ffi(rc, "native_apply_prop")
    return out
end

end