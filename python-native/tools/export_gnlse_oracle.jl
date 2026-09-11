# Development only: runtime Python never imports or launches Julia.
ENV["AMALTHEA_USE_RUST_NATIVE"] = "0"
ENV["AMALTHEA_USE_RUST_STEPPER"] = "0"
using Amalthea, DelimitedFiles, TOML, Cubature
Amalthea.set_fftw_mode(:estimate)
Amalthea.set_fftw_threads(1)
root = abspath(ARGS[1]); mkpath(root)
common = (λ0=800e-9, λlims=(550e-9, 1700e-9), trange=400e-15,
          τfwhm=20e-15, power=2000., shotnoise=false, saveN=7,
          loss=1.5, ϕ=[.1, 2e-15, 10e-30])
cases = ["base" => (;), "fine" => (δt=.5e-15,),
         "sech" => (pulseshape=:sech,), "energy" => (power=nothing, energy=50e-12,),
         "no-kerr" => (;), "no-raman" => (raman=false,),
         "no-shock" => (shock=false,), "no-loss" => (loss=0.,), "frame" => (;),
         "sio2" => (ramanmodel=:SiO2,),
         "sio2-fine" => (ramanmodel=:SiO2, δt=.5e-15,),
         "sio2-long" => (ramanmodel=:SiO2, trange=2e-12,)]
for (name, overrides) in cases
    kw = merge(common, overrides)
    gamma = name == "no-kerr" ? 0. : .01
    betas = name == "frame" ? [1e5, 1e-9, -20e-27, 50e-42] : [0., 0., -20e-27, 50e-42]
    E, grid, L, rhs!, FT, output = Amalthea.Interface.prop_gnlse_args(gamma, .03, betas; kw...)
    dir = joinpath(root, name); mkpath(dir)
    params = Dict("gamma"=>gamma, "flength"=>.03, "betas"=>betas)
    for (k,v) in pairs(kw)
        isnothing(v) && continue
        params[string(k)] = v isa Symbol ? string(v) : v isa Tuple ? collect(v) : v
    end
    open(joinpath(dir, "parameters.toml"), "w") do io
        TOML.print(io, params)
    end
    if startswith(name, "sio2")
        # Preserve unmodified production response; refine only the normalization
        # for tight same-input checks, independently of the Python formula.
        original = rhs!.resp[2].r
        unit = Amalthea.Raman.raman_response(grid.to, :SiO2)
        default_norm = 1/unit.scale
        raw(t) = unit(t)/unit.scale
        refined_norm, error = hquadrature(raw, 0., 1e-9; reltol=1e-13)
        h = similar(rhs!.resp[2].ht); original(h, 1.)
        writedlm(joinpath(dir, "raman-default.txt"), h)
        open(joinpath(dir, "normalization.toml"), "w") do io
            TOML.print(io, Dict("default"=>default_norm, "refined"=>refined_norm,
                               "relative_error_estimate"=>error/refined_norm,
                               "omega"=>unit.ωi, "amplitude"=>unit.Ai,
                               "gaussian"=>unit.Γi, "lorentzian"=>unit.γi))
        end
        refined_response!(h, rho) = (original(h, rho); h .*= default_norm/refined_norm)
        response = Amalthea.Nonlinear.RamanPolarEnv(grid.to, refined_response!)
        rhs! = Amalthea.NonlinearRHS.TransModeAvg(grid, rhs!.FT,
            (rhs!.resp[1], response), rhs!.densityfun, rhs!.norm!, rhs!.aeff)
    end
    nl = similar(E); rhs!(nl, E, 0.)
    writedlm(joinpath(dir, "setup.txt"), hcat(real.(E), imag.(E), real.(L), imag.(L), real.(nl), imag.(nl)))
    if name == "base" || startswith(name, "sio2")
        h = similar(rhs!.resp[2].ht); rhs!.resp[2].r(h, 1.)
        writedlm(joinpath(dir, "raman.txt"), h)
        println("EPS0=", Amalthea.PhysData.ε_0, " energy=", Amalthea.Fields.energyfuncs(grid)[1](FT\E))
    end
    window!(y,z,dz,interp) = (y .= FT * ((FT\(y .* grid.ωwin)) .* grid.twin))
    for fixed in (false, true)
        z, field, _ = Amalthea.RK45.solve_precon(rhs!, L, E, 0., .001, .03;
            rtol=1e-9, atol=1e-12, min_dt=fixed ? .001 : 1e-15, max_dt=.001,
            output=true, outputN=7, stepfun=window!)
        suffix = fixed ? "fixed" : "adaptive"
        writedlm(joinpath(dir, "$suffix.txt"), hcat(real.(field), imag.(field)))
        writedlm(joinpath(dir, "$suffix-z.txt"), z)
    end
    if name == "base" || startswith(name, "sio2")
        complete = Amalthea.prop_gnlse(gamma, .03, betas; kw...)
        writedlm(joinpath(dir, "entrypoint.txt"), hcat(real.(complete["Eω"]), imag.(complete["Eω"])))
        writedlm(joinpath(dir, "entrypoint-z.txt"), complete["z"])
    end
    # Interior of a single interval, independently verifies the nonlinear stage math.
    z, field, _ = Amalthea.RK45.solve_precon(rhs!, L, E, 0., .001, .001;
        min_dt=.001, max_dt=.001, output=true, outputN=3, stepfun=window!)
    writedlm(joinpath(dir, "interval.txt"), hcat(real.(field), imag.(field)))
end
println("Exported GNLSE setup and trajectory oracles to ", root)
