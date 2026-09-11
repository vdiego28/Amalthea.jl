# Independent Julia setup and trajectories, never a Python runtime dependency.
ENV["AMALTHEA_USE_RUST_NATIVE"]="0"
ENV["AMALTHEA_USE_RUST_STEPPER"]="0"
ENV["AMALTHEA_USE_RUST_DISPERSION"]="0"
using Amalthea, DelimitedFiles, TOML
Amalthea.set_fftw_mode(:estimate)
Amalthea.set_fftw_threads(1)
root=abspath(ARGS[1]); mkpath(root)
common=(λ0=800e-9,λlims=(400e-9,1700e-9),trange=300e-15,τfwhm=20e-15,
        energy=10e-6,shotnoise=false,envelope=true,raman=false,plasma=false,saveN=7)
cases=["base"=>(;),"reduced"=>(model=:reduced,),"no-loss"=>(loss=false,),
       "no-kerr"=>(kerr=false,),"he12"=>(modes=:HE12,),"fine"=>(δt=.25e-15,),
       "molecular-kerr"=>(;)]
for (name,overrides) in cases
    kw=merge(common,overrides); gas=name=="molecular-kerr" ? :N2 : :Ar
    radius=125e-6; length=.02; pressure=2.
    E,grid,L,rhs!,FT,output=Amalthea.Interface.prop_capillary_args(radius,length,gas,pressure;kw...)
    dir=joinpath(root,name); mkpath(dir)
    params=Dict("radius"=>radius,"flength"=>length,"gas"=>string(gas),"pressure"=>pressure)
    for (k,v) in pairs(kw)
        params[string(k)]=v isa Symbol ? string(v) : v isa Tuple ? collect(v) : v
    end
    open(joinpath(dir,"parameters.toml"),"w") do io; TOML.print(io,params); end
    beta=similar(grid.ω); Amalthea.NonlinearRHS.norm_βfun(rhs!.norm!)(beta,0.)
    nl=similar(E); rhs!(nl,E,0.)
    writedlm(joinpath(dir,"setup.txt"),hcat(real.(E),imag.(E),real.(L),imag.(L),real.(nl),imag.(nl),beta))
    writedlm(joinpath(dir,"material.txt"),[rhs!.densityfun(0.),rhs!.aeff(0.),Amalthea.Fields.energyfuncs(grid)[1](FT\E)])
    window!(y,z,dz,interp)=(y .= FT*((FT\(y.*grid.ωwin)).*grid.twin))
    for fixed in (true,false)
        z,field,_=Amalthea.RK45.solve_precon(rhs!,L,E,0.,.001,length;
            rtol=1e-9,atol=1e-12,min_dt=fixed ? .001 : 1e-15,max_dt=.001,
            output=true,outputN=7,stepfun=window!)
        suffix=fixed ? "fixed" : "adaptive"
        writedlm(joinpath(dir,"$suffix.txt"),hcat(real.(field),imag.(field)))
        writedlm(joinpath(dir,"$suffix-z.txt"),z)
    end
    z,field,_=Amalthea.RK45.solve_precon(rhs!,L,E,0.,.001,.001;
        rtol=1e-9,atol=1e-12,min_dt=.001,max_dt=.001,output=true,outputN=7,stepfun=window!)
    writedlm(joinpath(dir,"interval.txt"),hcat(real.(field),imag.(field)))
    writedlm(joinpath(dir,"interval-z.txt"),z)
    if name=="base"
        complete=Amalthea.prop_capillary(radius,length,gas,pressure;kw...)
        writedlm(joinpath(dir,"entrypoint.txt"),hcat(real.(complete["Eω"]),imag.(complete["Eω"])))
        writedlm(joinpath(dir,"entrypoint-z.txt"),complete["z"])
    end
end
println("Exported independent capillary envelope fixtures to ",root)
