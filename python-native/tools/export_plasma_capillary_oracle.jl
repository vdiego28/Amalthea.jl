# Independent Julia plasma setup and trajectories, with every Rust toggle off.
ENV["AMALTHEA_USE_RUST_NATIVE"]="0"
ENV["AMALTHEA_USE_RUST_STEPPER"]="0"
ENV["AMALTHEA_USE_RUST_DISPERSION"]="0"
ENV["AMALTHEA_USE_RUST_IONISATION"]="0"
using Amalthea, DelimitedFiles, TOML
Amalthea.set_fftw_mode(:estimate); Amalthea.set_fftw_threads(1)
root=abspath(ARGS[1]);mkpath(root)
common=(λ0=800e-9,λlims=(200e-9,1700e-9),trange=300e-15,τfwhm=20e-15,
        energy=500e-6,shotnoise=false,envelope=false,raman=false,plasma=:ADK,saveN=7)
cases=["adk"=>(;),"ppt"=>(plasma=:PPT,),"no-plasma"=>(plasma=false,),
       "no-kerr"=>(kerr=false,),"no-thg"=>(thg=false,),"preion"=>(preionfrac=.01,),
       "fine"=>(δt=.04e-15,),"fourth"=>(;)]
for (name,overrides) in cases
    kw=merge(common,overrides); gas=:Ar;fifth=name!="fourth"
    if name=="ppt";kw=merge(kw,(PPT_options=Dict{Symbol,Any}(:cachedir=>joinpath(root,"cache")),));end
    radius=125e-6;length=.0002;pressure=2.;step=.00001
    E,grid,L,rhs!,FT,output=Amalthea.Interface.prop_capillary_args(radius,length,gas,pressure;kw...)
    dir=joinpath(root,name);mkpath(dir)
    params=Dict("radius"=>radius,"flength"=>length,"gas"=>string(gas),"pressure"=>pressure)
    for (k,v) in pairs(kw)
        params[string(k)]=v isa Symbol ? string(v) : v isa Tuple ? collect(v) : v isa AbstractDict ? Dict(string(a)=>b for (a,b) in v) : v
    end
    params["locextrap"]=fifth
    open(joinpath(dir,"parameters.toml"),"w") do io;TOML.print(io,params);end
    nl=similar(E);rhs!(nl,E,0.)
    writedlm(joinpath(dir,"setup.txt"),hcat(real.(E),imag.(E),real.(L),imag.(L),real.(nl),imag.(nl)))
    # Separate physical-field controls establish each cumtrapz intermediate.
    if kw.plasma!==false
        physical=@. 3e10*exp(-(grid.to/25e-15)^2)*cos(2pi*Amalthea.PhysData.c/800e-9*grid.to)
        rate=kw.plasma==:ADK ? Amalthea.Ionisation.IonRateADK(gas) : Amalthea.Ionisation.IonRatePPTCached(gas,grid.referenceλ;kw.PPT_options...)
        response=Amalthea.Nonlinear.PlasmaCumtrapz(grid.to,physical,rate,Amalthea.PhysData.ionisation_potential(gas);preionfrac=get(kw,:preionfrac,0.))
        Amalthea.Nonlinear.PlasmaScalar!(response,physical)
        writedlm(joinpath(dir,"plasma.txt"),hcat(physical,response.rate,response.fraction,response.J,response.P))
    end
    window!(y,z,dz,interp)=(y .= FT*((FT\(y.*grid.ωwin)).*grid.twin))
    for fixed in (true,false)
        z,field,_=Amalthea.RK45.solve_precon(rhs!,L,E,0.,step,length;
            rtol=1e-9,atol=1e-12,min_dt=fixed ? step : 1e-15,max_dt=step,
            output=true,outputN=7,stepfun=window!,locextrap=fifth)
        suffix=fixed ? "fixed" : "adaptive"
        writedlm(joinpath(dir,"$suffix.txt"),hcat(real.(field),imag.(field)))
        writedlm(joinpath(dir,"$suffix-z.txt"),z)
    end
    z,field,_=Amalthea.RK45.solve_precon(rhs!,L,E,0.,step,step;
        rtol=1e-9,atol=1e-12,min_dt=step,max_dt=step,output=true,outputN=7,stepfun=window!,locextrap=fifth)
    writedlm(joinpath(dir,"interval.txt"),hcat(real.(field),imag.(field)))
    writedlm(joinpath(dir,"interval-z.txt"),z)
    if name in ("adk","ppt")
        complete=Amalthea.prop_capillary(radius,length,gas,pressure;kw...)
        writedlm(joinpath(dir,"entrypoint.txt"),hcat(real.(complete["Eω"]),imag.(complete["Eω"])))
        writedlm(joinpath(dir,"entrypoint-z.txt"),complete["z"])
    end
    if name=="ppt"
        accepted=Ref(0)
        counted_window! = (y,z,dz,interp) -> begin
            accepted[]+=1;window!(y,z,dz,interp)
        end
        z,field,attempts=Amalthea.RK45.solve_precon(rhs!,L,E,0.,length,length;
            rtol=1e-12,atol=1e-14,min_dt=1e-15,max_dt=length,
            output=true,outputN=7,stepfun=counted_window!,locextrap=fifth)
        writedlm(joinpath(dir,"rejections.txt"),hcat(real.(field),imag.(field)))
        writedlm(joinpath(dir,"rejections-z.txt"),z)
        writedlm(joinpath(dir,"rejections-counts.txt"),[accepted[],attempts-accepted[]])
    end
    println("Exported plasma case ",name);flush(stdout)
end
