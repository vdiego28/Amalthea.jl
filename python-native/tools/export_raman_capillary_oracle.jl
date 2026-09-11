# Independent Julia capillary Raman trajectories. Force the FFT-convolution oracle.
for key in ("NATIVE","STEPPER","DISPERSION","IONISATION","RAMAN")
    ENV["AMALTHEA_USE_RUST_"*key]="0"
end
using Amalthea,DelimitedFiles,TOML
Amalthea.set_fftw_mode(:estimate);Amalthea.set_fftw_threads(1)
root=abspath(ARGS[1]);mkpath(root)
common=(λ0=800e-9,λlims=(200e-9,1700e-9),trange=300e-15,τfwhm=20e-15,
        energy=500e-6,shotnoise=false,plasma=false,saveN=7)
cases=[]
for gas in (:N2,:H2,:D2,:N2O,:CH4,:SF6), envelope in (true,false)
    push!(cases,("$(gas)-$(envelope ? "env" : "real")",gas,(envelope=envelope,)))
end
for (suffix,opts) in ["off"=>(raman=false,),"rotation"=>(vibration=false,),"vibration"=>(rotation=false,),
                       "no-kerr"=>(kerr=false,),"hot"=>(temperature=350.,),"fourth"=>(;),"fine"=>(δt=.04e-15,)]
    push!(cases,("N2-real-$suffix",:N2,merge((envelope=false,),opts)))
end
push!(cases,("N2-env-off",:N2,(envelope=true,raman=false,)))
push!(cases,("N2-env-thg",:N2,(envelope=true,thg=true,)))
push!(cases,("N2-real-no-thg",:N2,(envelope=false,thg=false,)))
push!(cases,("N2-real-plasma",:N2,(envelope=false,plasma=:PPT,)))
push!(cases,("H2-real-plasma",:H2,(envelope=false,plasma=:ADK,)))
for (name,gas,opts) in cases
    kw=merge(common,opts);fifth=!endswith(name,"fourth")
    if get(opts,:plasma,false)==:PPT
        kw=merge(kw,(PPT_options=Dict{Symbol,Any}(:cachedir=>joinpath(root,"cache")),))
    end
    radius=125e-6;flength=.0002;pressure=2.;step=.00001
    E,grid,L,rhs!,FT,output=Amalthea.Interface.prop_capillary_args(radius,flength,gas,pressure;kw...)
    dir=joinpath(root,name);mkpath(dir)
    params=Dict("radius"=>radius,"flength"=>flength,"gas"=>string(gas),"pressure"=>pressure)
    for (k,v) in pairs(kw)
        params[string(k)]=v isa Symbol ? string(v) : v isa Tuple ? collect(v) : v isa AbstractDict ? Dict(string(a)=>b for (a,b) in v) : v
    end
    params["locextrap"]=fifth
    open(joinpath(dir,"parameters.toml"),"w") do io;TOML.print(io,params);end
    nl=similar(E);rhs!(nl,E,0.)
    writedlm(joinpath(dir,"setup.txt"),hcat(real.(E),imag.(E),real.(L),imag.(L),real.(nl),imag.(nl)))
    if get(kw,:raman,true)
        response=Amalthea.Raman.raman_response(grid.to,gas;rotation=get(kw,:rotation,true),
                  vibration=get(kw,:vibration,true),temp=get(kw,:temperature,Amalthea.PhysData.roomtemp))
        h=zeros(length(grid.to));rho=rhs!.densityfun(0.);response(h,rho)
        writedlm(joinpath(dir,"raman.txt"),rho*h)
    end
    window!(y,z,dz,interp)=(y .= FT*((FT\(y.*grid.ωwin)).*grid.twin))
    for fixed in (true,false)
        z,field,_=Amalthea.RK45.solve_precon(rhs!,L,E,0.,step,flength;
            rtol=1e-9,atol=1e-12,min_dt=fixed ? step : 1e-15,max_dt=step,
            output=true,outputN=7,stepfun=window!,locextrap=fifth)
        suffix=fixed ? "fixed" : "adaptive"
        writedlm(joinpath(dir,"$suffix.txt"),hcat(real.(field),imag.(field)))
        writedlm(joinpath(dir,"$suffix-z.txt"),z)
    end
    z,field,_=Amalthea.RK45.solve_precon(rhs!,L,E,0.,step,step;
        rtol=1e-9,atol=1e-12,min_dt=step,max_dt=step,output=true,outputN=7,stepfun=window!,locextrap=fifth)
    writedlm(joinpath(dir,"interval.txt"),hcat(real.(field),imag.(field)));writedlm(joinpath(dir,"interval-z.txt"),z)
    if name in ("N2-env","N2-real","N2-real-plasma","H2-real-plasma","N2-env-thg")
        complete=Amalthea.prop_capillary(radius,flength,gas,pressure;kw...)
        writedlm(joinpath(dir,"entrypoint.txt"),hcat(real.(complete["Eω"]),imag.(complete["Eω"])));writedlm(joinpath(dir,"entrypoint-z.txt"),complete["z"])
    end
    println("Exported Raman capillary case ",name);flush(stdout)
end
