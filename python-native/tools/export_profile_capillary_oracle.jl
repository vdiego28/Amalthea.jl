# Exact position-dependent scalar capillary oracle; no runtime dependency.
for key in ("NATIVE","STEPPER","DISPERSION","IONISATION","RAMAN")
    ENV["AMALTHEA_USE_RUST_"*key]="0"
end
using Amalthea,DelimitedFiles,TOML
Amalthea.set_fftw_mode(:estimate);Amalthea.set_fftw_threads(1)
const PD=Amalthea.PhysData;const IF=Amalthea.Interface;const MO=Amalthea.Modes
root=abspath(ARGS[1]);mkpath(root)
const FLENGTH=.0002;const STEP=.00001
common=(λ0=800e-9,λlims=(200e-9,1700e-9),trange=300e-15,τfwhm=20e-15,
        energy=500e-6,shotnoise=false,plasma=false,raman=false,saveN=7)
function profiles(rname,pname)
    radius=rname=="constant" ? 125e-6 : rname=="flat" ? z->125e-6 :
        rname=="linear" ? z->125e-6*(1-.2z/FLENGTH) :
        z->125e-6*(1+.1sin(π*z/FLENGTH)+.05(z/FLENGTH)^2)
    pressure=pname=="constant" ? 2. : pname=="flat" ? z->2. :
        pname=="rise" ? (2.,4.) : pname=="fall" ? (4.,2.) :
        pname=="multi" ? ([0.,.00007,FLENGTH],[2.,4.,1.]) :
        z->2*(1+.4sin(π*z/FLENGTH)+.1(z/FLENGTH)^2)
    radius,pressure
end
cases=[]
for envelope in (true,false), (name,rname,pname) in
    [("base","constant","constant"),("rise","constant","rise"),("fall","constant","fall"),
     ("multi","constant","multi"),("pressure","constant","callable"),("taper","curved","constant"),
     ("both","curved","callable"),("flat-pressure","constant","flat"),("flat-radius","flat","constant")]
    push!(cases,("Ar-$(envelope ? "env" : "real")-$name",:Ar,rname,pname,(envelope=envelope,)))
end
for gas in (:N2,:H2,:D2,:N2O,:CH4,:SF6), envelope in (true,false)
    push!(cases,("$gas-$(envelope ? "env" : "real")",gas,"constant","rise",(envelope=envelope,raman=true,)))
end
for (name,rname,pname,options) in [
    ("no-raman","constant","rise",(raman=false,)),("plasma","constant","rise",(plasma=:PPT,)),
    ("rotation","constant","rise",(vibration=false,)),("hot","constant","rise",(temperature=350.,)),
    ("fourth","constant","rise",(;)),("fine","constant","rise",(δt=.08e-15,)),
    ("no-thg","constant","rise",(thg=false,)),("no-kerr","constant","rise",(kerr=false,)),
    ("reduced-taper","linear","multi",(model=:reduced,))]
    push!(cases,("N2-real-$name",:N2,rname,pname,merge((envelope=false,raman=true,),options)))
end
push!(cases,("N2-env-thg",:N2,"curved","multi",(envelope=true,thg=true,raman=true,)))
push!(cases,("H2-real-plasma",:H2,"constant","rise",(envelope=false,plasma=:ADK,raman=true,)))
for (name,gas,rname,pname,overrides) in cases
    kw=merge(common,overrides);fifth=!endswith(name,"fourth")
    radius,pressure=profiles(rname,pname)
    if kw.plasma==:PPT
        kw=merge(kw,(PPT_options=Dict{Symbol,Any}(:cachedir=>joinpath(root,"cache")),))
    end
    if pressure isa Function
        density=z->PD.density(gas,pressure(z),get(kw,:temperature,PD.roomtemp))
        gamma=PD.sellmeier_gas(gas)
        core=(ω;z)->sqrt(1+gamma(PD.wlfreq(ω)*1e6)*density(z))
        mode=Amalthea.Capillary.MarcatiliMode(radius,core;model=get(kw,:model,:full),loss=get(kw,:loss,true))
        grid=IF.makegrid(FLENGTH,kw.λ0,kw.λlims,kw.trange,kw.envelope,get(kw,:thg,!kw.envelope),get(kw,:δt,1.))
        response=IF.makeresponse(grid,gas,kw.raman,true,kw.plasma,get(kw,:thg,!kw.envelope),false,
            true,true,get(kw,:PPT_options,Dict{Symbol,Any}()),0.,get(kw,:temperature,PD.roomtemp))
        inputs=IF.makeinputs(mode,kw.λ0,nothing,kw.τfwhm,nothing,Float64[],nothing,kw.energy,:gauss,:linear,nothing)
        L,E,rhs!,FT=IF.setup(grid,mode,density,response,inputs,false,1e-3,Val(false))
    else
        E,grid,L,rhs!,FT,_=IF.prop_capillary_args(radius,FLENGTH,gas,pressure;kw...)
        mode=IF.makemode_s(:HE11,FLENGTH,radius,gas,pressure,get(kw,:temperature,PD.roomtemp),
                           get(kw,:model,:full),get(kw,:loss,true),false)
    end
    dir=joinpath(root,name);mkpath(dir)
    params=Dict("gas"=>string(gas),"radius_profile"=>rname,"pressure_profile"=>pname,"flength"=>FLENGTH,"locextrap"=>fifth)
    for (k,v) in pairs(kw)
        params[string(k)]=v isa Symbol ? string(v) : v isa Tuple ? collect(v) : v isa AbstractDict ? Dict(string(a)=>b for (a,b) in v) : v
    end
    open(joinpath(dir,"parameters.toml"),"w") do io;TOML.print(io,params);end
    writedlm(joinpath(dir,"initial.txt"),hcat(real.(E),imag.(E)))
    for (i,z) in enumerate((0.,.000037,.000137,.00024))
        nl=similar(E);rhs!(nl,E,z)
        op=similar(E);L isa AbstractArray ? (op.=L) : L(op,z)
        beta1=MO.dispersion(mode,1,PD.wlfreq(kw.λ0);z)
        beta0=MO.β(mode,PD.wlfreq(kw.λ0);z)
        neff=ones(ComplexF64,length(E));neff[grid.sidx]=MO.neff.(Ref(mode),grid.ω[grid.sidx];z)
        writedlm(joinpath(dir,"setup-$i.txt"),hcat(real.(nl),imag.(nl),real.(op),imag.(op),real.(neff),imag.(neff)))
        writedlm(joinpath(dir,"material-$i.txt"),[z,rhs!.densityfun(z),rhs!.aeff(z),beta1,beta0])
        if kw.raman
            response=Amalthea.Raman.raman_response(grid.to,gas;rotation=get(kw,:rotation,true),
                vibration=get(kw,:vibration,true),temp=get(kw,:temperature,PD.roomtemp))
            h=zeros(length(grid.to));rho=rhs!.densityfun(z);response(h,rho)
            writedlm(joinpath(dir,"raman-$i.txt"),rho*h)
        end
    end
    window!(y,z,dz,interp)=(y .= FT*((FT\(y.*grid.ωwin)).*grid.twin))
    function write_solution(suffix,z,field)
        writedlm(joinpath(dir,"$suffix.txt"),hcat(real.(field),imag.(field)))
        writedlm(joinpath(dir,"$suffix-z.txt"),z)
    end
    for fixed in (true,false)
        z,field,_=Amalthea.RK45.solve_precon(rhs!,L,E,0.,STEP,FLENGTH;rtol=1e-9,atol=1e-12,
            min_dt=fixed ? STEP : 1e-15,max_dt=STEP,output=true,outputN=7,stepfun=window!,locextrap=fifth)
        write_solution(fixed ? "fixed" : "adaptive",z,field)
    end
    linear_positions=Float64[];linear_samples=Vector{ComplexF64}[]
    function recorded!(out,z)
        L isa AbstractArray ? (out.=L) : L(out,z)
        push!(linear_positions,z);push!(linear_samples,copy(out))
    end
    z,field,_=Amalthea.RK45.solve_precon(rhs!,recorded!,E,0.,STEP,STEP;rtol=1e-9,atol=1e-12,
        min_dt=STEP,max_dt=STEP,output=true,outputN=7,stepfun=window!,locextrap=fifth)
    write_solution("interval",z,field)
    matrix=hcat(linear_samples...)
    writedlm(joinpath(dir,"linear-samples.txt"),hcat(real.(matrix),imag.(matrix)))
    writedlm(joinpath(dir,"linear-positions.txt"),linear_positions)
    if name in ("Ar-env-rise","Ar-real-taper","N2-real-plasma","N2-env-thg")
        complete=Amalthea.prop_capillary(radius,FLENGTH,gas,pressure;kw...)
        write_solution("entrypoint",complete["z"],complete["Eω"])
    end
    println("Exported profile capillary ",name);flush(stdout)
end
# Independent pressure/density spline data, including a zero endpoint.
for (name,gas,Z,P) in [("rise",:Ar,[0.,FLENGTH],[0.,4.]),("multi",:H2,[0.,.00007,FLENGTH],[2.,4.,1.]),
                       ("equal",:N2,[0.,FLENGTH],[2.,2.])]
    core,density=Amalthea.Capillary.gradient(gas,Z,P)
    dir=joinpath(root,"gradient-"*name);mkpath(dir)
    writedlm(joinpath(dir,"spline.txt"),hcat(core.dspl.x,core.dspl.y,core.dspl.D))
    z=collect(range(-.00001,.00024,length=137))
    writedlm(joinpath(dir,"samples.txt"),hcat(z,core.pfun.(z),density.(z)))
end
