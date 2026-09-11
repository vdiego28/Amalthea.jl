# Independent low-level Julia species/response mixtures; development only.
for key in ("NATIVE","STEPPER","DISPERSION","IONISATION","RAMAN")
    ENV["AMALTHEA_USE_RUST_"*key]="0"
end
using Amalthea,DelimitedFiles,TOML
Amalthea.set_fftw_mode(:estimate);Amalthea.set_fftw_threads(1)
const PD=Amalthea.PhysData;const IF=Amalthea.Interface;const MO=Amalthea.Modes
const FLENGTH=.0002;const STEP=parse(Float64,get(ENV,"AMALTHEA_MIXTURE_STEP",".00001"))
const RTOL=parse(Float64,get(ENV,"AMALTHEA_MIXTURE_RTOL","1e-9"))
root=abspath(ARGS[1]);mkpath(root)
function pressure_profile(name)
    name=="zero" && return 0.
    name=="one" && return 1.
    name=="two" && return 2.
    name=="half-Ar" && return PD.pressure(:Ar,PD.density(:Ar,2.)/2)
    name=="rise" && return (2.,4.)
    name=="multi" && return ([0.,.00007,FLENGTH],[1.,2.,1.5])
    name=="callable" && return z->1.0 + .3sin(π*z/FLENGTH)+.1*(z/FLENGTH)^2
    error("Unknown pressure profile")
end
radius_profile(name)=name=="constant" ? 125e-6 : z->125e-6*(1+.1sin(π*z/FLENGTH)+.05*(z/FLENGTH)^2)
function density_profile(gas,p,T)
    p isa Number && return let rho=PD.density(gas,p,T); z->rho;end
    p isa Function && return z->PD.density(gas,p(z),T)
    p[1] isa Number ? Capillary.gradient(gas,FLENGTH,p...;T)[2] : Capillary.gradient(gas,p...;T)[2]
end
function serializable(v)
    v===nothing && return "default"
    v isa Symbol && return string(v)
    v isa NamedTuple && return Dict(string(k)=>serializable(x) for (k,x) in pairs(v))
    v isa AbstractDict && return Dict(string(k)=>serializable(x) for (k,x) in v)
    v isa Tuple && return [serializable(x) for x in v]
    v
end
cases=[]
for envelope in (true,false)
    tag=envelope ? "env" : "real"
    for (name,gases,ps,radius,opts,overrides) in [
        ("ArNe",(:Ar,:Ne),("two","one"),"constant",(;),nothing),
        ("ArNe-no-Ne",(:Ar,:Ne),("two","zero"),"constant",(;),nothing),
        ("ArNe-no-kerr",(:Ar,:Ne),("two","one"),"constant",(kerr=false,),nothing),
        ("Ar-single",(:Ar,),("two",),"constant",(;),nothing),
        ("Ar-split",(:Ar,:Ar),("half-Ar","half-Ar"),"constant",(;),nothing),
        ("ArNe-fourth",(:Ar,:Ne),("two","one"),"constant",(;),nothing),
        ("ArNe-reduced",(:Ar,:Ne),("two","one"),"constant",(model=:reduced,loss=false),nothing),
        ("ArNe-thg",(:Ar,:Ne),("two","one"),"constant",(thg=envelope,),nothing),
        ("ArNe-gradient",(:Ar,:Ne),("rise","multi"),"constant",(;),nothing),
        ("ArNe-callable",(:Ar,:Ne),("two","callable"),"curved",(;),nothing),
        ("N2H2",(:N2,:H2),("two","one"),"constant",(;),nothing),
        ("N2H2-no-raman",(:N2,:H2),("two","one"),"constant",(raman=false,),nothing),
        ("N2H2-no-kerr",(:N2,:H2),("two","one"),"constant",(kerr=false,),nothing),
        ("N2H2-components",(:N2,:H2),("two","one"),"constant",(;),((rotation=false,),(vibration=false,))),
        ("N2H2-hot",(:N2,:H2),("two","one"),"constant",(temperature=350.,),nothing),
        ("N2H2-gradient",(:N2,:H2),("rise","multi"),"curved",(;),nothing),
        ("molecular-three",(:N2O,:CH4,:SF6),("one","one","one"),"constant",(;),nothing)]
        push!(cases,("$name-$tag",gases,ps,radius,merge((envelope=envelope,),opts),overrides))
    end
end
for (name,ps,radius,opts,overrides) in [
    ("plasma",("two","one"),"constant",(plasma=true,),nothing),
    ("plasma-preion",("two","one"),"constant",(plasma=true,),((preionfrac=.002,),(preionfrac=.005,))),
    ("plasma-N2",("two","one"),"constant",(plasma=false,),((plasma=:PPT,),(;))),
    ("plasma-H2",("two","one"),"constant",(plasma=false,),((;),(plasma=:ADK,))),
    ("plasma-gradient",("rise","multi"),"curved",(plasma=true,),nothing),
    ("plasma-nothg",("two","one"),"constant",(plasma=true,thg=false),nothing),
    ("fine",("two","one"),"constant",(δt=.08e-15,),nothing)]
    push!(cases,("N2H2-$name-real",(:N2,:H2),ps,radius,merge((envelope=false,),opts),overrides))
end
common=(λ0=800e-9,λlims=(200e-9,1700e-9),trange=300e-15,τfwhm=20e-15,energy=500e-6,
        shotnoise=false,plasma=false,raman=nothing,kerr=true,rotation=true,vibration=true,
        preionfrac=0.,temperature=PD.roomtemp,model=:full,loss=true)
for (name,gases,pnames,rname,options,overrides) in cases
    selected=get(ENV,"AMALTHEA_MIXTURE_CASE","")
    isempty(selected) || name==selected || continue
    kw=merge(common,options);thg=get(kw,:thg,!kw.envelope);fifth=!occursin("fourth",name)
    ps=pressure_profile.(pnames);radius=radius_profile(rname)
    grid=IF.makegrid(FLENGTH,kw.λ0,kw.λlims,kw.trange,kw.envelope,thg,get(kw,:δt,1.))
    density_funs=Tuple(density_profile(g,p,kw.temperature) for (g,p) in zip(gases,ps))
    density=z->[f(z) for f in density_funs]
    variable=radius isa Function || any(p->!(p isa Number),ps)
    if variable
        corefun=PD.ref_index_fun(gases)
        mode=Capillary.MarcatiliMode(radius,(ω;z)->corefun(PD.wlfreq(ω),density(z));model=kw.model,loss=kw.loss)
    else
        mode=Capillary.MarcatiliMode(radius,gases,ps;T=kw.temperature,model=kw.model,loss=kw.loss)
    end
    species=[merge(kw,isnothing(overrides) ? (;) : overrides[i]) for i in eachindex(gases)]
    ppt=Dict{Symbol,Any}(:cachedir=>joinpath(root,"cache"))
    responses=Tuple(IF.makeresponse(grid,g,s.raman,s.kerr,s.plasma,thg,false,s.rotation,s.vibration,
                                    ppt,s.preionfrac,s.temperature) for (g,s) in zip(gases,species))
    inputs=IF.makeinputs(mode,kw.λ0,nothing,kw.τfwhm,nothing,Float64[],nothing,kw.energy,:gauss,:linear,nothing)
    L,E,rhs!,FT=IF.setup(grid,mode,density,responses,inputs,false,1e-3,Val(!variable))
    dir=joinpath(root,name);mkpath(dir)
    params=serializable(kw);delete!(params,"raman")
    if kw.raman!==nothing;params["raman"]=kw.raman;end
    params["gases"]=collect(string.(gases));params["pressure_profiles"]=collect(pnames);params["radius_profile"]=rname
    params["locextrap"]=fifth;params["thg"]=thg;params["saveN"]=7
    if !isnothing(overrides);params["species_options"]=serializable(overrides);end
    open(joinpath(dir,"parameters.toml"),"w") do io;TOML.print(io,params);end
    writedlm(joinpath(dir,"initial.txt"),hcat(real.(E),imag.(E)))
    for (i,z) in enumerate((0.,.000037,.000137,.00024))
        out=similar(E);rhs!(out,E,z);op=similar(E);L isa AbstractArray ? (op.=L) : L(op,z)
        neff=ones(ComplexF64,length(E));neff[grid.sidx]=MO.neff.(Ref(mode),grid.ω[grid.sidx];z)
        writedlm(joinpath(dir,"setup-$i.txt"),hcat(real.(out),imag.(out),real.(op),imag.(op),real.(neff),imag.(neff)))
        writedlm(joinpath(dir,"material-$i.txt"),[z,MO.Aeff(mode;z),MO.dispersion(mode,1,PD.wlfreq(kw.λ0);z),MO.β(mode,PD.wlfreq(kw.λ0);z),density(z)...])
        for (j,(gas,s)) in enumerate(zip(gases,species))
            raman=s.raman===nothing ? gas in (:N2,:H2,:D2,:N2O,:CH4,:SF6) : s.raman
            if raman
                rr=Raman.raman_response(grid.to,gas;rotation=s.rotation,vibration=s.vibration,temp=s.temperature)
                h=zeros(length(grid.to));rho=density(z)[j];rr(h,rho)
                writedlm(joinpath(dir,"raman-$i-$j.txt"),rho*h)
            end
        end
    end
    accepted=Float64[]
    function window!(y,z,dz,interp)
        push!(accepted,z)
        if get(ENV,"AMALTHEA_MIXTURE_FILTER","on")=="on"
            y .= FT*((FT\(y.*grid.ωwin)).*grid.twin)
        end
    end
    function save_solution(suffix,z,field)
        writedlm(joinpath(dir,"$suffix.txt"),hcat(real.(field),imag.(field)))
        writedlm(joinpath(dir,"$suffix-z.txt"),z)
    end
    for fixed in (true,false)
        empty!(accepted)
        z,field,attempts=RK45.solve_precon(rhs!,L,E,0.,STEP,FLENGTH;rtol=RTOL,atol=1e-12,min_dt=fixed ? STEP : 1e-15,
                                  max_dt=STEP,output=true,outputN=7,stepfun=window!,locextrap=fifth)
        suffix=fixed ? "fixed" : "adaptive"
        save_solution(suffix,z,field)
        writedlm(joinpath(dir,"$suffix-accepted.txt"),accepted)
        open(joinpath(dir,"$suffix-controls.toml"),"w") do io
            TOML.print(io,Dict("dt"=>STEP,"rtol"=>RTOL,"accepted"=>length(accepted),"rejected"=>attempts-length(accepted)))
        end
    end
    if name=="N2H2-plasma-gradient-real" && STEP==1e-5 && RTOL==1e-9 && get(ENV,"AMALTHEA_MIXTURE_FILTER","on")=="on"
        for (suffix,dz,rtol,atol,max_dt) in (("adaptive-refined",2.5e-6,RTOL,1e-12,2.5e-6),
                ("adaptive-fine",1e-6,RTOL,1e-12,1e-6),("default",1e-4,1e-6,1e-10,FLENGTH/2))
            empty!(accepted)
            z,field,attempts=RK45.solve_precon(rhs!,L,E,0.,dz,FLENGTH;rtol,atol,min_dt=1e-15,
                max_dt,output=true,outputN=7,stepfun=window!,locextrap=fifth)
            save_solution(suffix,z,field)
            writedlm(joinpath(dir,"$suffix-accepted.txt"),accepted)
            open(joinpath(dir,"$suffix-controls.toml"),"w") do io
                TOML.print(io,Dict("dt"=>dz,"max_dt"=>max_dt,"rtol"=>rtol,"atol"=>atol,
                                  "accepted"=>length(accepted),"rejected"=>attempts-length(accepted)))
            end
        end
    end
    positions=Float64[];samples=Vector{ComplexF64}[]
    function recorded!(out,z)
        L isa AbstractArray ? (out.=L) : L(out,z)
        push!(positions,z);push!(samples,copy(out))
    end
    z,field,_=RK45.solve_precon(rhs!,recorded!,E,0.,STEP,STEP;rtol=RTOL,atol=1e-12,min_dt=STEP,
                              max_dt=STEP,output=true,outputN=7,stepfun=window!,locextrap=fifth)
    save_solution("interval",z,field)
    matrix=hcat(samples...)
    writedlm(joinpath(dir,"linear-positions.txt"),positions)
    writedlm(joinpath(dir,"linear-samples.txt"),hcat(real.(matrix),imag.(matrix)))
    println("Exported mixture ",name);flush(stdout)
end
