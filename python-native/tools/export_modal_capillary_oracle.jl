# Independent Julia modal setup, point kernels, fixed-node dense output and solves.
for key in ("NATIVE","STEPPER","DISPERSION","IONISATION","RAMAN")
    ENV["AMALTHEA_USE_RUST_"*key]="0"
end
using Amalthea, DelimitedFiles, TOML, LinearAlgebra, QuadGK
Amalthea.set_fftw_mode(:estimate);Amalthea.set_fftw_threads(1)
const IF=Amalthea.Interface;const MO=Amalthea.Modes;const PD=Amalthea.PhysData
const NR=Amalthea.NonlinearRHS
const LENGTH=1e-5;const STEP=2.5e-6;const RTOL=1e-9
root=abspath(ARGS[1]);mkpath(root)
writec(path,x)=writedlm(path,hcat(real.(vec(x)),imag.(vec(x))))
serialize(x::Symbol)=string(x)
serialize(x::Tuple)=collect(serialize.(x))
serialize(x::NamedTuple)=Dict(string(k)=>serialize(v) for (k,v) in pairs(x) if v!==nothing)
serialize(x)=x

struct EquivalentMode{M} <: MO.AbstractMode
    mode::M
end
MO.neff(m::EquivalentMode,w;z=0.)=MO.neff(m.mode,w;z)
MO.field(m::EquivalentMode,x;z=0.)=MO.field(m.mode,x;z)
MO.dimlimits(m::EquivalentMode;z=0.)=MO.dimlimits(m.mode;z)
MO.N(m::EquivalentMode;z=0.)=MO.N(m.mode;z)
MO.modeinfo(m::EquivalentMode)=MO.modeinfo(m.mode)

base=(modes=2,polarisation=:linear,pulse_variant="two",gas=:Ar,pressure=2.,
      gradient=false,custom=false,average=false,raman=false,kerr=true,plasma=false,
      preionfrac=0.,model=:full,loss=true,modal_components=nothing,modal_full=nothing,
      locextrap=true,radial_integral_rtol=1e-8,default_controls=false,energy_scale=1.)
cases=[]
for envelope in (true,false)
    for (name,options) in [
        ("radial",(;)),("radial-linear",(kerr=false,)),
        ("radial-fourth",(locextrap=false,)),("radial-reduced",(model=:reduced,loss=false)),
        ("radial-thg",(thg=envelope,)),
        ("circular",(polarisation=:circular,pulse_variant="one")),
        ("elliptic",(polarisation=-.6,pulse_variant="one")),
        ("full",(modes=(:HE21,:TE01,:TM01),pulse_variant="full")),
        ("full-linear",(modes=(:HE21,:TE01,:TM01),pulse_variant="full",kerr=false)),
        ("full-x",(modes=(:HE21,:HE22),modal_components=:x,pulse_variant="full")),
        ("raman",(gas=:N2,raman=true)),("raman-off",(gas=:N2,raman=false)),
        ("gradient",(gas=:N2,raman=true,gradient=true)),
        ("mixture",(gas=(:N2,:H2),pressure=(2.,1.),raman=true)),
        ("custom",(custom=true,modal_components=:y,modal_full=false)),
        ("default",(default_controls=true,)),
        ("average-TE",(modes=:TE01,average=true,pulse_variant="one")),
        ("average-TM",(modes=:TM01,average=true,pulse_variant="one")),
        ("average-HE21",(modes=:HE21,average=true,pulse_variant="one"))]
        tag=envelope ? "env" : "real"
        push!(cases,("$name-$tag",merge(base,(envelope=envelope,thg=!envelope),options)))
    end
end
for (name,options) in [
    ("vector-ADK",(plasma=:ADK,)),("vector-PPT",(plasma=:PPT,)),
    ("vector-preion",(plasma=:ADK,preionfrac=.002)),
    ("vector-gradient",(plasma=:ADK,gradient=true)),
    ("vector-plasma-only",(plasma=:ADK,kerr=false,thg=false)),
    ("full-plasma",(modes=(:HE21,:TE01,:TM01),pulse_variant="full",polarisation=:linear,plasma=:ADK)),
    ("mixture-plasma",(gas=(:N2,:H2),pressure=(2.,1.),raman=true,plasma=true,polarisation=:linear,pulse_variant="two",energy_scale=.2))]
    push!(cases,("$name-real",merge(base,(envelope=false,thg=true,polarisation=:circular,pulse_variant="one"),options)))
end

function makepulses(cfg)
    common=(λ0=800e-9,τfwhm=20e-15)
    if cfg.pulse_variant=="one"
        return IF.Pulses.GaussPulse(;common...,energy=500e-6*cfg.energy_scale,polarisation=cfg.polarisation)
    elseif cfg.pulse_variant=="two"
        return [IF.Pulses.GaussPulse(;common...,energy=400e-6*cfg.energy_scale,mode=:HE11),
                IF.Pulses.SechPulse(λ0=790e-9,τfwhm=17e-15,energy=100e-6*cfg.energy_scale,mode=:HE12,ϕ=[.2,12e-15])]
    end
    n=length(cfg.modes)
    [IF.Pulses.GaussPulse(;common...,energy=500e-6*cfg.energy_scale/n,mode=mode,ϕ=[.1i,(i-1)*5e-15]) for (i,mode) in enumerate(cfg.modes)]
end

for (name,cfg) in cases
    pointsonly=get(ENV,"AMALTHEA_MODAL_POINTS_ONLY","")=="1"
    pointsonly && cfg.plasma===false && continue
    selected=get(ENV,"AMALTHEA_MODAL_CASE","")
    isempty(selected) || name in split(selected,',') || continue
    println("Starting ",name);flush(stdout)
    dir=joinpath(root,name);mkpath(dir)
    radius=cfg.gradient ? z->125e-6*(1+.08sin(π*z/LENGTH)+.03*(z/LENGTH)^2) : 125e-6
    pressure=cfg.gradient ? (2.,4.) : cfg.pressure
    gas=cfg.gas
    if cfg.gradient
        coren,density=Capillary.gradient(gas,LENGTH,pressure...)
    elseif gas isa Tuple
        values=[PD.density(g,p) for (g,p) in zip(gas,pressure)]
        density=z->values
    else
        rho=PD.density(gas,pressure);density=z->rho
    end
    grid=IF.makegrid(LENGTH,800e-9,(200e-9,1700e-9),100e-15,cfg.envelope,cfg.thg,1.)
    both=IF.needpol(cfg.polarisation)
    specs=cfg.modes isa Int ? Tuple(Dict(:kind=>:HE,:n=>1,:m=>m) for m=1:cfg.modes) :
        cfg.modes isa Symbol ? (cfg.modes,) : cfg.modes
    modes=MO.AbstractMode[]
    for specification in specs
        options=IF.parse_mode(specification)
        made=cfg.gradient ? IF.makemodes_pol(both,radius,coren;model=cfg.model,loss=cfg.loss,options...) :
            IF.makemodes_pol(both,radius,gas,pressure;model=cfg.model,loss=cfg.loss,options...)
        append!(modes,made isa AbstractVector ? made : [made])
    end
    cfg.custom && (modes=EquivalentMode.(modes))
    radial=all(m->MO.modeinfo(m)[:kind]==:HE && MO.modeinfo(m)[:n]==1,modes)
    full=isnothing(cfg.modal_full) ? !radial : cfg.modal_full
    components=isnothing(cfg.modal_components) ? (both || !radial ? :xy : :y) : cfg.modal_components
    pol=components==:xy && !cfg.average
    ppt=Dict{Symbol,Any}(:cachedir=>joinpath(root,"cache"))
    if gas isa Tuple
        responses=Tuple(IF.makeresponse(grid,g,cfg.raman,cfg.kerr,cfg.plasma,cfg.thg,pol,true,true,ppt,cfg.preionfrac,PD.roomtemp) for g in gas)
    else
        responses=IF.makeresponse(grid,gas,cfg.raman,cfg.kerr,cfg.plasma,cfg.thg,pol,true,true,ppt,cfg.preionfrac,PD.roomtemp)
    end
    variable=cfg.gradient || cfg.custom
    collection=cfg.average ? modes[1] : modes
    inputs=IF.makeinputs(collection,800e-9,makepulses(cfg))
    if cfg.average
        linop,E,rhs!,FT=IF.setup(grid,collection,density,responses,inputs,false,cfg.radial_integral_rtol,Val(!variable))
    else
        linop=variable ? LinearOps.make_linop(grid,modes,800e-9) : LinearOps.make_const_linop(grid,modes,800e-9)
        E,rhs!,FT=Amalthea.setup(grid,density,responses,inputs,modes,components;
                                full,rtol=cfg.radial_integral_rtol,mfcn=2_000_000)
    end
    params=serialize(cfg);params["field_shape"]=collect(size(E));params["components"]=string(components);params["full"]=full
    params["checked_quadrature"]=true
    open(joinpath(dir,"parameters.toml"),"w") do io;TOML.print(io,params);end
    writec(joinpath(dir,"initial.txt"),E)
    for (zi,z) in enumerate((0.,.37LENGTH,1.2LENGTH))
        op=similar(E);linop isa AbstractArray ? (op.=linop) : linop(op,z)
        writec(joinpath(dir,"linear-$zi.txt"),op)
        writedlm(joinpath(dir,"density-$zi.txt"),vcat(z,density(z)))
        nl=similar(E);rhs!(nl,E,z);writec(joinpath(dir,"rhs-$zi.txt"),nl)
        if !cfg.average
            writedlm(joinpath(dir,"quadrature-$zi.txt"),[norm(rhs!.err),cfg.radial_integral_rtol*norm(nl),rhs!.ncalls])
            @assert norm(rhs!.err)<=cfg.radial_integral_rtol*norm(nl)
            NR.reset!(rhs!,E,z)
            a=MO.dimlimits(modes[1];z)[3][1]
            for (pi,(r,theta)) in enumerate(((.13,.2),(.51,.73),(.87,2.1)))
                NR.Erω_to_Prω!(rhs!,(r*a,theta))
                writec(joinpath(dir,"point-time-$zi-$pi.txt"),rhs!.Er)
                writec(joinpath(dir,"point-polarization-$zi-$pi.txt"),rhs!.Pr)
                writec(joinpath(dir,"point-spectrum-$zi-$pi.txt"),rhs!.Prω)
                writec(joinpath(dir,"point-projection-$zi-$pi.txt"),rhs!.Prω*transpose(rhs!.ts.Ems))
                species=gas isa Tuple ? responses : (responses,)
                for (si,items) in enumerate(species), response in items
                    response isa Amalthea.Nonlinear.PlasmaCumtrapz || continue
                    writedlm(joinpath(dir,"plasma-rate-$zi-$pi-$si.txt"),hcat(response.rate,response.fraction))
                    writedlm(joinpath(dir,"plasma-current-$zi-$pi-$si.txt"),response.J)
                    writedlm(joinpath(dir,"plasma-polarization-$zi-$pi-$si.txt"),response.P)
                end
            end
        end
    end
    if pointsonly;println("Exported modal plasma points ",name);flush(stdout);continue;end
    accepted=Float64[]
    function checked_rhs!(out,field,z)
        rhs!(out,field,z)
        if !cfg.average
            @assert norm(rhs!.err)<=cfg.radial_integral_rtol*norm(out)
        end
    end
    function window!(y,z,dz,interp)
        push!(accepted,z);y.=FT*((FT\(y.*grid.ωwin)).*grid.twin)
    end
    for fixed in (true,false)
        empty!(accepted);suffix=fixed ? "fixed" : "adaptive"
        dz=cfg.default_controls && !fixed ? 1e-4 : STEP
        maxdt=cfg.default_controls && !fixed ? LENGTH/2 : STEP
        rtol=cfg.default_controls && !fixed ? 1e-6 : RTOL
        atol=cfg.default_controls && !fixed ? 1e-10 : 1e-12
        z,field,attempts=RK45.solve_precon(checked_rhs!,linop,E,0.,dz,LENGTH;rtol,atol,min_dt=fixed ? STEP : 1e-15,
                max_dt=maxdt,output=true,outputN=7,stepfun=window!,locextrap=cfg.locextrap)
        writec(joinpath(dir,"$suffix.txt"),field);writedlm(joinpath(dir,"$suffix-z.txt"),z)
        writedlm(joinpath(dir,"$suffix-accepted.txt"),accepted)
        open(joinpath(dir,"$suffix-controls.toml"),"w") do io
            TOML.print(io,Dict("dt"=>dz,"max_dt"=>maxdt,"rtol"=>rtol,"atol"=>atol,"accepted"=>length(accepted),"rejected"=>attempts-length(accepted)))
        end
    end
    if name in ("radial-thg-env","mixture-plasma-real")
        controlresponses=gas isa Tuple ? Tuple(IF.makeresponse(grid,g,cfg.raman,cfg.kerr,false,cfg.thg,pol,
                                        true,true,ppt,cfg.preionfrac,PD.roomtemp) for g in gas) :
            IF.makeresponse(grid,gas,cfg.raman,cfg.kerr,cfg.plasma,false,pol,true,true,ppt,cfg.preionfrac,PD.roomtemp)
        _,controlrhs,controlft=Amalthea.setup(grid,density,controlresponses,inputs,modes,components;
                                             full,rtol=cfg.radial_integral_rtol,mfcn=2_000_000)
        control!(out,field,z)=begin
            controlrhs(out,field,z)
            @assert norm(controlrhs.err)<=cfg.radial_integral_rtol*norm(out)
        end
        _,controlfield,_=RK45.solve_precon(control!,linop,E,0.,STEP,LENGTH;
                rtol=RTOL,atol=1e-12,min_dt=1e-15,max_dt=STEP,output=true,outputN=7,
                stepfun=window!,locextrap=cfg.locextrap)
        writec(joinpath(dir,name=="radial-thg-env" ? "thg-control.txt" : "plasma-control.txt"),controlfield)
    end
    # Same-node mathematics gate independent of the two adaptive spatial rules.
    gx,gw=QuadGK.gauss(12)
    rule=full && !cfg.average ? [(gx[i],gx[j],gw[i]*gw[j]) for i=1:12 for j=1:12] : [(gx[i],gw[i]) for i=1:12]
    rulematrix=reduce(hcat,collect.(rule));writedlm(joinpath(dir,"rule.txt"),rulematrix')
    function fixednodes!(out,E,z)
        if cfg.average;rhs!(out,E,z);return;end
        NR.reset!(rhs!,E,z);ndim=full ? 2 : 1
        dl=MO.dimlimits(modes[1];z);lower=collect(dl[2])[1:ndim];upper=collect(dl[3])[1:ndim]
        half=(upper-lower)/2;mid=(upper+lower)/2
        points=rulematrix[1:ndim,:].*half.+mid
        pointvalues=zeros(length(E)*2,size(points,2));NR.pointcalc!(pointvalues,points,rhs!)
        integrated=pointvalues*rulematrix[end,:]*prod(half)
        out.=reshape(reinterpret(ComplexF64,integrated),size(E))
    end
    positions=Float64[];samples=Vector{ComplexF64}[]
    function recorded!(out,z)
        linop isa AbstractArray ? (out.=linop) : linop(out,z)
        push!(positions,z);push!(samples,vec(copy(out)))
    end
    z,field,_=RK45.solve_precon(fixednodes!,recorded!,E,0.,STEP,STEP;rtol=RTOL,atol=1e-12,
            min_dt=STEP,max_dt=STEP,output=true,outputN=7,stepfun=window!,locextrap=cfg.locextrap)
    writec(joinpath(dir,"interval.txt"),field);writedlm(joinpath(dir,"interval-z.txt"),z)
    writedlm(joinpath(dir,"linear-positions.txt"),positions);writec(joinpath(dir,"linear-samples.txt"),hcat(samples...))
    println("Exported modal ",name);flush(stdout)
end
