# Independent molecular Raman setup, no Rust offloads or propagation.
ENV["AMALTHEA_USE_RUST_NATIVE"]="0"
ENV["AMALTHEA_USE_RUST_RAMAN"]="0"
using Amalthea,DelimitedFiles,TOML
root=abspath(ARGS[1]);mkpath(root)
time=collect(range(-512e-15,step=1e-15,length=1024))
writedlm(joinpath(root,"input-time.txt"),time)
p=Amalthea.PhysData
open(joinpath(root,"constants.toml"),"w") do io
    TOML.print(io,Dict("C"=>p.c,"EPS0"=>p.ε_0,"HBAR"=>p.ħ,"K_B"=>p.k_B,"M_U"=>p.m_u,"AMG"=>p.amg))
end
aliases=Dict(:Δα=>"delta_alpha",:dαdQ=>"dalpha_dQ",:Ωv=>"omega_v",:μ=>"mu",
             :Bρr=>"B_r",:Aρr=>"A_r",:Bρv=>"B_v",:Aρv=>"A_v",:τ2r=>"tau_r",:τ2v=>"tau_v",:Cv=>"C_v")
cases=[]
for gas in (:N2,:H2,:D2,:N2O,:CH4,:SF6)
    for (suffix,opts) in ["base"=>(;),"rotation"=>(vibration=false,),"vibration"=>(rotation=false,),"hot"=>(temp=350.,)]
        push!(cases,("$(gas)-$suffix",gas,opts))
    end
end
push!(cases,("H2-range",:H2,(minJ=2,maxJ=12,temp=200.,)))
push!(cases,("N2O-short",:N2O,(minJ=0,maxJ=8,temp=350.,)))
push!(cases,("O2-empty",:O2,(rotation=false,vibration=false,)))
for (name,gas,extra) in cases
    kw=merge((rotation=true,vibration=true,minJ=0,maxJ=50,temp=p.roomtemp),extra)
    response=Amalthea.Raman.raman_response(time,gas;kw...)
    dir=joinpath(root,name);mkpath(dir)
    settings=Dict("gas"=>string(gas),"rotation"=>kw.rotation,"vibration"=>kw.vibration,"minJ"=>kw.minJ,"maxJ"=>kw.maxJ,"temperature"=>kw.temp)
    open(joinpath(dir,"settings.toml"),"w") do io;TOML.print(io,settings);end
    params=Dict(get(aliases,k,string(k))=>(v isa Symbol ? string(v) : v) for (k,v) in pairs(p.raman_parameters(gas)) if k!=:kind)
    open(joinpath(dir,"parameters.toml"),"w") do io;TOML.print(io,params);end
    writedlm(joinpath(dir,"grid.txt"),hcat(response.t,response.w))
    flat=Amalthea.Raman.flatten_sdo_oscillators(response)
    rows=zeros(length(flat),5)
    densities=[0.,.1p.amg,p.amg,20p.amg]
    for (j,r) in enumerate(flat)
        rows[j,:]=[r.Ω,r.K,Amalthea.Raman.hrdamp(r,densities[2]),Amalthea.Raman.hrdamp(r,densities[3]),Amalthea.Raman.hrdamp(r,densities[4])]
    end
    writedlm(joinpath(dir,"oscillators.txt"),rows)
    for (i,rho) in enumerate(densities)
        h=zeros(length(time));response(h,rho)
        writedlm(joinpath(dir,"response-$i.txt"),h)
    end
    writedlm(joinpath(dir,"densities.txt"),densities)
end
errors=Dict{String,String}()
for (name,gas,kw) in [("O2-rotation",:O2,(vibration=false,)),("O2-vibration",:O2,(rotation=false,)),
                       ("H2-truncated",:H2,(minJ=20,maxJ=50))]
    try
        Amalthea.Raman.raman_response(time,gas;kw...)
        errors[name]="unexpected success"
    catch error
        errors[name]=sprint(showerror,error)
    end
end
open(joinpath(root,"errors.toml"),"w") do io;TOML.print(io,errors);end
println("Exported ",length(cases)," independent molecular Raman setups")
