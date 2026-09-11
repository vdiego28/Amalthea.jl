ENV["AMALTHEA_USE_RUST_NATIVE"]="0"
ENV["AMALTHEA_USE_RUST_IONISATION"]="0"
using Amalthea,DelimitedFiles,TOML
import HCubature: hquadrature
I=Amalthea.Ionisation;P=Amalthea.PhysData
root=abspath(ARGS[1]);mkpath(root)
xs=[0.,.01,.2,1.,3.,10.,25.99,26.,26.01,27.,40.]
for m in 0:2
    values=I.φ.(m,xs)
    refined=map(xs) do x
        bx=BigFloat(x)
        bx==0 && return 0.
        first(hquadrature(y->(bx^2-y^2)^abs(m)*exp(y^2-bx^2),big"0",bx;rtol=big"1e-25")) |> Float64
    end
    writedlm(joinpath(root,"phi-$m.txt"),hcat(xs,values,refined))
end
cases=Pair{String,Any}[]
for material in (:He,:HeJ,:HeB,:Ne,:Ar,:ArB,:Kr,:Xe,:N2,:O2)
    push!(cases,string(material)=>(material,800e-9,(;)))
end
for (name,kw) in ["base1030"=>(;),"no-stark"=>(stark_shift=false,),"no-dipole"=>(dipole_corr=false,),
                  "average"=>(cycle_average=true,),"integral"=>(sum_integral=true,),
                  "no-msum"=>(msum=false,),"single"=>(occupancy=1,),"cnl"=>(Cnl=1.4,),
                  "tight"=>(sum_tol=1e-9,),"occupancy"=>(occupancy=m->m==0 ? 2 : 1,)]
    push!(cases,name=>(:Ar,1030e-9,kw))
end
for (name,(material,lambda,kw)) in cases
    rate=I.IonRatePPT(material,lambda;kw...)
    maxfield=1.5*I.barrier_suppression(rate.ionpot,rate.Z)
    fields=collect(range(1e8,maxfield,length=37));fields=vcat(-reverse(fields),0.,fields)
    path=joinpath(root,name);mkpath(path)
    writedlm(joinpath(path,"rates.txt"),hcat(fields,[iszero(x) ? NaN : rate(x) for x in fields]))
    params=Dict("ionpot"=>rate.ionpot,"lambda0"=>lambda,"Z"=>rate.Z,"l"=>rate.l,
                "delta_alpha"=>rate.Δα,"alpha_ion"=>rate.α_ion_au*P.au_polarisability,
                "alpha_ion_au"=>rate.α_ion_au,"omega0_au"=>rate.ω0_au,
                "sum_tol"=>rate.sum_tol,"cycle_average"=>rate.cycle_average,
                "sum_integral"=>rate.sum_integral,"msum"=>rate.msum,
                "occupancy"=>name=="occupancy" ? "callable" : rate.occupancy,
                "zero_evaluated"=>false,"material"=>string(material),"stark_shift"=>get(kw,:stark_shift,true),
                "dipole_corr"=>get(kw,:dipole_corr,true))
    !ismissing(rate.Cnl) && (params["Cnl"]=rate.Cnl)
    open(joinpath(path,"parameters.toml"),"w") do io;TOML.print(io,params);end
end
# Numeric setup exercises higher orbital order and explicit corrections.
rate=I.IonRatePPT(P.ionisation_potential(:Ar),800e-9,1.,2;Δα=2e-41,α_ion=1e-41)
fields=collect(range(1e9,4e10,length=31))
writedlm(joinpath(root,"numeric-l2.txt"),hcat(fields,rate.(fields)))
E,rates=I.makePPTcache(P.ionisation_potential(:Ar),800e-9,1.,1;N=1024,Δα=0.0,α_ion=0.0)
table=I.IonRatePPTAccel(E,rates)
writedlm(joinpath(root,"table.txt"),hcat(E,rates))
queries=vcat(0.,prevfloat(table.Emin),table.Emin,collect(range(table.Emin,table.Emax,length=91)),table.Emax*1.1)
writedlm(joinpath(root,"table-queries.txt"),hcat(queries,table.(queries)))
nonuniform_E=[1.,2.,4.,5.,8.,9.]
nonuniform_rate=[1.,2.,0.,8.,32.,55.]
nonuniform=I.IonRatePPTAccel(nonuniform_E,nonuniform_rate)
queries=collect(range(.5,10.,length=47))
writedlm(joinpath(root,"nonuniform-table.txt"),hcat(nonuniform_E,nonuniform_rate))
writedlm(joinpath(root,"nonuniform-queries.txt"),hcat(queries,nonuniform.(queries)))
println("Exported independent PPT setup, rates, phi and table to ",root)
