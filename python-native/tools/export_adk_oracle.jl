ENV["AMALTHEA_USE_RUST_NATIVE"]="0"
ENV["AMALTHEA_USE_RUST_IONISATION"]="0"
using Amalthea, DelimitedFiles, TOML
root=abspath(ARGS[1]);mkpath(root)
P=Amalthea.PhysData;I=Amalthea.Ionisation
open(joinpath(root,"constants.toml"),"w") do io
    TOML.print(io,Dict("electron"=>P.electron,"mass"=>P.m_e,"hbar"=>P.ħ,
                       "energy"=>P.au_energy,"field"=>P.au_Efield))
end
for material in (:He,:HeJ,:HeB,:Ne,:Ar,:ArB,:Kr,:Xe,:H,:N2,:H2,:O2,:CH4,:N2O,:SF6,:D2)
    for threshold in (true,false), average in (true,false), occupancy in (1,2)
        rate=I.IonRateADK(material;threshold,cycle_average=average,occupancy)
        name="$(material)-$(threshold)-$(average)-$(occupancy)"
        path=joinpath(root,name);mkpath(path)
        reference_threshold=I.ADK_threshold(rate.ionpot)
        fields=unique(sort(vcat([0.,1e3,prevfloat(reference_threshold),reference_threshold,
                                nextfloat(reference_threshold)],10 .^ range(8.,12.,length=41))))
        fields=vcat(-reverse(fields[2:end]),fields)
        writedlm(joinpath(path,"rates.txt"),hcat(fields,rate.(fields)))
        params=Dict("material"=>string(material),"occupancy"=>occupancy,
                    "threshold"=>threshold,"cycle_average"=>average,
                    "ionpot"=>rate.ionpot,"nstar"=>rate.nstar,"cn_sq"=>rate.cn_sq,
                    "omega_p"=>rate.ω_p,"omega_t_prefac"=>rate.ω_t_prefac,
                    "thr"=>rate.thr,"avfac"=>rate.avfac,
                    "potential_eV"=>P.ionisation_potential(material;unit=:eV),
                    "potential_atomic"=>P.ionisation_potential(material;unit=:atomic))
        open(joinpath(path,"parameters.toml"),"w") do io;TOML.print(io,params);end
    end
end
println("Exported independent ADK setup and rates to ",root)
