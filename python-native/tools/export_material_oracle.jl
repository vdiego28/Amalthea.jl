# Independent setup oracle; the installed Python package never invokes Julia.
ENV["AMALTHEA_USE_RUST_NATIVE"] = "0"
using Amalthea, DelimitedFiles, TOML, CoolProp
const PD = Amalthea.PhysData
root=abspath(ARGS[1]); mkpath(root)
open(joinpath(root,"metadata.toml"),"w") do io
    TOML.print(io,Dict("coolprop"=>CoolProp.get_global_param_string("version"),
                      "N_A"=>PD.N_A,"roomtemp"=>PD.roomtemp,"epsilon0"=>PD.ε_0))
end
for gas in PD.gas
    dir=joinpath(root,string(gas)); mkpath(dir)
    states=[(p,t) for t in (273.15,293.15,330.) for p in (0.,.1,1.,10.,50.)]
    thermo=map(states) do (p,t)
        rho=PD.density(gas,p,t)
        [p,t,rho,PD.pressure(gas,rho,t)]
    end
    writedlm(joinpath(dir,"thermo.txt"),permutedims(hcat(thermo...)))
    wavelengths=[200e-9,300e-9,400e-9,800e-9,1030e-9,1600e-9,3e-6]
    optical=map(wavelengths) do wavelength
        polar=PD.sellmeier_gas(gas)(wavelength*1e6)
        index=PD.ref_index(gas,wavelength,1.7,305.)
        [wavelength,real(polar),imag(polar),real(index),imag(index)]
    end
    writedlm(joinpath(dir,"optical.txt"),permutedims(hcat(optical...)))
    if gas != :Air
        gamma=PD.γ3_gas(gas)
        writedlm(joinpath(dir,"gamma3.txt"),[real(gamma) imag(gamma)])
    end
end
println("Exported all gas material fixtures to ",root)
