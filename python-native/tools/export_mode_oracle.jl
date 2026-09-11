# Development-only independent setup oracle; never used by the Python runtime.
ENV["AMALTHEA_USE_RUST_NATIVE"] = "0"
ENV["AMALTHEA_USE_RUST_DISPERSION"] = "0"
using Amalthea, DelimitedFiles, TOML, Cubature, SpecialFunctions, FiniteDifferences
const PD=Amalthea.PhysData
const CP=Amalthea.Capillary
const MO=Amalthea.Modes
root=abspath(ARGS[1]); mkpath(root)
open(joinpath(root,"metadata.toml"),"w") do io
    TOML.print(io,Dict("hbar"=>PD.ħ,"electron"=>PD.electron,"epsilon0"=>PD.ε_0))
end
wavelengths=[100e-9,173e-9,231e-9,405e-9,793e-9,1037e-9,1.9e-6,3.7e-6,8.3e-6,19e-6]
silica=PD.ref_index_fun(:SiO2)
writedlm(joinpath(root,"silica.txt"),hcat(wavelengths,real.(silica.(wavelengths)),imag.(silica.(wavelengths))))
writedlm(joinpath(root,"silica-data.txt"),PD.data_glass(:SiO2))
for (kind,n,m) in [(:HE,1,1),(:HE,2,3),(:HE,8,8),(:TE,0,1),(:TM,0,4)]
    for model in (:full,:reduced), loss in (false,true), profile in (false,true)
        name=join((kind,n,m,model,loss,profile),"_"); dir=joinpath(root,name); mkpath(dir)
        radius=profile ? z->125e-6*(1+.17*sin(3z)) : 125e-6
        core=(ω;z)->PD.ref_index(:Ar,PD.wlfreq(ω),profile ? 1.3+.7*z^2 : 1.3,305.)
        clad=profile ? (ω;z)->1.45+.01*z+(.002+.003*z)*im : (ω;z)->silica(PD.wlfreq(ω))
        mode=CP.MarcatiliMode(radius,n,m,kind,.37,core,clad;model,loss)
        integral,err=hquadrature(r->r*besselj(n-1,mode.unm*r)^4,0,1;reltol=5e-14)
        err <= 5e-14*integral || error("refined effective area failed")
        area_factor=2π/4*besselj(kind==:HE ? n : 2,mode.unm)^4/integral
        rows=map((0.,.137,.71)) do z
            [z,CP.radius(mode,z),mode.unm,MO.N(mode;z),MO.Aeff(mode;z),CP.radius(mode,z)^2*area_factor,err/integral]
        end
        writedlm(joinpath(dir,"setup.txt"),permutedims(hcat(rows...)))
        optics=Vector{Float64}[]; fields=Vector{Float64}[]
        for z in (0.,.137,.71)
            for λ in wavelengths
                ω=PD.wlfreq(λ); v=MO.neff(mode,ω;z)
                push!(optics,[z,ω,real(v),imag(v),MO.β(mode,ω;z),MO.α(mode,ω;z)])
            end
            for r in (0.,.19,.53,.91,1.1), θ in (0.,.29,1.7,4.1)
                x=r*CP.radius(mode,z); f=MO.field(mode,(x,θ);z)
                push!(fields,[z,x,θ,f...])
            end
        end
        writedlm(joinpath(dir,"optics.txt"),permutedims(hcat(optics...)))
        writedlm(joinpath(dir,"fields.txt"),permutedims(hcat(fields...)))
        ω0=PD.wlfreq(793e-9)
        writedlm(joinpath(dir,"dispersion.txt"),[MO.dispersion(mode,q,ω0;z=.137) for q in 0:7])
        for q in 1:7
            fdm=Amalthea.Maths.FDMs[q]; f=x->MO.β(mode,x*ω0;z=.137)
            step,acc=FiniteDifferences.estimate_step(fdm,f,1.)
            nodes=1 .+ step.*fdm.grid
            writedlm(joinpath(dir,"derivative$(q)-samples.txt"),hcat(nodes,f.(nodes)))
            writedlm(joinpath(dir,"derivative$(q)-step.txt"),[step,acc,ω0])
        end
    end
end
# Additional core models and the explicit full-model cutoff clamp.
extra=Vector{Float64}[]
for (gi,gas) in enumerate((:He,:N2,:vacuum)), loss in (false,true)
    mode=gas==:vacuum ? CP.MarcatiliMode(80e-6;loss) : CP.MarcatiliMode(80e-6,gas,2.1;loss)
    for λ in (300e-9,800e-9,1600e-9)
        ω=PD.wlfreq(λ); v=MO.neff(mode,ω)
        push!(extra,[gi,loss,ω,real(v),imag(v)])
    end
end
writedlm(joinpath(root,"extra-cores.txt"),permutedims(hcat(extra...)))
cutoff=Vector{Float64}[]
for loss in (false,true)
    mode=CP.MarcatiliMode(100e-9,1,1,:HE,0.,(ω;z)->1.,(ω;z)->20im;loss)
    for λ in (400e-9,800e-9,1600e-9)
        ω=PD.wlfreq(λ); v=MO.neff(mode,ω)
        push!(cutoff,[loss,ω,real(v),imag(v)])
    end
end
writedlm(joinpath(root,"cutoff.txt"),permutedims(hcat(cutoff...)))
# Exact stencil and adaptive-step evidence on a smooth analytic scalar function.
for q in 1:7
    fdm=Amalthea.Maths.FDMs[q]; dir=joinpath(root,"derivative$q"); mkpath(dir)
    writedlm(joinpath(dir,"stencil.txt"),hcat(fdm.grid,fdm.coefs))
    f=x->exp(x)
    step,acc=FiniteDifferences.estimate_step(fdm,f,1.)
    writedlm(joinpath(dir,"estimate.txt"),[step,acc,fdm(f,1.)])
    writedlm(joinpath(dir,"samples.txt"),hcat(1 .+ step.*fdm.grid,f.(1 .+ step.*fdm.grid)))
    bound=fdm.bound_estimator
    bstep=first(FiniteDifferences.estimate_step(bound,f,1.))
    writedlm(joinpath(dir,"bound.txt"),[bstep,bound.∇f_magnitude_mult,bound.f_error_mult])
    writedlm(joinpath(dir,"bound-samples.txt"),hcat(1 .+ bstep.*bound.grid,f.(1 .+ bstep.*bound.grid)))
end
println("Exported independent Marcatili fixtures to ",root)
