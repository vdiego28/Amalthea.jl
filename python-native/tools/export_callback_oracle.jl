# Independent synthetic Cartesian modes with exact profiles and Kerr responses.
for key in ("NATIVE","STEPPER","DISPERSION","IONISATION","RAMAN")
    ENV["AMALTHEA_USE_RUST_"*key]="0"
end
using Amalthea, DelimitedFiles, TOML, LinearAlgebra, QuadGK
Amalthea.set_fftw_mode(:estimate);Amalthea.set_fftw_threads(1)
const IF=Amalthea.Interface;const MO=Amalthea.Modes;const PD=Amalthea.PhysData
const NR=Amalthea.NonlinearRHS
const LENGTH=1e-5;const STEP=2.5e-6
root=abspath(ARGS[1]);mkpath(root)
writec(path,x)=writedlm(path,hcat(real.(vec(x)),imag.(vec(x))))
radius(z)=125e-6*(1+.08sin(π*z/LENGTH)+.03*(z/LENGTH)^2)
struct CartesianCallbackMode <: MO.AbstractMode
    index::Int
    varying::Bool
end
position(m,z)=m.varying ? z : 0.
MO.dimlimits(m::CartesianCallbackMode;z=0.)=begin
    a=radius(position(m,z));(:cartesian,(-a,-a/2),(a,a/2))
end
function MO.field(m::CartesianCallbackMode,xy;z=0.)
    a=radius(position(m,z));x,y=xy./(a,a/2)
    value=(1-x^2)*(1-y^2)*(m.index==1 ? 1. : x)
    value.*[1.,.3]
end
MO.N(m::CartesianCallbackMode;z=0.)=begin
    a=radius(position(m,z));integral=(m.index==1 ? 16/15 : 16/105)*(16/15)*1.09
    .5sqrt(PD.ε_0/PD.μ_0)*(a^2/2)*integral
end
MO.neff(m::CartesianCallbackMode,w;z=0.)=1.0001+(m.index-1)*2e-6+1e-7im+position(m,z)/LENGTH*1e-5

for envelope in (true,false), variant in ("varying","linear","constant")
    name="cartesian-$variant-"*(envelope ? "env" : "real")
    println("Starting ",name);flush(stdout);dir=joinpath(root,name);mkpath(dir)
    varying=variant!="constant";nonlinear=variant!="linear"
    modes=[CartesianCallbackMode(i,varying) for i=1:2]
    density=z->[PD.density(:Ar,2. + (varying ? z/LENGTH : 0.)),
                PD.density(:Ne,1. + (varying ? .4sin(π*z/LENGTH) : 0.))]
    grid=IF.makegrid(LENGTH,800e-9,(200e-9,1700e-9),100e-15,envelope,!envelope,1.)
    pulses=[IF.Pulses.GaussPulse(λ0=800e-9,τfwhm=20e-15,energy=300e-6),
            IF.Pulses.SechPulse(λ0=790e-9,τfwhm=17e-15,energy=100e-6,ϕ=[.2,12e-15])]
    inputs=Tuple((mode=i,fields=(pulse.field,)) for (i,pulse) in enumerate(pulses))
    responses=Tuple(IF.makeresponse(grid,g,false,nonlinear,false,!envelope,true,true,true,
                                    Dict{Symbol,Any}(),0.,PD.roomtemp) for g in (:Ar,:Ne))
    linop=LinearOps.make_linop(grid,modes,800e-9)
    E,rhs!,FT=Amalthea.setup(grid,density,responses,inputs,modes,:xy;full=true,rtol=1e-9,mfcn=2_000_000)
    open(joinpath(dir,"parameters.toml"),"w") do io
        TOML.print(io,Dict("field_shape"=>collect(size(E)),"envelope"=>envelope,"varying"=>varying,"kerr"=>nonlinear))
    end
    writec(joinpath(dir,"initial.txt"),E)
    for (zi,z) in enumerate((0.,.37LENGTH,1.2LENGTH))
        op=similar(E);linop(op,z);writec(joinpath(dir,"linear-$zi.txt"),op)
        writedlm(joinpath(dir,"normalization-$zi.txt"),[MO.N(m;z) for m in modes])
        writedlm(joinpath(dir,"density-$zi.txt"),density(z))
        out=similar(E);rhs!(out,E,z);writec(joinpath(dir,"rhs-coarse-$zi.txt"),out)
        @assert norm(rhs!.err)<=1e-9*norm(out)
        errors=[norm(rhs!.err),1e-9*norm(out)]
        rhs!.rtol=1e-11
        rhs!(out,E,z);writec(joinpath(dir,"rhs-$zi.txt"),out)
        @assert norm(rhs!.err)<=1e-11*norm(out)
        append!(errors,[norm(rhs!.err),1e-11*norm(out)])
        writedlm(joinpath(dir,"quadrature-$zi.txt"),errors)
        rhs!.rtol=1e-9
    end
    if get(ENV,"AMALTHEA_CALLBACK_SETUP_ONLY","")=="1"
        println("Exported callback setup ",name);flush(stdout);continue
    end
    function checked!(out,field,z)
        rhs!(out,field,z);@assert norm(rhs!.err)<=1e-9*norm(out)
    end
    window!(y,z,dz,interp)=(y.=FT*((FT\(y.*grid.ωwin)).*grid.twin))
    for fixed in (true,false)
        suffix=fixed ? "fixed" : "adaptive"
        z,field,_=RK45.solve_precon(checked!,linop,E,0.,STEP,LENGTH;rtol=1e-9,atol=1e-12,
             min_dt=fixed ? STEP : 1e-15,max_dt=STEP,output=true,outputN=7,stepfun=window!)
        writec(joinpath(dir,"$suffix.txt"),field);writedlm(joinpath(dir,"$suffix-z.txt"),z)
    end
    gx,gw=QuadGK.gauss(12);rule=reduce(hcat,([gx[i],gx[j],gw[i]*gw[j]] for i=1:12 for j=1:12))
    writedlm(joinpath(dir,"rule.txt"),rule')
    function fixednodes!(out,field,z)
        NR.reset!(rhs!,field,z);_,lo,hi=MO.dimlimits(modes[1];z)
        half=(collect(hi)-collect(lo))/2;mid=(collect(hi)+collect(lo))/2
        points=rule[1:2,:].*half.+mid;values=zeros(length(E)*2,size(points,2))
        NR.pointcalc!(values,points,rhs!)
        out.=reshape(reinterpret(ComplexF64,values*rule[end,:]*prod(half)),size(E))
    end
    positions=Float64[];samples=Vector{ComplexF64}[]
    function recorded!(out,z)
        linop(out,z);push!(positions,z);push!(samples,vec(copy(out)))
    end
    z,field,_=RK45.solve_precon(fixednodes!,recorded!,E,0.,STEP,STEP;rtol=1e-9,atol=1e-12,
        min_dt=STEP,max_dt=STEP,output=true,outputN=7,stepfun=window!)
    writec(joinpath(dir,"interval.txt"),field);writedlm(joinpath(dir,"linear-positions.txt"),positions)
    writec(joinpath(dir,"linear-samples.txt"),hcat(samples...))
    println("Exported callback ",name);flush(stdout)
end
