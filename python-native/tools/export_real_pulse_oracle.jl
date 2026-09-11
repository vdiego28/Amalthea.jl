ENV["AMALTHEA_USE_RUST_NATIVE"]="0"
using Amalthea, FFTW, DelimitedFiles
const F=Amalthea.Fields
root=abspath(ARGS[1]);mkpath(root)
grid=Amalthea.Grid.RealGrid(.1,800e-9,(150e-9,1800e-9),100e-15)
FT=plan_rfft(zeros(length(grid.t));flags=FFTW.ESTIMATE)
writedlm(joinpath(root,"time-axis.txt"),grid.t)
writedlm(joinpath(root,"omega-axis.txt"),grid.ω)
x=collect(range(0.,1.;length=151)); omega=1.5e15 .+ 2.5e15.*x.^1.2
intensity=@. exp(-((omega-2.7e15)/4e14)^2)
phase=@. 8*(omega-2.7e15)/4e14 + 3e-30*(omega-2.7e15)^2
writedlm(joinpath(root,"source.txt"),hcat(omega,intensity,phase))
g=F.GaussField(λ0=780e-9,τfwhm=4e-15,energy=30e-9)
cep=F.GaussField(λ0=800e-9,τfwhm=2.5e-15,power=1e7,ϕ=[1.1,3e-15,1e-30])
s=F.SechField(λ0=880e-9,τw=2e-15,energy=20e-9,ϕ=[.3,-4e-15])
d=F.DataField(omega,intensity,phase;energy=10e-9,ϕ=[.3,2e-15,2e-30])
dc=F.DataField(omega,sqrt.(intensity).*exp.(im.*phase);energy=10e-9,ϕ=[.3,2e-15,2e-30])
gain!(E,grid)=(E .*= .8 .* exp.(-im.*grid.ω.*2e-15))
p=F.PropagatedField(gain!,g)
energy_t,energy_ω=F.energyfuncs(grid)
for (name,fields) in ["gauss"=>(g,),"cep-power"=>(cep,),"sech"=>(s,),
                      "multi"=>(g,s),"data"=>(d,),"complex-data"=>(dc,),
                      "propagated"=>(p,),"mixed"=>(p,d)]
    dir=joinpath(root,name);mkpath(dir)
    E=sum(f(grid,FT) for f in fields);time=FT\E
    writedlm(joinpath(dir,"spectrum.txt"),hcat(real.(E),imag.(E)))
    writedlm(joinpath(dir,"time.txt"),hcat(time,F.It(time,grid)))
    writedlm(joinpath(dir,"energy.txt"),[energy_t(time),energy_ω(E),maximum(F.It(time,grid))])
end
println("Exported carrier-resolved pulse fixtures to ",root)
