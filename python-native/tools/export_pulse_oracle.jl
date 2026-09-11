ENV["AMALTHEA_USE_RUST_NATIVE"] = "0"
ENV["AMALTHEA_USE_RUST_STEPPER"] = "0"
using Amalthea, DelimitedFiles
Amalthea.set_fftw_mode(:estimate); Amalthea.set_fftw_threads(1)
root=abspath(ARGS[1]); mkpath(root)
common=(λ0=800e-9, λlims=(500e-9,1800e-9), trange=600e-15,
        shotnoise=false, raman=false, saveN=7)
# Curved, nonuniform supplied frequency coordinates, phase crossing multiple wraps.
x=collect(range(0.,1.;length=151)); omega=1.5e15 .+ 1.8e15.*x.^1.2
intensity=@. exp(-((omega-2.4e15)/2e14)^2)*(1+0.15*cos((omega-2.4e15)/5e13))
phase=@. 7*(omega-2.4e15)/4e14 + 1e-29*(omega-2.4e15)^2
writedlm(joinpath(root,"source.txt"),hcat(omega,intensity,phase))
gain!(E,grid)=(E .*= .8 .* exp.(-1im .* ((grid.ω .- grid.ω0).*3e-15 .+ 5e-30.*(grid.ω .- grid.ω0).^2)))
gauss(;kwargs...)=Pulses.GaussPulse(;λ0=780e-9,τfwhm=24e-15,energy=30e-12,ϕ=[.2,-20e-15,1e-29],kwargs...)
sech()=Pulses.SechPulse(λ0=880e-9,τw=15e-15,power=800.,ϕ=[1.1,40e-15])
data()=Pulses.DataPulse(omega,intensity,phase;energy=20e-12,ϕ=[.3,5e-15,2e-29])
complex_data()=Pulses.DataPulse(omega,sqrt.(intensity).*exp.(1im.*phase);energy=20e-12,ϕ=[.3,5e-15,2e-29])
for (name,pulses) in ["gauss"=>gauss(), "sech"=>sech(), "multi"=>[gauss(),sech()],
                      "data"=>data(), "complex-data"=>complex_data(),
                      "propagated"=>gauss(propagator=gain!),
                      "mixed"=>[gauss(propagator=gain!),data()]]
    dir=joinpath(root,name);mkpath(dir)
    E,g,L,rhs!,FT,out=Amalthea.Interface.prop_gnlse_args(.01,.02,[0.,0.,-20e-27];common...,pulses)
    nl=similar(E);rhs!(nl,E,0.)
    writedlm(joinpath(dir,"setup.txt"),hcat(real.(E),imag.(E),real.(nl),imag.(nl)))
    window!(y,z,dz,interp)=(y .= FT*((FT\(y.*g.ωwin)).*g.twin))
    z,y,_=Amalthea.RK45.solve_precon(rhs!,L,E,0.,.001,.001;min_dt=.001,max_dt=.001,
        output=true,outputN=3,stepfun=window!)
    writedlm(joinpath(dir,"interval.txt"),hcat(real.(y),imag.(y)))
    output=Amalthea.prop_gnlse(.01,.02,[0.,0.,-20e-27];common...,pulses)
    writedlm(joinpath(dir,"full.txt"),hcat(real.(output["Eω"]),imag.(output["Eω"])))
    writedlm(joinpath(dir,"z.txt"),output["z"])
end
