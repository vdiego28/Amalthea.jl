# Companion profile fixture: non-vacuous attenuation limits and zero density.
for key in ("NATIVE","STEPPER","DISPERSION","IONISATION","RAMAN")
    ENV["AMALTHEA_USE_RUST_"*key]="0"
end
using Amalthea,DelimitedFiles
Amalthea.set_fftw_mode(:estimate);Amalthea.set_fftw_threads(1)
root=joinpath(abspath(ARGS[1]),"limits");mkpath(root)
common=(λ0=800e-9,λlims=(200e-9,1700e-9),trange=300e-15,τfwhm=20e-15,
        energy=1e-15,shotnoise=false,plasma=false,kerr=false,saveN=7)
for envelope in (true,false), variable in (true,false)
    radius=variable ? z->1e-6 : 1e-6
    E,grid,L,rhs!,FT,_=Amalthea.Interface.prop_capillary_args(radius,.0002,:Ar,2.;common...,envelope,raman=false)
    op=similar(E);variable ? L(op,0.) : (op.=L)
    name="$(envelope ? "env" : "real")-$(variable ? "variable" : "constant")"
    writedlm(joinpath(root,name*"-linop.txt"),hcat(real.(op),imag.(op)))
    window!(y,z,dz,interp)=(y .= FT*((FT\(y.*grid.ωwin)).*grid.twin))
    z,field,_=Amalthea.RK45.solve_precon(rhs!,L,E,0.,.00005,.0002;
        min_dt=.00005,max_dt=.00005,output=true,outputN=7,stepfun=window!)
    writedlm(joinpath(root,name*".txt"),hcat(real.(field),imag.(field)))
end
for envelope in (true,false)
    E,grid,L,rhs!,FT,_=Amalthea.Interface.prop_capillary_args(125e-6,.0002,:N2,(0.,2.);
        common...,envelope,raman=true)
    nl=similar(E);rhs!(nl,E,0.)
    all(isfinite,nl) || error("zero-density N2 oracle unexpectedly nonfinite")
    name="zero-$(envelope ? "env" : "real")"
    writedlm(joinpath(root,name*"-rhs.txt"),hcat(real.(nl),imag.(nl)))
    window!(y,z,dz,interp)=(y .= FT*((FT\(y.*grid.ωwin)).*grid.twin))
    z,field,_=Amalthea.RK45.solve_precon(rhs!,L,E,0.,.00005,.0002;
        min_dt=.00005,max_dt=.00005,output=true,outputN=7,stepfun=window!)
    writedlm(joinpath(root,name*".txt"),hcat(real.(field),imag.(field)))
end
println("Exported profile loss-clamp and zero-density limits")
