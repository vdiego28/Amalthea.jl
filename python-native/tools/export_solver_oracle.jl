using Amalthea, DelimitedFiles
outdir = abspath(ARGS[1])
mkpath(outdir)
ENV["AMALTHEA_USE_RUST_NATIVE"] = "0"
ENV["AMALTHEA_USE_RUST_STEPPER"] = "0"
f!(out, y, z) = (out .= (1 + .3z) .* y.^2)
for fifth in (false, true), fixed in (false, true)
    name = "$(fifth ? "fifth" : "fourth")-$(fixed ? "fixed" : "adaptive")"
    z, y, _ = Amalthea.RK45.solve_precon(f!, ComplexF64[.13+.2im], ComplexF64[1],
        0., .05, .4; rtol=1e-10, atol=1e-12, max_dt=.05,
        min_dt=fixed ? .05 : 1e-15, locextrap=fifth, output=true, outputN=17)
    writedlm(joinpath(outdir, "$name.txt"), hcat(z, real.(vec(y)), imag.(vec(y))))
end
for fifth in (false, true), stop in (.35, .4)
    no_rhs!(out, y, z) = fill!(out, 0)
    filter!(y, z, dt, interp) = (y .*= .9)
    count = stop == .35 ? 8 : 5
    z, y, _ = Amalthea.RK45.solve_precon(no_rhs!, ComplexF64[0], ComplexF64[1],
        0., .1, stop; min_dt=.1, max_dt=.1, locextrap=fifth,
        output=true, outputN=count, stepfun=filter!)
    name = "filter-$(fifth ? "fifth" : "fourth")-$stop"
    writedlm(joinpath(outdir, "$name.txt"), hcat(z, real.(vec(y))))
end
println("Exported independent Julia solver trajectories to ", outdir)

for fifth in (false, true)
    nonlinear!(out, y, z) = (out .= y.^2)
    window!(y, z, dt, interp) = (y .*= [.95, .7])
    z, y, _ = Amalthea.RK45.solve_precon(nonlinear!, ComplexF64[.13+.2im, -.1+.4im],
        ComplexF64[1, .7im], 0., .05, .15; min_dt=.05, max_dt=.05,
        locextrap=fifth, output=true, outputN=7, stepfun=window!)
    name = "window-$(fifth ? "fifth" : "fourth")"
    writedlm(joinpath(outdir, "$name.txt"), hcat(z, permutedims(real.(y)), permutedims(imag.(y))))
end
