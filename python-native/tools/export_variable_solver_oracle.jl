# Independent serial variable-linear-operator driver oracle.
ENV["AMALTHEA_USE_RUST_NATIVE"]="0"
ENV["AMALTHEA_USE_RUST_STEPPER"]="0"
using Amalthea,DelimitedFiles,TOML
root=abspath(ARGS[1]);mkpath(root)
base=ComplexF64[.13+.2im -.08+.31im; -.05+.4im .11-.17im]
slope=ComplexF64[.04+.13im .03-.17im; -.02+.07im .06+.11im]
initial=ComplexF64[1 .3; .7im .4+.1im]
for fifth in (false,true), kind in ("interval","fixed","adaptive","filtered","filtered-adaptive","constant","linear")
    adaptive=occursin("adaptive",kind)
    start=.125;stop=kind=="interval" ? .175 : .425
    dt=adaptive ? .3 : .05
    positions=Float64[];accepted=Float64[]
    function linear!(out,z)
        push!(positions,z)
        out .= base .+ (kind=="constant" ? 0 : sin(3z)).*slope
    end
    function nonlinear!(out,y,z)
        if kind=="linear"
            fill!(out,0)
        else
            out .= (1+.3z).*y.^2 .+ .04sum(y)
        end
    end
    function filter!(y,z,dz,interp)
        push!(accepted,z)
        if startswith(kind,"filtered")
            y .*= [.9998 .9997; .9996 .9999]
        end
    end
    count=kind=="interval" ? 7 : 17
    name="$(fifth ? "fifth" : "fourth")-$kind";dir=joinpath(root,name);mkpath(dir)
    solution=try
        Amalthea.RK45.solve_precon(nonlinear!,linear!,initial,start,dt,stop;
            min_dt=adaptive ? 1e-15 : dt,max_dt=dt,rtol=1e-10,atol=1e-12,
            locextrap=fifth,output=true,outputN=count,stepfun=filter!)
    catch error
        message=sprint(showerror,error)
        if fifth && kind=="filtered-adaptive" && occursin("step repetition",message)
            write(joinpath(dir,"expected-failure.txt"),message)
            println("Exported expected repetition failure ",name);flush(stdout)
            continue
        end
        rethrow()
    end
    z,y,_=solution
    field=reshape(y,length(initial),:)
    writedlm(joinpath(dir,"field.txt"),hcat(z,permutedims(real.(field)),permutedims(imag.(field))))
    writedlm(joinpath(dir,"linear-positions.txt"),positions)
    writedlm(joinpath(dir,"accepted.txt"),accepted)
    println("Exported variable solver ",name);flush(stdout)
end
