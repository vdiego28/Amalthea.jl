# Persistent production Julia/Rust worker for the separate Python snapshot.
using TOML, SHA, Logging, LinearAlgebra
const IMPORT_SECONDS = @elapsed using Amalthea
using HDF5
Amalthea.set_fftw_mode(:estimate)
Amalthea.set_fftw_threads(1)
BLAS.set_num_threads(1)
global_logger(NullLogger())
const CASE_FILE, CASE_NAME, BACKEND = ARGS
const CONFIG = TOML.parsefile(CASE_FILE)
const CASE = CONFIG["cases"][CASE_NAME]
const BASE = merge(CONFIG["common"], CASE["options"])
BACKEND in ("julia", "rust") || error("unknown backend")

function inputs(control=false)
    args = deepcopy(CASE["args"])
    options = Dict(Symbol(k)=>v for (k,v) in BASE)
    options[:lambda_lims] = Tuple(options[:lambda_lims])
    for k in (:plasma, :polarisation, :ramanmodel)
        haskey(options,k) && options[k] isa String && (options[k] = Symbol(options[k]))
    end
    if CASE["api"] == "capillary"
        args[3] = Symbol(args[3])
        args[4] isa AbstractVector && (args[4] = Tuple(args[4]))
    end
    if control
        feature = CASE["control"]
        if feature == "gradient"
            args[4] = args[4][1]
        elseif feature == "kerr" && CASE["api"] == "gnlse"
            args[1] = 0.0
        else
            options[Symbol(feature)] = false
        end
    end
    args, options
end

function setup(control=false)
    args, options = inputs(control)
    f = CASE["api"] == "gnlse" ? Amalthea.Interface.prop_gnlse_args : Amalthea.Interface.prop_capillary_args
    f(args...; options...)
end

function dump_array(root, name, array)
    value = eltype(array) <: Complex ? ComplexF64.(array) : Float64.(array)
    path = joinpath(root,name*".bin")
    open(path,"w") do io; write(io, value); end
    Dict("shape"=>collect(size(value)),"dtype"=>(eltype(value)<:Complex ? "<c16" : "<f8"),
         "sha256"=>bytes2hex(sha256(read(path))))
end

function measure(operation, root)
    ispath(root) && error("output already exists")
    mkpath(root)
    record = Dict{String,Any}("operation"=>operation,"requested_backend"=>BACKEND,
        "package"=>pathof(Amalthea),"julia"=>string(VERSION),"import_seconds"=>IMPORT_SECONDS,
        "arrays"=>Dict{String,Any}())
    if operation == "cold"
        args, options = inputs()
        f = CASE["api"] == "gnlse" ? Amalthea.prop_gnlse : Amalthea.prop_capillary
        elapsed = @elapsed result = f(args...; options...)
        field = result["Eω"]; z = result["z"]
        record["first_public_seconds"] = elapsed
        # Public output already records grids; this extra setup is outside timing.
        E,grid,L,rhs!,FT,output = setup()
    else
        GC.gc()
        elapsed = @elapsed fixture = setup(operation == "control")
        E,grid,L,rhs!,FT,output = fixture
        record["setup_seconds"] = elapsed
        accepted = Ref(0)
        temporal = FT\E
        function window!(y,z,dz,interpolant)
            accepted[] += 1
            y .*= grid.ωwin
            ldiv!(temporal,FT,y)
            temporal .*= grid.twin
            mul!(y,FT,temporal)
        end
        elapsed = @elapsed begin
            z,field,attempts = Amalthea.RK45.solve_precon(rhs!,L,E,0.,CASE["step"],grid.zmax;
                rtol=1e-9,atol=1e-12,min_dt=1e-15,max_dt=CASE["step"],output=true,
                outputN=BASE["saveN"],stepfun=window!,status_period=Inf)
        end
        record["solve_seconds"] = elapsed
        record["complete_seconds"] = record["setup_seconds"] + elapsed
        record["accepted_steps"] = accepted[]
        record["rejected_steps"] = attempts - accepted[]
        if operation in ("check","control")
            record["arrays"]["initial"] = dump_array(root,"initial",E)
            op = similar(E); L isa AbstractArray ? (op .= L) : L(op,0.)
            record["arrays"]["linear"] = dump_array(root,"linear",op)
            nl = similar(E); rhs!(nl,E,0.)
            record["arrays"]["rhs"] = dump_array(root,"rhs",nl)
        end
    end
    actual = string(Amalthea.RK45._LAST_STEPPER_TYPE[])
    record["actual_stepper"] = actual
    record["native_eligible"] = occursin("RustNativeStepper",actual)
    BACKEND == "julia" && record["native_eligible"] && error("unexpected native selection")
    elapsed = @elapsed copied = copy(field)
    record["copy_seconds"] = elapsed
    copied == field || error("copy changed values")
    elapsed = @elapsed h5open(joinpath(root,"output.h5"),"w") do file
        file["Eω"] = field; file["z"] = z
    end
    record["hdf5_seconds"] = elapsed
    record["hdf5_scope"] = "Julia HDF5 numeric field and saved positions only"
    record["hdf5_bytes"] = filesize(joinpath(root,"output.h5"))
    for (name,array) in (("field",field),("z",z),("t",grid.t),("omega",grid.ω))
        record["arrays"][name] = dump_array(root,name,array)
    end
    record["peak_rss_bytes"] = Sys.maxrss()
    open(joinpath(root,"result.toml"),"w") do io; TOML.print(io,record); end
end

println("READY");flush(stdout)
for line in eachline(stdin)
    line == "stop" && break
    operation,root = split(line,'\t')
    operation in ("cold","check","control","sample") || error("unknown operation")
    measure(operation,root)
    println("DONE");flush(stdout)
end
