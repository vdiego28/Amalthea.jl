module Scans
import ArgParse: ArgParseSettings, parse_args, parse_item, @add_arg_table!
import Logging: @info, @warn
import Printf: @sprintf
import Base: length, size
import Amalthea: Utils
import FileWatching.Pidfile: mkpidlock
import HDF5
import Distributed: @spawnat, addprocs, rmprocs, fetch, remotecall_fetch
import Dates
import SHA

"""
    AbstractExec

Abstract supertype for scan execution modes.
"""
abstract type AbstractExec end

"""
    LocalExec

Execution mode to simply run the whole scan in the current Julia session in a `for` loop.
"""
struct LocalExec <: AbstractExec end

"""
    RangeExec(range)

Execution mode to run a subsection of the scan, given by a `UnitRange`, in the current Julia session.
"""
struct RangeExec <: AbstractExec
    r::UnitRange{Int}
end


"""
    BatchExec(Nbatches, batch)

Execution mode to divide the scan into `Nbatches` chunks and run only the given `batch`.
"""
struct BatchExec <: AbstractExec
    Nbatches::Int
    batch::Int
end


"""
    QueueExec(nproc=0, queuefile=""; threads_per_worker=1)

Execution mode to run a scan using a file-based queueing system. Can be run in multiple separate
Julia sessions, or can spawn `nproc` subprocesses which then take items from the queue to run.

Possible values for `nproc` are:
- `0`: run only in the current Julia process
- `n > 0`: spawn `n` subprocesses and run on these
- `-1`: spawn as many subprocesses as the number of logical cores on the CPU
    (`Base.Sys.CPU_THREADS`)

If `queuefile` is given, the queuefile is stored at that path. If omitted, the queuefile is 
stored under `Utils.cachedir()` using a stable digest of the scan name. Note that the
queuefile is deleted at the end of the scan. `threads_per_worker` controls Julia, FFTW,
and resident-Rayon concurrency inside each spawned worker; its default of one avoids
process × thread oversubscription.
"""
struct QueueExec <: AbstractExec
    nproc::Int
    queuefile::String
    threads_per_worker::Int
end

function QueueExec(; nproc=0, queuefile="", threads_per_worker=1)
    threads_per_worker >= 1 || throw(ArgumentError("threads_per_worker must be positive"))
    QueueExec(Int(nproc), String(queuefile), Int(threads_per_worker))
end
QueueExec(nproc::Integer) = QueueExec(; nproc)
QueueExec(nproc::Integer, queuefile::AbstractString; threads_per_worker=1) =
    QueueExec(; nproc, queuefile, threads_per_worker)

"""
    CondorExec(scriptfile, ncores)

Execution mode which submits a scan to an HTCondor queue system claiming `ncores` cores.

!!! note
    `scriptfile` must **always** be `@__FILE__`
"""
struct CondorExec <: AbstractExec
    scriptfile::String
    ncores::Int
end

"""
    SlurmExec(scriptfile, ncores)

Execution mode which submits a scan to an slurm queue system claiming `ncores` cores.

!!! note
    `scriptfile` must **always** be `@__FILE__`
"""
struct SlurmExec <: AbstractExec
    scriptfile::String
    ncores::Int
end

"""
    SSHExec(localexec, scriptfile, hostname, subdir; files=String[])

Execution mode which transfers the `scriptfile` file to the host given by `hostname` via SSH
and executes the scan on that host with a mode defined by `localexec`. `subdir` gives the
subdirectory (relative to the home directory) where scans are stored on the remote host. A
subfolder with automatically chosen name will be created in `subdir` to store this scan.

Optional keyword argument `files` is a list of auxiliary files to transfer to the remote host
along with the script. These files will be placed in the same directory as the scan script.

!!! note
    `scriptfile` must **always** be `@__FILE__`
"""
struct SSHExec{eT} <: AbstractExec
    localexec::eT
    scriptfile::String
    hostname::String
    subdir::String
    files::Vector{String}
end

function SSHExec(le::CondorExec, hostname, subdir; files=String[])
    SSHExec(le, le.scriptfile, hostname, subdir, files)
end

function SSHExec(le::SlurmExec, hostname, subdir; files=String[])
    SSHExec(le, le.scriptfile, hostname, subdir, files)
end

struct Scan{eT}
    name::String
    variables::Vector{Symbol}
    arrays::Vector
    exec::eT
end

"""
    Scan(name; kwargs...)
    Scan(name, cmdlineargs::Vector{String}; kwargs...)
    Scan(name, ex::AbstractExec; kwargs...)

Create a new `Scan` with name `name` and variables given as keyword arguments. The execution
mode `ex` can be given directly or via command-line arguments to the script. **If given,
command-line arguments overwrite any explicitly passed execution mode.** An
explicit argument vector is used as given, independently of global `ARGS`.
Implicit command-line arguments are ignored in an initialized IJulia notebook.
Neither global `ARGS` nor an explicitly supplied argument vector is modified.

If neither an explicit execution mode nor command-line arguments are given,
`ex` defaults to `LocalExec`, i.e. running the whole scan locally in the current Julia process.
"""
function Scan(name, cmdlineargs::Vector{String}=_scan_default_args(); kwargs...)
    _scan_with_exec(name, makeexec(copy(cmdlineargs)); kwargs...)
end

function Scan(name, ex::AbstractExec; kwargs...)
    cmdlineargs = _scan_default_args()
    _scan_with_exec(name, isempty(cmdlineargs) ? ex : makeexec(cmdlineargs); kwargs...)
end

function _scan_default_args(context::Module=Main)
    if isdefined(context, :IJulia)
        ijulia = getfield(context, :IJulia)
        if ijulia isa Module && isdefined(ijulia, :inited) &&
                getfield(ijulia, :inited) === true
            return String[]
        end
    end
    copy(ARGS)
end

function _scan_with_exec(name, ex::AbstractExec; kwargs...)
    variables = Symbol[]
    arrays = Vector[]
    for (var, arr) in kwargs
        push!(variables, var)
        push!(arrays, arr)
    end
    Scan(name, variables, arrays, ex)
end

length(s::Scan) = (length(s.arrays) == 0) ? 0 : prod(length, s.arrays)
size(s::Scan) = (length(s.arrays) == 0) ? (0,) : Tuple(length.(s.arrays))

"""
    addvariable!(scan, variable::Symbol, array)
    addvariable!(scan; kwargs...)

Add scan variable(s) to the `scan`, either as a single pair of `Symbol` and array, or as a 
sequence of keyword arguments.
"""
function addvariable!(scan, variable::Symbol, array)
    push!(scan.variables, variable)
    push!(scan.arrays, array)
end

function addvariable!(scan; kwargs...)
    for (var, arr) in kwargs
        push!(scan.variables, var)
        push!(scan.arrays, arr)
    end
end

"""
    makefilename(scan, scanidx)

Make an appropriate file name for an `HDF5Output` or `prop_capillary` output for the
`scan` at the current `scanidx`.

# Examples
```
scan = Scan("scan_example"; energy=collect(range(5e-6, 200e-6; length=64)))
runscan(scan) do scanidx, energyi
    prop_capillary(125e-6, 3, :He, 0.8; λ0=800e-9, τfwhm=10e-15, energy=energyi
                   filepath=makefilename(scan, scanidx))
end
```
"""
makefilename(scan::Scan, scanidx) = makefilename(scan.name, scanidx)
makefilename(name::AbstractString, scanidx) = @sprintf("%s_%05d.h5", name, scanidx)

function makeexec(args::Vector{String})
    isempty(args) && return LocalExec()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--local", "-l"
            help = "Execute the whole scan locally."
            action = :store_true
        "--range", "-r"
            help = "Linear range of scan indices to execute."
            arg_type = UnitRange{Int}
        "--batch", "-b"
            help = "Number of batches and batch index to execute"
            arg_type = Tuple{Int, Int}
        "--queue", "-q"
            help = """Use a file-based queue to execute the scan. Can be run in parallel
                      using the --procs/-p option."""
            action = :store_true
        "--procs", "-p"
            help = """Number of processes to use with queued execution. If 0, use only the
                      main Julia instance. If -1, use as many processes as the machine
                      has logical cores."""
            arg_type = Int
            default = 0
    end
    args = parse_args(args, s)
    for k in keys(args)
        isnothing(args[k]) && delete!(args, k)
    end
    args["local"] && return LocalExec()
    haskey(args, "range") && return RangeExec(args["range"])
    haskey(args, "r") && return RangeExec(args["r"])
    haskey(args, "batch") && return BatchExec(args["batch"]...)
    haskey(args, "b") && return BatchExec(args["b"]...)
    args["queue"] && return QueueExec(args["procs"])
    error("Command-line arguments do not define a valid execution mode.")
end

# Enable parsing of command-line arguments of the form "1:5" to a UnitRange
function parse_item(::Type{UnitRange{Int}}, x::AbstractString)
    parts = split(x, ":")
    length(parts) == 2 || throw(ArgumentError("Invalid format for UnitRange{Int}: $x. Expected start:stop"))
    return parse(Int, parts[1]):parse(Int, parts[2])
end

# Enable parsing of command-line arguments of the form "1,5" to a Tuple of integers
parse_item(::Type{Tuple{Int, Int}}, x::AbstractString) = Tuple(parse(Int, xi) for xi in split(x, ","))

function logiter(scan, scanidx, args)
    logmsg = @sprintf("Running scan: %s (%d points)\nIndex: %05d\nVariables:\n",
                      scan.name, length(scan), scanidx)
    for (variable, value) in zip(scan.variables, args)
        logmsg *= @sprintf("\t%s: %g\n", variable, value)
    end
    @info logmsg
end

function getvalue(scan, variable, scanidx)
    values = vec(collect(Iterators.product(scan.arrays...)))[scanidx]
    idx = findfirst(scan.variables .== variable)
    values[idx]
end

"""
    runscan(f, scan)

Run the function `f` in a scan with arguments defined by the `scan::Scan`.
The function `f` must have the signature `f(scanidx, args...)` where the length of `args`
is the number of variables to be scanned over. Can be used with the `do` block syntax.

The exact subset and order of scan points which is run depends on `scan.exec`, see
[`Scan`](@ref).

# Examples
```
scan = Scan("scan_example"; energy=collect(range(5e-6, 200e-6; length=64)))
runscan(scan) do scanidx, energyi
    prop_capillary(125e-6, 3, :He, 0.8; λ0=800e-9, τfwhm=10e-15, energy=energyi)
end
```
"""
function runscan(f, scan::Scan{LocalExec})
    out = Any[]
    for (scanidx, args) in enumerate(Iterators.product(scan.arrays...))
        logiter(scan, scanidx, args)
        try
            push!(out, f(scanidx, args...))
        catch e
            bt = catch_backtrace()
            msg = "Error at scanidx $scanidx:\n"*sprint(showerror, e, bt)
            @warn msg
        end
        Base.GC.gc()
    end
    out
end

function runscan(f, scan::Scan{RangeExec})
    out = Any[]
    combos = vec(collect(Iterators.product(scan.arrays...)))
    for scanidx in scan.exec.r
        args = combos[scanidx]
        logiter(scan, scanidx, args)
        try
            push!(out, f(scanidx, args...))
        catch e
            bt = catch_backtrace()
            msg = "Error at scanidx $scanidx:\n"*sprint(showerror, e, bt)
            @warn msg
        end
        Base.GC.gc()
    end
    out
end

"""
    chunks(a::AbstractArray, n::Int)

Split array a into n chunks, spreading the entries of a evenly.

# Examples
```jldoctest
julia> a = collect(range(1, length=10));
julia> Scans.chunks(a, 3)
3-element Array{Array{Int64,1},1}:
 [1, 4, 7, 10]
 [2, 5, 8]
 [3, 6, 9]
```
"""
function chunks(a::AbstractArray, n::Int)
    N = length(a)
    done = 0
    out = [Array{eltype(a), 1}() for ii=1:n]
    while done < N
        push!(out[mod(done, n)+1], a[done+1])
        done += 1
    end
    return out
end

function runscan(f, scan::Scan{BatchExec})
    combos = vec(collect(Iterators.product(scan.arrays...)))
    linidx = collect(1:length(scan))
    chs = chunks(linidx, scan.exec.Nbatches)
    idcs = chs[scan.exec.batch]
    scanidcs_this = linidx[idcs]
    combos_this = combos[idcs]
    for (scanidx, args) in zip(scanidcs_this, combos_this)
        logiter(scan, scanidx, args)
        try
            f(scanidx, args...)
        catch e
            bt = catch_backtrace()
            msg = "Error at scanidx $scanidx:\n"*sprint(showerror, e, bt)
            @warn msg
        end
        Base.GC.gc()
    end
end

function runscan(f, scan::Scan{QueueExec})
    if scan.exec.nproc == 0
        _runscan(f, scan)
    else
        nproc = if scan.exec.nproc == -1
            max(1, fld(Base.Sys.CPU_THREADS, scan.exec.threads_per_worker))
        else
            scan.exec.nproc
        end
        procs = addprocs(nproc; exeflags="--threads=$(scan.exec.threads_per_worker)")
        try
            threads_per_worker = scan.exec.threads_per_worker
            setup_expr = quote
                using Amalthea
                import FFTW
                Amalthea.settings["fftw_threads"] = $threads_per_worker
                FFTW.set_num_threads($threads_per_worker)
                nothing
            end
            for p in procs
                remotecall_fetch(Core.eval, p, Main, setup_expr)
            end
            futures = [@spawnat p _runscan(f, scan) for p in procs]
            fetch.(futures)
        finally
            rmprocs(procs)
        end
    end
end

function _runscan(f, scan::Scan{QueueExec})
    if isempty(scan.exec.queuefile)
        qdir = Utils.cachedir()
        isdir(qdir) || mkpath(qdir)
        digest = bytes2hex(SHA.sha256(scan.name))[1:16]
        qfile = joinpath(qdir, "qfile_$digest.h5")
    else
        qfile = scan.exec.queuefile
    end
    lockpath = qfile*"_lock"

    combos = vec(collect(Iterators.product(scan.arrays...)))
    qfile_created = false
    while true
        scanidx = nothing
        qdata = fill(2, length(scan))
        mkpidlock(lockpath; stale_age=120) do
            # first process to catch the pidlock creates the queue file
            if ~isfile(qfile)
                if qfile_created
                    # The queue file was already created, so another process
                    # must have completed the scan and removed it. Signal
                    # that we should stop by setting scanidx to nothing.
                    scanidx = nothing
                    qdata = fill(2, length(scan))
                    return
                end
                HDF5.h5open(qfile, "cw") do file
                    file["qdata"] = zeros(Int, length(scan))
                end
            end
            qfile_created = true
            # read the queue data
            qdata = HDF5.h5open(qfile) do file
                read(file["qdata"])
            end
            # find the first index which is neither done nor in progress
            scanidx = findfirst(qdata) do qi
                qi == 0
            end
            if ~isnothing(scanidx)
                # mark the index as in progress
                HDF5.h5open(qfile, "r+") do file
                    file["qdata"][scanidx] = 1
                end
            end
        end # release pidlock
        if isnothing(scanidx) # no scan points left to start
            if all(qdata .> 1) # completely done--either all done or failed
                rm(qfile; force=true) # remove the queue file
            end
            break # break out of the loop
        end
        logiter(scan, scanidx, combos[scanidx])
        code = 2 # code for finished successfully
        try
            f(scanidx, combos[scanidx]...) # run scan function
        catch e
            code = 3 # code for failed
            bt = catch_backtrace()
            msg = "Error at scanidx $scanidx:\n"*sprint(showerror, e, bt)
            @warn msg
        end
        mkpidlock(lockpath; stale_age=10) do # acquire lock on qfile again
            if isfile(qfile)
                HDF5.h5open(qfile, "r+") do file
                    file["qdata"][scanidx] = code # mark as done/failed
                end
            end
        end
        Base.GC.gc()
    end
end

function runscan(f, scan::Scan{SlurmExec})
    # make submission file for slurm
    script = scan.exec.scriptfile
    dir = dirname(script)
    cores = scan.exec.ncores
    name = scan.name
    @info "Submitting slurm job for $script running on $cores cores."
    # Adding the --queue command-line argument below means that when running the Condor job,
    # the SlurmExec is ignored even if explicitly defined inside the script.
    lines = [
        "#!/bin/bash",
        "#SBATCH --ntasks=1",
        "#SBATCH --cpus-per-task=1",
        "#SBATCH -o %x_%a.stdout",
        "#SBATCH -e %x_%a.stderr",
        "#SBATCH --array=1-$cores",
        "#SBATCH --chdir $dir",
        "julia $(basename(script)) --queue"
    ]
    subfile = joinpath(dir, "$name.sh")
    @info "Writing job file to $subfile..."
    open(subfile, "w") do file
        for l in lines
            write(file, l*"\n")
        end
    end
    @info "Submitting job..."
    out = read(Cmd(["sbatch", subfile]), String)
    @info "Slurm submission output:\n$out"
end

function runscan(f, scan::Scan{CondorExec})
    # make submission file for HTCondor
    cmd = split(string(Base.julia_cmd()))[1]
    julia = strip(cmd, ['`', '\''])
    script = scan.exec.scriptfile
    cores = scan.exec.ncores
    name = scan.name
    @info "Submitting Condor job for $script running on $cores cores."
    # Adding the --queue command-line argument below means that when running the Condor job,
    # the CondorExec is ignored even if explicitly defined inside the script.
    lines = [
        "executable = $julia",
        """arguments = "$(basename(script)) --queue" """,
        "log = $name.log.\$(Process)",
        "output = $name.out.\$(Process)",
        "error = $name.err.\$(Process)",
        "stream_error = True",
        "initialdir = $(dirname(script))",
        "request_cpus = 1",
        "queue $cores"
    ]
    subfile = joinpath(dirname(script), "doit.sub")
    @info "Writing job file to $subfile..."
    open(subfile, "w") do file
        for l in lines
            write(file, l*"\n")
        end
    end
    @info "Submitting job..."
    out = read(Cmd(["condor_submit", subfile]), String)
    @info "Condor submission output:\n$out"
end

function changexec(scan, newexec)
    newscan = _scan_with_exec(scan.name, newexec)
    for (var, arr) in zip(scan.variables, scan.arrays)
        addvariable!(newscan, var, arr)
    end
    newscan
end

function runscan(f, scan::Scan{<:SSHExec})
    if gethostname() == scan.exec.hostname
        # running on the machine defined in the SSH Exec? just run the scan
        runscan(f, changexec(scan, scan.exec.localexec))
    else
        # running somewhere else? submit the job via SSH
        host = scan.exec.hostname
        subdir = scan.exec.subdir
        script = scan.exec.scriptfile
        scriptfile = basename(script)
        name = scan.name
        folder = Dates.format(Dates.now(), "yyyymmdd_HHMMSS") * "_$name"
        
        subdir_esc = Base.shell_escape_posixly(subdir)
        folder_esc = Base.shell_escape_posixly(folder)
        scriptfile_esc = Base.shell_escape_posixly(scriptfile)
        remotedir = "\$HOME/$subdir_esc/$folder_esc"

        @info "Making directory $remotedir"
        read(Cmd(["ssh", host, "mkdir -p $remotedir"]))
        @info "Transferring file..."
        read(Cmd(["scp", script, "$host:$remotedir"]))
        if length(scan.exec.files) > 0
            @info "Transferring auxiliary files..."
            for fi in scan.exec.files
                read(Cmd(["scp", fi, "$host:$remotedir"]))
            end
        end
        @info "Running Amalthea script on remote host $host"
        read(Cmd(["ssh", host, "julia", "$remotedir/$scriptfile_esc"]), String)
    end
end

end
