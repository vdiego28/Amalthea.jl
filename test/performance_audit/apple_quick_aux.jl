#!/usr/bin/env julia

using Amalthea
import Logging: NullLogger, with_logger
import Distributed

length(ARGS) == 2 || error("usage: apple_quick_aux.jl modal|scan OUTPUT_JSON")
kind, output_path = ARGS
# Scan interprets any remaining process arguments as its own execution options.
empty!(ARGS)
Amalthea.set_fftw_mode(:estimate)
Amalthea.set_fftw_threads(1)

json_string(s) = "\"" * replace(string(s), "\\" => "\\\\", "\"" => "\\\"") * "\""
function write_json(path, pairs)
    mkpath(dirname(path))
    open(path, "w") do io
        write(io, "{\n" * join(("  \"$k\": $v" for (k, v) in pairs), ",\n") * "\n}\n")
    end
end

if kind == "modal"
    a = 40e-6; gas = :Ar; pressure = 1.0; λ0 = 800e-9
    grid = Grid.RealGrid(1e-3, λ0, (400e-9, 2000e-9), 0.2e-12)
    modes = (Capillary.MarcatiliMode(a, gas, pressure; m=1),
             Capillary.MarcatiliMode(a, gas, pressure; m=2))
    density = PhysData.density(gas, pressure)
    densityfun(z) = density
    responses = (Nonlinear.Kerr_field(PhysData.γ3_gas(gas)),)
    input = Fields.GaussField(λ0=λ0, τfwhm=25e-15, energy=2e-7)
    Eω, transform, _ = withenv("AMALTHEA_USE_RUST_NATIVE" => "0") do
        with_logger(NullLogger()) do
            Amalthea.setup(grid, densityfun, responses, input, modes, :y; mfcn=128)
        end
    end
    xs = reshape(collect(range(0.08a, 0.92a; length=65)), 1, :)
    reference = zeros(2length(Eω), size(xs, 2))
    threaded = similar(reference)
    NonlinearRHS.reset!(transform, Eω, 0.0)
    transform.modal_threaded = false
    NonlinearRHS.pointcalc!(reference, xs, transform)
    NonlinearRHS.reset!(transform, Eω, 0.0)
    transform.modal_threaded = !isempty(transform.modal_scratch)
    NonlinearRHS.pointcalc!(threaded, xs, transform)
    GC.gc()
    elapsed = @elapsed for _ in 1:5
        NonlinearRHS.reset!(transform, Eω, 0.0)
        NonlinearRHS.pointcalc!(threaded, xs, transform)
    end
    write_json(output_path, [
        "kind" => json_string(kind),
        "threads" => string(Threads.nthreads()),
        "threaded_enabled" => string(transform.modal_threaded),
        "exact" => string(threaded == reference),
        "elapsed_seconds" => repr(elapsed / 5),
        "points" => string(size(xs, 2)),
    ])
elseif kind == "scan"
    Core.eval(Main, :(using Amalthea))
    before = sort(Distributed.workers())
    dir = mktempdir()
    name = "apple_quick_scan_$(getpid())"
    scan = Scan(name, Scans.QueueExec(nproc=2, threads_per_worker=1); value=1:8)
    elapsed = @elapsed Main.eval(quote
        Amalthea.Scans.runscan($scan) do scanidx, value
            write(joinpath($dir, string(scanidx)), "$(value):$(Threads.nthreads())")
        end
    end)
    files = sort(parse.(Int, readdir(dir)))
    worker_threads_ok = all(s -> endswith(s, ":1"),
                            read.(joinpath.(dir, string.(1:8)), String))
    cleanup_ok = sort(Distributed.workers()) == before
    rm(dir; recursive=true)
    write_json(output_path, [
        "kind" => json_string(kind),
        "elapsed_seconds" => repr(elapsed),
        "exact_once" => string(files == collect(1:8)),
        "worker_threads_ok" => string(worker_threads_ok),
        "cleanup_ok" => string(cleanup_ok),
    ])
else
    error("unknown kind $kind")
end

println(output_path)
