using TestItems

@testitem "Scan argument ownership" tags=[:io] begin
using Amalthea
using Test

saved_args = copy(ARGS)
try
    empty!(ARGS)
    @test Scans.Scan("local"; energy=[1, 2]).exec isa Scans.LocalExec
    requested = Scans.QueueExec(2)
    @test Scans.Scan("explicit", requested).exec === requested

    append!(ARGS, ["--range", "2:3"])
    for _ in 1:2
        scan = Scans.Scan("cli", requested; energy=[1, 2, 3], pressure=[4, 5])
        @test scan.exec isa Scans.RangeExec
        @test scan.exec.r == 2:3
        @test scan.variables == [:energy, :pressure]
        @test scan.arrays == [[1, 2, 3], [4, 5]]
        @test ARGS == ["--range", "2:3"]

        changed = Scans.changexec(scan, requested)
        @test changed.exec === requested
        @test changed.variables == scan.variables
        @test changed.arrays == scan.arrays
    end
    @test Scans.Scan("default").exec.r == 2:3
    args = ["--batch", "3,2"]
    scan = Scans.Scan("argument_vector", args)
    @test scan.exec isa Scans.BatchExec
    @test (scan.exec.Nbatches, scan.exec.batch) == (3, 2)
    @test args == ["--batch", "3,2"]
    @test Scans.Scan("empty_vector", String[]).exec isa Scans.LocalExec
    @test ARGS == ["--range", "2:3"]

    # The argument selector accepts a module so the IJulia contract can be
    # exercised without installing a notebook kernel or changing Main.IJulia.
    context = Module(:ScanNotebookFixture)
    @test Scans._scan_default_args(context) == ARGS
    @test Scans._scan_default_args(context) !== ARGS
    ijulia = Core.eval(context, :(module IJulia end))
    @test Base.invokelatest(Scans._scan_default_args, context) == ARGS
    Core.eval(ijulia, :(inited = false))
    @test Base.invokelatest(Scans._scan_default_args, context) == ARGS
    Core.eval(ijulia, :(inited = true))
    empty!(ARGS)
    append!(ARGS, ["/tmp/jupyter/runtime/kernel-test.json"])
    notebook_args = Base.invokelatest(Scans._scan_default_args, context)
    @test isempty(notebook_args)
    @test Scans.Scan("notebook", notebook_args).exec isa Scans.LocalExec
    @test Scans.Scan("notebook_explicit", ["--range", "1:2"]).exec.r == 1:2
    @test ARGS == ["/tmp/jupyter/runtime/kernel-test.json"]
finally
    empty!(ARGS)
    append!(ARGS, saved_args)
end
end
