using TestItems

@testitem "Serial and parallel test manifests agree" tags=[:rust] begin
    import Test: @test

    repo_root = normpath(joinpath(@__DIR__, ".."))
    roots_file = joinpath(@__DIR__, "test_roots.txt")
    roots = [
        normpath(joinpath(repo_root, line))
        for line in strip.(readlines(roots_file))
        if !isempty(line) && !startswith(line, "#")
    ]

    function manifest_name(path)
        dirname(path) == (@__DIR__) && return basename(path)
        replace(relpath(path, repo_root), '\\' => '/')
    end

    rust_tag = r"tags\s*=\s*\[[^\]]*:rust\b"
    serial_files = Set{String}()
    for root in roots
        for (dir, _, files) in walkdir(root)
            for file in files
                endswith(file, ".jl") || continue
                path = joinpath(dir, file)
                occursin(rust_tag, read(path, String)) || continue
                push!(serial_files, manifest_name(path))
            end
        end
    end

    discovery = joinpath(@__DIR__, "parallel_group_tests.py")
    output = read(`python3 $discovery --group rust --list-files`, String)
    parallel_files = Set(filter(!isempty, split(chomp(output), '\n')))

    @test parallel_files == serial_files
    @test "amalthea/tests/test_gpu_cuda.jl" in parallel_files
end
