using TestItems

@testitem "Maintained test coverage and scheduler manifests agree" tags=[:rust] begin
    import Test: @test, @testset

    repo_root = normpath(joinpath(@__DIR__, ".."))
    roots = [
        normpath(joinpath(repo_root, line))
        for line in strip.(readlines(joinpath(@__DIR__, "test_roots.txt")))
        if !isempty(line) && !startswith(line, "#")
    ]
    groups = Set(
        line
        for line in strip.(readlines(joinpath(@__DIR__, "test_groups.txt")))
        if !isempty(line) && !startswith(line, "#")
    )

    function manifest_name(path)
        dirname(path) == (@__DIR__) && return basename(path)
        replace(relpath(path, repo_root), '\\' => '/')
    end

    item_re = r"(?m)^\s*@testitem\s+\"([^\"]+)\"\s+tags\s*=\s*\[([^\]\n]*)\]\s+begin\b"
    item_start_re = r"(?m)^\s*@testitem\b"
    declarations = Dict{String,Vector{Tuple{String,Set{String}}}}()
    for root in roots
        for (dir, _, files) in walkdir(root)
            for file in files
                endswith(file, ".jl") || continue
                path = joinpath(dir, file)
                text = read(path, String)
                matches = collect(eachmatch(item_re, text))
                @test length(matches) == length(collect(eachmatch(item_start_re, text)))
                isempty(matches) && continue
                declarations[manifest_name(path)] = [
                    (m.captures[1], Set(
                        tag.captures[1]
                        for tag in eachmatch(r":([A-Za-z0-9_]+)", m.captures[2])
                    ))
                    for m in matches
                ]
            end
        end
    end

    discovery = joinpath(@__DIR__, "parallel_group_tests.py")
    for (path, items) in declarations
        for (name, tags) in items
            @testset "$path::$name has a maintained group" begin
                @test !isempty(intersect(tags, groups))
            end
        end
    end

    for group in groups
        expected = Set{String}()
        for (path, items) in declarations
            matches = [name for (name, tags) in items if group in tags]
            if length(matches) == 1
                push!(expected, path)
            else
                union!(expected, ("$path::$name" for name in matches))
            end
        end

        output = read(`python3 $discovery --group $group --list-items`, String)
        scheduled = Set(filter(!isempty, split(chomp(output), '\n')))
        @testset "$group discovery parity" begin
            @test scheduled == expected
        end

        timings_path = joinpath(@__DIR__, "$(group)_test_timings.txt")
        timing_keys = if isfile(timings_path)
            Set(
                first(rsplit(line; limit=2))
                for line in strip.(readlines(timings_path))
                if !isempty(line) && !startswith(line, "#")
            )
        else
            Set{String}()
        end
        @testset "$group timing coverage" begin
            for item in scheduled
                @test item in timing_keys || first(split(item, "::"; limit=2)) in timing_keys
            end
        end
    end

    workflow = read(joinpath(repo_root, ".github", "workflows", "run_tests.yml"), String)
    workflow_groups = Set(
        replace(m.captures[1], '-' => '_')
        for m in eachmatch(r"(?m)^\s+group:\s*([A-Za-z0-9_-]+)\s*$", workflow)
    )
    @test groups ⊆ workflow_groups

    rust_items = read(`python3 $discovery --group rust --list-items`, String)
    @test "amalthea/tests/test_gpu_cuda.jl" in split(chomp(rust_items), '\n')
end
