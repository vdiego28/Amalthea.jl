using Pkg
Pkg.activate(".")

using Test

content = read("luna-rust/tests/test_gpu_cuda.jl", String)
lines = split(content, "\n")
start_idx = findfirst(l -> startswith(l, "@testitem"), lines)
end_idx = findlast(l -> l == "end", lines)
inner_code = join(lines[start_idx+1:end_idx-1], "\n")
open("temp_test_gpu_cuda.jl", "w") do f
    write(f, inner_code)
end

content2 = read("luna-rust/tests/test_scans_io.jl", String)
lines2 = split(content2, "\n")
start_idx2 = findfirst(l -> startswith(l, "@testitem"), lines2)
end_idx2 = findlast(l -> l == "end", lines2)
inner_code2 = join(lines2[start_idx2+1:end_idx2-1], "\n")
open("temp_test_scans_io.jl", "w") do f
    write(f, inner_code2)
end

@testset "GPU CUDA Dispatch" begin
    include("temp_test_gpu_cuda.jl")
end

@testset "Scans IO" begin
    include("temp_test_scans_io.jl")
end
