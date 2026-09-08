# Development-only independent Julia setup oracle. No Python implementation used.
using Amalthea, DelimitedFiles, TOML
outdir = abspath(ARGS[1])
mkpath(outdir)
cases = Any[]
for (name, kind, λ0, λlims, trange, δt, thg) in [
    ("real-standard", "real", 800e-9, (400e-9, 1600e-9), 300e-15, 1.0, false),
    ("real-fine", "real", 1030e-9, (300e-9, 2000e-9), 500e-15, 1e-16, false),
    ("env-standard", "env", 800e-9, (400e-9, 1600e-9), 300e-15, 1.0, false),
    ("env-fine", "env", 1030e-9, (300e-9, 2000e-9), 500e-15, 1e-16, false),
    ("env-thg", "env", 800e-9, (400e-9, 1600e-9), 300e-15, 1.0, true),
]
    grid = kind == "real" ? Amalthea.Grid.RealGrid(1.0, λ0, λlims, trange, δt) :
        Amalthea.Grid.EnvGrid(1.0, λ0, λlims, trange; δt, thg)
    kwargs = Dict{String,Any}("zmax"=>1.0, "reference_lambda"=>λ0,
        "lambda_lims"=>collect(λlims), "trange"=>trange, "delta_t"=>δt)
    kind == "env" && (kwargs["thg"] = thg)
    push!(cases, Dict("name"=>name, "kind"=>kind, "kwargs"=>kwargs))
    for (label, values) in [("t", grid.t), ("to", grid.to), ("omega", grid.ω),
        ("omega_over", grid.ωo), ("omega_win", grid.ωwin), ("twin", grid.twin),
        ("towin", grid.towin), ("sidx", Int.(grid.sidx))]
        writedlm(joinpath(outdir, "$name-$label.txt"), values)
    end
end
open(joinpath(outdir, "cases.toml"), "w") do io
    TOML.print(io, Dict("cases" => cases))
end
println("Exported ", length(cases), " independent Julia grids to ", outdir)
