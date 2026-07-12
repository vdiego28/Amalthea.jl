using Documenter
using Amalthea

modulesdir = joinpath(Amalthea.Utils.lunadir(), "docs", "src", "modules")

makedocs(
    sitename = "Amalthea.jl Documentation",
    authors = "Diego Andrés Valenzuela Berríos, Christian Brahms, and John C. Travers",
    pages = Any[
        "Home" => "index.md",
        "The numerical model" => [
            "General description" => "model/model.md",
            "Modal decompositions" => "model/modal_decompositions.md",
            "Nonlinear responses" => "model/nonlinear_responses.md",
            "Noise model" => "model/noise.md"
        ],
        "The simple interface" => "interface.md",
        "Parameter scans" => "scans.md",
        "Modules" => [
            "$(split(fi, ".")[1]).jl" => "modules/$fi" for fi in readdir(modulesdir)
        ],
    ],
    format = Documenter.HTML(
        prettyurls = false
    )
)

deploydocs(
    repo = "github.com/vdiego28/Amalthea.jl.git",
)