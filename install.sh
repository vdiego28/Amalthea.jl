#!/usr/bin/env bash
# Install or update a tagged Amalthea.jl release on Linux or macOS.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/vdiego28/Amalthea.jl/main/install.sh | bash
#   curl -fsSL https://raw.githubusercontent.com/vdiego28/Amalthea.jl/main/install.sh | bash -s -- --version v1.0.4

set -euo pipefail

REPOSITORY="vdiego28/Amalthea.jl"
PACKAGE_URL="https://github.com/${REPOSITORY}.git"
MIN_JULIA_VERSION="1.10"

usage() {
    cat <<'EOF'
Usage: install.sh [--version TAG]

Install the latest tagged Amalthea.jl release, or the specified release tag,
into Julia's default environment. If Julia is unavailable, install it through
Juliaup first. Rerun the command to update Amalthea.

Options:
  --version TAG  Install a specific release tag (for example, v1.0.4).
  -h, --help     Show this help message.
EOF
}

die() {
    printf 'amalthea installer: %s\n' "$*" >&2
    exit 1
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || die "requires '$1' on PATH"
}

version=""
while [ "$#" -gt 0 ]; do
    case "$1" in
        --version)
            [ "$#" -ge 2 ] || die "--version requires a release tag"
            version="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown option: $1 (use --help for usage)"
            ;;
    esac
done

case "$(uname -s)" in
    Linux|Darwin) ;;
    *) die "supports Linux and macOS only; on Windows use WSL or the manual Julia installation command" ;;
esac

require_command curl

if [ -z "$version" ]; then
    version="$(curl -fsSL "https://api.github.com/repos/${REPOSITORY}/releases/latest" | sed -n 's/^[[:space:]]*"tag_name"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p')"
    [ -n "$version" ] || die "could not determine the latest release; pass --version TAG to select one explicitly"
fi

[[ "$version" =~ ^v[0-9]+(\.[0-9]+){1,3}(-[0-9A-Za-z][0-9A-Za-z._-]*)?$ ]] ||
    die "release tag must be a v-prefixed semantic version (for example, v1.0.4)"

if [ -n "${JULIA:-}" ] && [ ! -x "$JULIA" ]; then
    die "JULIA is set but is not an executable file: $JULIA"
fi

find_julia() {
    if [ -n "${JULIA:-}" ]; then
        printf '%s\n' "$JULIA"
    elif command -v julia >/dev/null 2>&1; then
        command -v julia
    elif [ -x "${HOME}/.juliaup/bin/julia" ]; then
        printf '%s\n' "${HOME}/.juliaup/bin/julia"
    fi
}

julia_bin="$(find_julia || true)"
if [ -z "$julia_bin" ]; then
    printf 'Julia was not found; installing it with Juliaup...\n'
    curl -fsSL https://install.julialang.org | sh -s -- --yes
    julia_bin="$(find_julia || true)"
    [ -n "$julia_bin" ] || die "Juliaup completed but 'julia' was not found; open a new shell and rerun this installer"
fi

"$julia_bin" -e "VERSION >= v\"${MIN_JULIA_VERSION}\" || error(\"Amalthea requires Julia ${MIN_JULIA_VERSION} or newer; found \$(VERSION)\")"

printf 'Installing Amalthea %s with %s...\n' "$version" "$julia_bin"
"$julia_bin" -e "using Pkg; Pkg.add(url=\"${PACKAGE_URL}\", rev=\"${version}\"); Pkg.precompile()"

printf '\nAmalthea %s is ready. Verify it with:\n\n' "$version"
printf '  julia -e %q\n' 'using Amalthea; Amalthea.backend_report()'
printf '\nRerun this installer to update to the latest release.\n'
