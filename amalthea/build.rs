use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=src/kernels.cu");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=AMALTHEA_REQUIRE_CUDA_TESTS");
    println!("cargo:rerun-if-env-changed=NVCC");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let dest_path = out_dir.join("kernels.ptx");
    let require_cuda = env::var("AMALTHEA_REQUIRE_CUDA_TESTS").is_ok_and(|value| value == "1");

    let nvcc = find_nvcc();
    if let Err(message) = compile_or_fallback(nvcc.as_deref(), &dest_path, require_cuda) {
        panic!("{message}");
    }
}

fn find_nvcc() -> Option<PathBuf> {
    if let Some(nvcc) = env::var_os("NVCC") {
        return Some(PathBuf::from(nvcc));
    }
    // Prefer the explicitly restored local CUDA toolchain. `NVCC` remains an
    // override for CI and nonstandard installations; the compatibility
    // symlink is only a fallback for ordinary developer machines.
    let cuda_nvcc = PathBuf::from("/usr/local/cuda-13.3/bin/nvcc");
    if cuda_nvcc.exists() {
        Some(cuda_nvcc)
    } else if Path::new("/usr/local/cuda/bin/nvcc").exists() {
        Some(PathBuf::from("/usr/local/cuda/bin/nvcc"))
    } else if Command::new("nvcc").arg("--version").status().is_ok() {
        Some(PathBuf::from("nvcc"))
    } else {
        None
    }
}

fn compile_or_fallback(
    nvcc: Option<&Path>,
    dest_path: &Path,
    require_cuda: bool,
) -> Result<(), String> {
    let failure = match nvcc {
        Some(nvcc) => {
            let status = match std::fs::remove_file(dest_path) {
                Ok(()) => run_nvcc(nvcc, dest_path),
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                    run_nvcc(nvcc, dest_path)
                }
                Err(error) => return Err(format!("failed to clear prior PTX output: {error}")),
            };
            match status {
                Ok(status) if status.success() && is_real_ptx(dest_path) => return Ok(()),
                Ok(status) if status.success() => {
                    "nvcc reported success but did not produce real PTX".to_owned()
                }
                Ok(status) => format!("nvcc compilation failed with status {status}"),
                Err(error) => format!("failed to invoke nvcc: {error}"),
            }
        }
        None => "nvcc compiler not found".to_owned(),
    };

    if require_cuda {
        return Err(format!(
            "AMALTHEA_REQUIRE_CUDA_TESTS=1 requires nvcc and real PTX: {failure}"
        ));
    }

    println!("cargo:warning={failure}; creating dummy PTX for CPU-only build");
    write_dummy_ptx(dest_path).map_err(|error| format!("failed to write dummy PTX: {error}"))
}

fn run_nvcc(nvcc: &Path, dest_path: &Path) -> std::io::Result<std::process::ExitStatus> {
    Command::new(nvcc)
        .args(["--ptx", "src/kernels.cu", "-o"])
        .arg(dest_path)
        .status()
}

fn is_real_ptx(path: &Path) -> bool {
    std::fs::read_to_string(path).is_ok_and(|ptx| {
        ptx.contains(".version")
            && ptx.contains(".target")
            && ptx.contains(".address_size")
            && ptx.contains(".visible .entry")
    })
}

fn write_dummy_ptx(dest_path: &Path) -> std::io::Result<()> {
    // Keep include_str! valid for CPU-only development; strict CUDA mode never reaches this path.
    std::fs::write(dest_path, "// DUMMY PTX\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_ptx_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "amalthea-build-rs-{name}-{}-{}.ptx",
            std::process::id(),
            line!()
        ))
    }

    #[test]
    fn cpu_only_build_writes_dummy_ptx_without_nvcc() {
        let dest_path = test_ptx_path("cpu-fallback");
        let _ = std::fs::remove_file(&dest_path);

        compile_or_fallback(None, &dest_path, false).unwrap();
        assert_eq!(
            std::fs::read_to_string(&dest_path).unwrap(),
            "// DUMMY PTX\n"
        );
        assert!(!is_real_ptx(&dest_path));

        std::fs::remove_file(dest_path).unwrap();
    }

    #[test]
    fn strict_cuda_build_rejects_missing_nvcc() {
        let dest_path = test_ptx_path("strict-missing-nvcc");
        let _ = std::fs::remove_file(&dest_path);

        let error = compile_or_fallback(None, &dest_path, true).unwrap_err();
        assert!(error.contains("AMALTHEA_REQUIRE_CUDA_TESTS=1"));
        assert!(!dest_path.exists());
    }

    #[test]
    fn real_ptx_requires_nvcc_output_markers() {
        let dest_path = test_ptx_path("ptx-markers");
        std::fs::write(
            &dest_path,
            ".version 8.0\n.target sm_80\n.address_size 64\n.visible .entry kernel() {}\n",
        )
        .unwrap();
        assert!(is_real_ptx(&dest_path));

        std::fs::write(&dest_path, "// DUMMY PTX\n").unwrap();
        assert!(!is_real_ptx(&dest_path));

        std::fs::remove_file(dest_path).unwrap();
    }
}
