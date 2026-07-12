use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=src/kernels.cu");
    println!("cargo:rerun-if-changed=build.rs");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let dest_path = out_dir.join("kernels.ptx");

    // Try to locate nvcc
    let nvcc_path = "/usr/local/cuda/bin/nvcc";
    let has_nvcc = std::path::Path::new(nvcc_path).exists()
        || Command::new("nvcc").arg("--version").status().is_ok();

    if has_nvcc {
        let nvcc = if std::path::Path::new(nvcc_path).exists() {
            nvcc_path
        } else {
            "nvcc"
        };

        let status = Command::new(nvcc)
            .args(["--ptx", "src/kernels.cu", "-o"])
            .arg(&dest_path)
            .status();

        match status {
            Ok(s) if s.success() => {
                println!("cargo:warning=Successfully compiled kernels.cu to PTX");
                return;
            }
            other => {
                println!(
                    "cargo:warning=nvcc compilation failed or returned error: {:?}",
                    other
                );
            }
        }
    } else {
        println!("cargo:warning=nvcc compiler not found. Creating dummy PTX.");
    }

    // Write dummy PTX so inclusion still succeeds on CPU-only systems
    std::fs::write(&dest_path, "// DUMMY PTX\n").unwrap();
}
