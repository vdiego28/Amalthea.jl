// Exercise the CUDA build policy as a real Cargo integration-test target.
// Build scripts themselves are not compiled with the ordinary test harness,
// so including the small policy module here makes its `#[cfg(test)]` cases
// part of `cargo test` without duplicating the implementation.
#[allow(dead_code)]
mod build_script {
    include!("../build.rs");
}
