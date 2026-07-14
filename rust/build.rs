use std::env;
use std::path::PathBuf;
use std::process::Command;

fn brew_prefix(pkg: &str) -> String {
    let out = Command::new("brew")
        .arg("--prefix")
        .arg(pkg)
        .output()
        .unwrap_or_else(|_| panic!("failed to run `brew --prefix {pkg}`"));
    String::from_utf8(out.stdout).unwrap().trim().to_string()
}

fn main() {
    let ort_inc = env::var("ORT_INCLUDE_DIR")
        .expect("set ORT_INCLUDE_DIR to the ORT 1.27 prebuilt include dir");
    let mlxc = brew_prefix("mlx-c");
    let mlx = brew_prefix("mlx");
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());

    println!("cargo:rerun-if-changed=wrapper_ort.h");
    println!("cargo:rerun-if-changed=wrapper_mlx.h");
    println!("cargo:rerun-if-env-changed=ORT_INCLUDE_DIR");

    // --- ORT plugin-EP C ABI bindings (pure C header; pulls in onnxruntime_ep_c_api.h) ---
    bindgen::Builder::default()
        .header("wrapper_ort.h")
        .clang_arg(format!("-I{ort_inc}"))
        .allowlist_type("Ort.*")
        .allowlist_type("ONNX.*")
        .allowlist_function("Ort.*")
        .generate()
        .expect("ORT bindgen failed")
        .write_to_file(out.join("ort.rs"))
        .unwrap();

    // --- mlx-c bindings (bind the C API DIRECTLY, no mlx-rs crate) ---
    bindgen::Builder::default()
        .header("wrapper_mlx.h")
        .clang_arg(format!("-I{mlxc}/include"))
        .allowlist_function("mlx_.*")
        .allowlist_type("mlx_.*")
        .allowlist_var("MLX_.*")
        .generate()
        .expect("mlx-c bindgen failed")
        .write_to_file(out.join("mlx.rs"))
        .unwrap();

    // We call ORT purely through the OrtApi function-pointer table handed to
    // CreateEpFactories, so we do NOT link libonnxruntime. Only mlx-c + mlx + frameworks.
    println!("cargo:rustc-link-search=native={mlxc}/lib");
    println!("cargo:rustc-link-search=native={mlx}/lib");
    println!("cargo:rustc-link-lib=dylib=mlxc");
    println!("cargo:rustc-link-lib=dylib=mlx");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{mlxc}/lib");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{mlx}/lib");
    for fw in ["Metal", "Foundation", "QuartzCore", "Accelerate"] {
        println!("cargo:rustc-link-lib=framework={fw}");
    }
}
