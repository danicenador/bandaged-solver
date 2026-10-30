//! On the windows-gnu toolchain, tell the linker where the interpreter's `python3.lib` lives
//! (see .cargo/config.toml for why raw-dylib linking is disabled).

use std::process::Command;

fn main() {
    println!("cargo:rerun-if-env-changed=PYO3_PYTHON");
    let python_feature = std::env::var_os("CARGO_FEATURE_PYTHON").is_some();
    let target = std::env::var("TARGET").unwrap_or_default();
    if !python_feature || !target.ends_with("windows-gnu") {
        return;
    }
    let python = std::env::var("PYO3_PYTHON").unwrap_or_else(|_| "python".into());
    let out = Command::new(&python).args(["-c", "import sys; print(sys.base_prefix)"]).output();
    if let Ok(out) = out
        && out.status.success()
    {
        let prefix = String::from_utf8_lossy(&out.stdout).trim().to_string();
        println!("cargo:rustc-link-search=native={}/libs", prefix);
    }
}
