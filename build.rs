use std::{env, process::Command};

fn main() {
    // TODO: this does not work
    unsafe {
        env::set_var("RUST_LOG", "trace");
        env::set_var("CARGO_FEATURE_validation", "1");
    }

    Command::new("D:/sdks/VulkanSDK/1.4.304.1/Bin/glslc.exe")
        .args(["res/shader.vert", "-o", "res/shader.vert.spv"])
        .status()
        .unwrap();

    Command::new("D:/sdks/VulkanSDK/1.4.304.1/Bin/glslc.exe")
        .args(["res/shader.frag", "-o", "res/shader.frag.spv"])
        .status()
        .unwrap();

    Command::new("D:/sdks/VulkanSDK/1.4.304.1/Bin/glslc.exe")
        .args(["res/shader.comp", "-o", "res/shader.comp.spv"])
        .status()
        .unwrap();
}
