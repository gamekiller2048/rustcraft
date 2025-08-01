use std::{process::Command};

fn main() {
    Command::new("D:/sdks/VulkanSDK/1.4.304.1/Bin/glslc.exe")
        .args(["res/shader.vert", "-o", "res/shader.vert.spv"])
        .status()
        .unwrap();

    Command::new("D:/sdks/VulkanSDK/1.4.304.1/Bin/glslc.exe")
            .args(["res/shader.frag", "-o", "res/shader.frag.spv"])
            .status()
            .unwrap();
}