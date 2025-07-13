use std::{process::Command};

fn main() {
    Command::new("D:/sdks/VulkanSDK/1.4.304.1/Bin/glslc.exe")
        .args(["src/shader.vert", "-o", "src/shader.vert.spv"])
        .status()
        .unwrap();

    Command::new("D:/sdks/VulkanSDK/1.4.304.1/Bin/glslc.exe")
            .args(["src/shader.frag", "-o", "src/shader.frag.spv"])
            .status()
            .unwrap();
}