use ash::vk;
use std::ptr;

use super::vulkan_context::VulkanContext;

pub struct ShaderModule {}

impl ShaderModule {
    pub fn create_shader_module(context: &VulkanContext, bytes: &Vec<u8>) -> vk::ShaderModule {
        let create_info = vk::ShaderModuleCreateInfo {
            s_type: vk::StructureType::SHADER_MODULE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::ShaderModuleCreateFlags::empty(),
            code_size: bytes.len(),
            p_code: bytes.as_ptr() as *const u32,
            ..Default::default()
        };

        unsafe {
            context
                .device
                .create_shader_module(&create_info, None)
                .unwrap()
        }
    }

    pub fn destroy_shader_module(context: &VulkanContext, shader: vk::ShaderModule) {
        unsafe {
            context.device.destroy_shader_module(shader, None);
        }
    }
}
