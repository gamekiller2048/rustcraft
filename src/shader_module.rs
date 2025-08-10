use ash::vk;
use std::{cell::RefCell, ptr, rc::Rc};

use crate::vulkan_allocator::VulkanAllocator;
use crate::vulkan_context::VulkanContext;

pub struct ShaderModule {}

impl ShaderModule {
    pub fn create_shader_module(
        context: &VulkanContext,
        bytes: &Vec<u8>,
        allocator: &Rc<RefCell<VulkanAllocator>>,
    ) -> vk::ShaderModule {
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
                .create_shader_module(
                    &create_info,
                    Some(&allocator.borrow_mut().get_allocation_callbacks()),
                )
                .unwrap()
        }
    }

    pub fn destroy_shader_module(
        context: &VulkanContext,
        shader: vk::ShaderModule,
        allocator: &Rc<RefCell<VulkanAllocator>>,
    ) {
        unsafe {
            context.device.destroy_shader_module(
                shader,
                Some(&allocator.borrow_mut().get_allocation_callbacks()),
            );
        }
    }
}
