use ash::vk;
use std::ptr;

use super::vulkan_context::VulkanContext;

pub struct DescriptorPool;

impl DescriptorPool {
    pub fn create_descriptor_pool(
        context: &VulkanContext,
        pool_sizes: &[vk::DescriptorPoolSize],
        max_sets: u32,
    ) -> vk::DescriptorPool {
        let create_info = vk::DescriptorPoolCreateInfo {
            s_type: vk::StructureType::DESCRIPTOR_POOL_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::DescriptorPoolCreateFlags::empty(),
            pool_size_count: pool_sizes.len() as u32,
            p_pool_sizes: pool_sizes.as_ptr(),
            max_sets: max_sets,
            ..Default::default()
        };

        unsafe {
            context
                .device
                .create_descriptor_pool(&create_info, None)
                .unwrap()
        }
    }

    pub fn destroy_descriptor_pool(context: &VulkanContext, descriptor_pool: vk::DescriptorPool) {
        unsafe {
            context
                .device
                .destroy_descriptor_pool(descriptor_pool, None);
        }
    }

    pub fn create_descriptor_sets(
        context: &VulkanContext,
        descriptor_pool: vk::DescriptorPool,
        layouts: &[vk::DescriptorSetLayout],
    ) -> Vec<vk::DescriptorSet> {
        let allocate_info = vk::DescriptorSetAllocateInfo {
            s_type: vk::StructureType::DESCRIPTOR_SET_ALLOCATE_INFO,
            p_next: ptr::null(),
            descriptor_pool: descriptor_pool,
            descriptor_set_count: layouts.len() as u32,
            p_set_layouts: layouts.as_ptr(),
            ..Default::default()
        };

        unsafe {
            context
                .device
                .allocate_descriptor_sets(&allocate_info)
                .unwrap()
        }
    }

    pub fn write_descriptors(
        context: &VulkanContext,
        writes: &[vk::WriteDescriptorSet],
        copies: &[vk::CopyDescriptorSet],
    ) {
        unsafe {
            context.device.update_descriptor_sets(&writes, copies);
        }
    }
}
