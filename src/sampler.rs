use ash::vk;
use std::sync::Arc;

use crate::vulkan_allocator::VulkanAllocator;
use crate::vulkan_context::VulkanContext;

pub struct Sampler {}

impl Sampler {
    pub fn destroy_sampler(
        context: &VulkanContext,
        sampler: vk::Sampler,
        allocator: &Arc<VulkanAllocator>,
    ) {
        unsafe {
            context
                .device
                .destroy_sampler(sampler, Some(&allocator.callbacks));
        }
    }
}
