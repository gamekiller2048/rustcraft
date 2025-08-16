use ash::vk;
use std::sync::Arc;

use crate::vulkan_allocator::VulkanAllocator;
use crate::vulkan_context::VulkanContext;

pub struct Semaphore {}

impl Semaphore {
    pub fn destroy_semaphore(
        context: &VulkanContext,
        semaphore: vk::Semaphore,
        allocator: &Arc<VulkanAllocator>,
    ) {
        unsafe {
            context
                .device
                .destroy_semaphore(semaphore, Some(&allocator.callbacks));
        }
    }
}
