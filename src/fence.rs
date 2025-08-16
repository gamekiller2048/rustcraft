use ash::vk;
use std::sync::Arc;

use crate::vulkan_allocator::VulkanAllocator;
use crate::vulkan_context::VulkanContext;

pub struct Fence {}

impl Fence {
    pub fn destroy_fence(
        context: &VulkanContext,
        fence: vk::Fence,
        allocator: &Arc<VulkanAllocator>,
    ) {
        unsafe {
            context
                .device
                .destroy_fence(fence, Some(&allocator.callbacks));
        }
    }
}
