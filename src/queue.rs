use ash::vk;

use super::vulkan_context::VulkanContext;

#[derive(Clone, Copy)]
pub struct Queue {
    pub queue: vk::Queue,
    pub family_index: u32,
}

impl Queue {
    pub fn new(context: &VulkanContext, family_index: u32, queue_index: u32) -> Self {
        let queue = unsafe { context.device.get_device_queue(family_index, queue_index) };

        Self {
            queue,
            family_index,
        }
    }
}
