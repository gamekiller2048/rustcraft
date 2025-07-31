use ash::vk;

use super::vulkan_context::VulkanContext;

pub struct Pipeline {}

impl Pipeline {
    pub fn destroy_pipeline(context: &VulkanContext, pipeline: vk::Pipeline) {
        unsafe {
            context.device.destroy_pipeline(pipeline, None);
        }
    }
}
