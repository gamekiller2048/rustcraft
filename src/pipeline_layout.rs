use ash::vk;
use std::ptr;

use super::vulkan_context::VulkanContext;

pub struct PipelineLayout {}

impl PipelineLayout {
    pub fn create_pipeline_layout(
        context: &VulkanContext,
        descriptor_set_layouts: &[vk::DescriptorSetLayout],
        push_constants: &[vk::PushConstantRange],
    ) -> vk::PipelineLayout {
        let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo {
            s_type: vk::StructureType::PIPELINE_LAYOUT_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::PipelineLayoutCreateFlags::empty(),
            set_layout_count: descriptor_set_layouts.len() as u32,
            p_set_layouts: descriptor_set_layouts.as_ptr(),
            p_push_constant_ranges: push_constants.as_ptr(),
            push_constant_range_count: push_constants.len() as u32,
            ..Default::default()
        };

        unsafe {
            context
                .device
                .create_pipeline_layout(&pipeline_layout_create_info, None)
                .unwrap()
        }
    }

    pub fn destroy_pipeline_layout(context: &VulkanContext, pipeline_layout: vk::PipelineLayout) {
        unsafe {
            context
                .device
                .destroy_pipeline_layout(pipeline_layout, None);
        }
    }
}
