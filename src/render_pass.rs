use ash::vk;
use std::ptr;

use super::vulkan_context::VulkanContext;

pub struct RenderPass {
    pub render_pass: vk::RenderPass,
    pub color_attachment_index: u32,
    pub depth_attachment_index: u32,
}

impl RenderPass {
    pub fn new(
        context: &VulkanContext,
        attachments: &[vk::AttachmentDescription],
        subpasses: &[vk::SubpassDescription],
        dependencies: &[vk::SubpassDependency],
        color_attachment_index: u32,
        depth_attachment_index: u32,
    ) -> Self {
        let render_pass = Self::create_render_pass(context, attachments, subpasses, dependencies);

        Self {
            render_pass,
            color_attachment_index,
            depth_attachment_index,
        }
    }

    pub fn destroy(&self, context: &VulkanContext) {
        Self::destroy_render_pass(context, self.render_pass);
    }

    pub fn create_render_pass(
        context: &VulkanContext,
        attachments: &[vk::AttachmentDescription],
        subpasses: &[vk::SubpassDescription],
        dependencies: &[vk::SubpassDependency],
    ) -> vk::RenderPass {
        let render_pass_create_info = vk::RenderPassCreateInfo {
            s_type: vk::StructureType::RENDER_PASS_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::RenderPassCreateFlags::empty(),
            attachment_count: attachments.len() as u32,
            p_attachments: attachments.as_ptr(),
            subpass_count: subpasses.len() as u32,
            p_subpasses: subpasses.as_ptr(),
            p_dependencies: dependencies.as_ptr(),
            dependency_count: dependencies.len() as u32,
            ..Default::default()
        };

        unsafe {
            context
                .device
                .create_render_pass(&render_pass_create_info, None)
                .unwrap()
        }
    }

    pub fn destroy_render_pass(context: &VulkanContext, render_pass: vk::RenderPass) {
        unsafe {
            context.device.destroy_render_pass(render_pass, None);
        }
    }
}
