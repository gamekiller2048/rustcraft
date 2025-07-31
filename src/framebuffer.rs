use ash::vk;
use std::ptr;

use super::vulkan_context::VulkanContext;

pub struct Framebuffer;

impl Framebuffer {
    pub fn create_framebuffer(
        context: &VulkanContext,
        render_pass: vk::RenderPass,
        attachments: &[vk::ImageView],
        extent: vk::Extent2D,
        layers: u32,
    ) -> vk::Framebuffer {
        let create_info = vk::FramebufferCreateInfo {
            s_type: vk::StructureType::FRAMEBUFFER_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::FramebufferCreateFlags::empty(),
            render_pass: render_pass,
            attachment_count: attachments.len() as u32,
            p_attachments: attachments.as_ptr(),
            width: extent.width,
            height: extent.height,
            layers: layers,
            ..Default::default()
        };

        unsafe {
            context
                .device
                .create_framebuffer(&create_info, None)
                .unwrap()
        }
    }

    pub fn destroy_framebuffer(context: &VulkanContext, framebuffer: vk::Framebuffer) {
        unsafe {
            context.device.destroy_framebuffer(framebuffer, None);
        }
    }
}
