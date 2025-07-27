use std::ptr;
use ash::vk;

use super::vulkan_context::VulkanContext;

pub struct Command<'a> {
    pub context: &'a VulkanContext,
    pub command_buffer: vk::CommandBuffer
}

impl<'a> Command<'a> {
    pub fn reset(&self) {
        unsafe {
            self.context.device.reset_command_buffer(self.command_buffer, vk::CommandBufferResetFlags::empty()).unwrap();
        }
    }
    
    pub fn begin(&self) {
        let begin_info = vk::CommandBufferBeginInfo {
            s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
            p_next: ptr::null(),
            flags: vk::CommandBufferUsageFlags::empty(),
            p_inheritance_info: ptr::null(),
            ..Default::default()
        };

        unsafe {
            self.context.device.begin_command_buffer(self.command_buffer, &begin_info).unwrap()
        };
    }

    pub fn end(&self) {
        unsafe {
            self.context.device.end_command_buffer(self.command_buffer).unwrap()
        };
    }

    pub fn begin_render_pass(&self, render_pass: vk::RenderPass, framebuffer: vk::Framebuffer, render_area: &vk::Rect2D, clear_values: &[vk::ClearValue], contents: vk::SubpassContents) {
        let begin_info = vk::RenderPassBeginInfo {
            s_type: vk::StructureType::RENDER_PASS_BEGIN_INFO,
            p_next: ptr::null(),
            render_pass: render_pass,
            framebuffer: framebuffer,
            render_area: *render_area,
            clear_value_count: clear_values.len() as u32,
            p_clear_values: clear_values.as_ptr(),
            ..Default::default()
        };

        unsafe {
            self.context.device.cmd_begin_render_pass(self.command_buffer, &begin_info, contents)
        };
    }

    pub fn end_render_pass(&self) {
        unsafe {
            self.context.device.cmd_end_render_pass(self.command_buffer)
        };
    }

    pub fn bind_pipeline(&self, pipeline_bind_point: vk::PipelineBindPoint, graphics_pipeline: vk::Pipeline) {
        unsafe {
            self.context.device.cmd_bind_pipeline(self.command_buffer, pipeline_bind_point, graphics_pipeline)
        };
    }

    pub fn bind_vertex_buffers(&self, first_binding: u32, vertex_buffers: &[vk::Buffer], offsets: &[vk::DeviceSize]) {
        unsafe {
            self.context.device.cmd_bind_vertex_buffers(self.command_buffer, first_binding, vertex_buffers, offsets)
        };
    }

    pub fn bind_index_buffer(&self, index_buffers: vk::Buffer, offsets: vk::DeviceSize, index_type: vk::IndexType) {
        unsafe {
            self.context.device.cmd_bind_index_buffer(self.command_buffer, index_buffers, offsets, index_type)
        };
    }
    
    pub fn set_viewports_scissors(&self, viewports: &[vk::Viewport], scissors: &[vk::Rect2D]) {
        unsafe {
            self.context.device.cmd_set_viewport(self.command_buffer, 0, viewports);
            self.context.device.cmd_set_scissor(self.command_buffer, 0, scissors);
        };
    }
}
