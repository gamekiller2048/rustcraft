use ash::vk;
use std::ptr;

use super::vulkan_context::VulkanContext;

pub struct CommandBuffer {
    pub command_buffer: vk::CommandBuffer,
}

impl CommandBuffer {
    pub fn new(
        context: &VulkanContext,
        command_pool: vk::CommandPool,
        level: vk::CommandBufferLevel,
    ) -> Self {
        let command_buffer = Self::create_command_buffers(context, command_pool, level, 1)[0];

        Self { command_buffer }
    }

    pub fn create_command_buffers(
        context: &VulkanContext,
        command_pool: vk::CommandPool,
        level: vk::CommandBufferLevel,
        num_buffers: u32,
    ) -> Vec<vk::CommandBuffer> {
        let allocate_info = vk::CommandBufferAllocateInfo {
            s_type: vk::StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
            p_next: ptr::null(),
            command_pool: command_pool,
            level: level,
            command_buffer_count: num_buffers,
            ..Default::default()
        };

        unsafe {
            context
                .device
                .allocate_command_buffers(&allocate_info)
                .unwrap()
        }
    }

    pub fn reset(&self, context: &VulkanContext) {
        unsafe {
            context
                .device
                .reset_command_buffer(self.command_buffer, vk::CommandBufferResetFlags::empty())
                .unwrap();
        }
    }

    pub fn begin(&self, context: &VulkanContext, flags: vk::CommandBufferUsageFlags) {
        let begin_info = vk::CommandBufferBeginInfo {
            s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
            p_next: ptr::null(),
            flags: flags,
            p_inheritance_info: ptr::null(),
            ..Default::default()
        };

        unsafe {
            context
                .device
                .begin_command_buffer(self.command_buffer, &begin_info)
                .unwrap();
        }
    }

    pub fn end(&self, context: &VulkanContext) {
        unsafe {
            context
                .device
                .end_command_buffer(self.command_buffer)
                .unwrap();
        }
    }

    pub fn begin_render_pass(
        &self,
        context: &VulkanContext,
        render_pass: vk::RenderPass,
        framebuffer: vk::Framebuffer,
        render_area: &vk::Rect2D,
        clear_values: &[vk::ClearValue],
        contents: vk::SubpassContents,
    ) {
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
            context
                .device
                .cmd_begin_render_pass(self.command_buffer, &begin_info, contents);
        }
    }

    pub fn end_render_pass(&self, context: &VulkanContext) {
        unsafe { context.device.cmd_end_render_pass(self.command_buffer) };
    }

    pub fn bind_pipeline(
        &self,
        context: &VulkanContext,
        pipeline_bind_point: vk::PipelineBindPoint,
        graphics_pipeline: vk::Pipeline,
    ) {
        unsafe {
            context.device.cmd_bind_pipeline(
                self.command_buffer,
                pipeline_bind_point,
                graphics_pipeline,
            );
        }
    }

    pub fn bind_vertex_buffers(
        &self,
        context: &VulkanContext,
        first_binding: u32,
        vertex_buffers: &[vk::Buffer],
        offsets: &[vk::DeviceSize],
    ) {
        unsafe {
            context.device.cmd_bind_vertex_buffers(
                self.command_buffer,
                first_binding,
                vertex_buffers,
                offsets,
            );
        }
    }

    pub fn bind_index_buffer(
        &self,
        context: &VulkanContext,
        index_buffers: vk::Buffer,
        offset: vk::DeviceSize,
        index_type: vk::IndexType,
    ) {
        unsafe {
            context.device.cmd_bind_index_buffer(
                self.command_buffer,
                index_buffers,
                offset,
                index_type,
            );
        }
    }

    pub fn set_viewports_scissors(
        &self,
        context: &VulkanContext,
        viewports: &[vk::Viewport],
        scissors: &[vk::Rect2D],
    ) {
        unsafe {
            context
                .device
                .cmd_set_viewport(self.command_buffer, 0, viewports);
            context
                .device
                .cmd_set_scissor(self.command_buffer, 0, scissors);
        }
    }

    pub fn pipeline_barrier(
        &self,
        context: &VulkanContext,
        src_stage: vk::PipelineStageFlags,
        dst_stage: vk::PipelineStageFlags,
        memory_barriers: &[vk::MemoryBarrier],
        buffer_memory_barriers: &[vk::BufferMemoryBarrier],
        image_memory_barriers: &[vk::ImageMemoryBarrier],
    ) {
        unsafe {
            context.device.cmd_pipeline_barrier(
                self.command_buffer,
                src_stage,
                dst_stage,
                vk::DependencyFlags::empty(),
                memory_barriers,
                buffer_memory_barriers,
                image_memory_barriers,
            );
        }
    }
}
