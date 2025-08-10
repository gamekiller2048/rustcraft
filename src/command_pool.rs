use ash::vk;
use std::{cell::RefCell, ptr, rc::Rc};

use crate::vulkan_allocator::VulkanAllocator;
use crate::vulkan_context::VulkanContext;

pub struct CommandPool {
    pub command_pool: vk::CommandPool,
    pub queue: vk::Queue,
}

impl CommandPool {
    pub fn new(
        context: &VulkanContext,
        flags: vk::CommandPoolCreateFlags,
        queue: vk::Queue,
        queue_family_index: u32,
        allocator: &Rc<RefCell<VulkanAllocator>>,
    ) -> Self {
        let command_pool =
            Self::create_command_pool(context, flags, queue, queue_family_index, allocator);

        Self {
            command_pool,
            queue,
        }
    }

    pub fn free_command_buffers(
        &self,
        context: &VulkanContext,
        command_buffers: &[vk::CommandBuffer],
    ) {
        unsafe {
            context
                .device
                .free_command_buffers(self.command_pool, command_buffers);
        }
    }

    pub fn destroy(&self, context: &VulkanContext, allocator: &Rc<RefCell<VulkanAllocator>>) {
        Self::destroy_command_pool(context, self.command_pool, allocator);
    }

    pub fn submit(
        &self,
        context: &VulkanContext,
        command_buffers: &[vk::CommandBuffer],
        wait_semaphores: &[vk::Semaphore],
        wait_stages: &[vk::PipelineStageFlags],
        signal_semaphores: &[vk::Semaphore],
        fence: vk::Fence,
    ) {
        let submit_info = vk::SubmitInfo {
            s_type: vk::StructureType::SUBMIT_INFO,
            p_next: ptr::null(),
            wait_semaphore_count: wait_semaphores.len() as u32,
            p_wait_semaphores: wait_semaphores.as_ptr(),
            p_wait_dst_stage_mask: wait_stages.as_ptr(),
            command_buffer_count: command_buffers.len() as u32,
            p_command_buffers: command_buffers.as_ptr(),
            signal_semaphore_count: signal_semaphores.len() as u32,
            p_signal_semaphores: signal_semaphores.as_ptr(),
            ..Default::default()
        };

        unsafe {
            context
                .device
                .queue_submit(self.queue, std::slice::from_ref(&submit_info), fence)
                .unwrap();
        }
    }

    pub fn wait_queue(&self, context: &VulkanContext) {
        unsafe {
            context.device.queue_wait_idle(self.queue).unwrap();
        }
    }

    pub fn create_command_pool(
        context: &VulkanContext,
        flags: vk::CommandPoolCreateFlags,
        queue: vk::Queue,
        queue_family_index: u32,
        allocator: &Rc<RefCell<VulkanAllocator>>,
    ) -> vk::CommandPool {
        let create_info = vk::CommandPoolCreateInfo {
            s_type: vk::StructureType::COMMAND_POOL_CREATE_INFO,
            p_next: ptr::null(),
            flags: flags,
            queue_family_index: queue_family_index,
            ..Default::default()
        };

        let command_pool = unsafe {
            context
                .device
                .create_command_pool(
                    &create_info,
                    Some(&allocator.borrow_mut().get_allocation_callbacks()),
                )
                .unwrap()
        };

        command_pool
    }

    pub fn destroy_command_pool(
        context: &VulkanContext,
        command_pool: vk::CommandPool,
        allocator: &Rc<RefCell<VulkanAllocator>>,
    ) {
        unsafe {
            context.device.destroy_command_pool(
                command_pool,
                Some(&allocator.borrow_mut().get_allocation_callbacks()),
            );
        }
    }
}
