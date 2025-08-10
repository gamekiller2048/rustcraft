use ash::vk;
use std::{cell::RefCell, ptr, rc::Rc};

use crate::vulkan_allocator::VulkanAllocator;
use crate::vulkan_context::VulkanContext;

pub struct Frame {
    pub swap_image_available_semaphore: vk::Semaphore,
    pub render_finished_semaphore: vk::Semaphore,
    pub in_flight_fence: vk::Fence,
}

pub struct FramesInFlight {
    pub frames: Vec<Frame>,
    pub curr_frame: usize,
}

impl FramesInFlight {
    pub fn new(
        context: &VulkanContext,
        max_frames: usize,
        allocator: &Rc<RefCell<VulkanAllocator>>,
    ) -> Self {
        let semaphore_create_info = vk::SemaphoreCreateInfo {
            s_type: vk::StructureType::SEMAPHORE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::SemaphoreCreateFlags::empty(),
            ..Default::default()
        };

        let fence_create_info = vk::FenceCreateInfo {
            s_type: vk::StructureType::FENCE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::FenceCreateFlags::SIGNALED,
            ..Default::default()
        };

        let mut frames: Vec<Frame> = Vec::with_capacity(max_frames);

        for i in 0..max_frames {
            let swap_image_available_semaphore = unsafe {
                context
                    .device
                    .create_semaphore(
                        &semaphore_create_info,
                        Some(&allocator.borrow_mut().get_allocation_callbacks()),
                    )
                    .unwrap()
            };
            let render_finished_semaphore = unsafe {
                context
                    .device
                    .create_semaphore(
                        &semaphore_create_info,
                        Some(&allocator.borrow_mut().get_allocation_callbacks()),
                    )
                    .unwrap()
            };
            let in_flight_fence = unsafe {
                context
                    .device
                    .create_fence(
                        &fence_create_info,
                        Some(&allocator.borrow_mut().get_allocation_callbacks()),
                    )
                    .unwrap()
            };

            frames.push(Frame {
                swap_image_available_semaphore,
                render_finished_semaphore,
                in_flight_fence,
            });
        }

        Self {
            frames,
            curr_frame: 0,
        }
    }

    pub fn step(&mut self) {
        self.curr_frame = (self.curr_frame + 1) % self.frames.len();
    }

    pub fn current_frame(&self) -> &Frame {
        &self.frames[self.curr_frame]
    }
}
