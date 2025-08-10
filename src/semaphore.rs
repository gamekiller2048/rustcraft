use ash::vk;
use std::cell::RefCell;
use std::rc::Rc;

use crate::vulkan_allocator::VulkanAllocator;
use crate::vulkan_context::VulkanContext;

pub struct Semaphore {}

impl Semaphore {
    pub fn destroy_semaphore(
        context: &VulkanContext,
        semaphore: vk::Semaphore,
        allocator: &Rc<RefCell<VulkanAllocator>>,
    ) {
        unsafe {
            context.device.destroy_semaphore(
                semaphore,
                Some(&allocator.borrow_mut().get_allocation_callbacks()),
            );
        }
    }
}
