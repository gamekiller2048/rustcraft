use ash::vk;
use std::cell::RefCell;
use std::rc::Rc;

use crate::vulkan_allocator::VulkanAllocator;
use crate::vulkan_context::VulkanContext;

pub struct Fence {}

impl Fence {
    pub fn destroy_fence(
        context: &VulkanContext,
        fence: vk::Fence,
        allocator: &Rc<RefCell<VulkanAllocator>>,
    ) {
        unsafe {
            context.device.destroy_fence(
                fence,
                Some(&allocator.borrow_mut().get_allocation_callbacks()),
            );
        }
    }
}
