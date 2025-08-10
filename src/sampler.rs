use ash::vk;
use std::cell::RefCell;
use std::rc::Rc;

use crate::vulkan_allocator::VulkanAllocator;
use crate::vulkan_context::VulkanContext;

pub struct Sampler {}

impl Sampler {
    pub fn destroy_sampler(
        context: &VulkanContext,
        sampler: vk::Sampler,
        allocator: &Rc<RefCell<VulkanAllocator>>,
    ) {
        unsafe {
            context.device.destroy_sampler(
                sampler,
                Some(&allocator.borrow_mut().get_allocation_callbacks()),
            );
        }
    }
}
