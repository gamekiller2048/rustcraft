use ash::vk;
use std::{marker::PhantomData, ptr, sync::Arc};

use crate::vulkan_allocator::VulkanAllocator;
use crate::vulkan_context::VulkanContext;

pub struct ImageView {
    pub image_view: vk::ImageView,
}

impl ImageView {
    pub fn create_image_view(
        context: &VulkanContext,
        image: vk::Image,
        format: vk::Format,
        view_type: vk::ImageViewType,
        subresource_range: vk::ImageSubresourceRange,
        allocator: &Arc<VulkanAllocator>,
    ) -> vk::ImageView {
        let create_info = vk::ImageViewCreateInfo {
            s_type: vk::StructureType::IMAGE_VIEW_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::ImageViewCreateFlags::empty(),
            image: image,
            view_type: view_type,
            format: format,
            components: vk::ComponentMapping {
                r: vk::ComponentSwizzle::R,
                g: vk::ComponentSwizzle::G,
                b: vk::ComponentSwizzle::B,
                a: vk::ComponentSwizzle::A,
            },
            subresource_range: subresource_range,
            _marker: PhantomData,
        };

        unsafe {
            context
                .device
                .create_image_view(&create_info, Some(&allocator.callbacks))
                .unwrap()
        }
    }

    pub fn destroy_image_view(
        context: &VulkanContext,
        image_view: vk::ImageView,
        allocator: &Arc<VulkanAllocator>,
    ) {
        unsafe {
            context
                .device
                .destroy_image_view(image_view, Some(&allocator.callbacks));
        }
    }

    pub fn new(
        context: &VulkanContext,
        image: vk::Image,
        format: vk::Format,
        view_type: vk::ImageViewType,
        subresource_range: vk::ImageSubresourceRange,
        allocator: &Arc<VulkanAllocator>,
    ) -> Self {
        let image_view = Self::create_image_view(
            context,
            image,
            format,
            view_type,
            subresource_range,
            allocator,
        );

        Self { image_view }
    }

    pub fn destroy(&self, context: &VulkanContext, allocator: &Arc<VulkanAllocator>) {
        Self::destroy_image_view(context, self.image_view, allocator);
    }
}
