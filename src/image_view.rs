use ash::vk;
use std::ptr;

use super::vulkan_context::VulkanContext;

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
            ..Default::default()
        };

        unsafe {
            context
                .device
                .create_image_view(&create_info, None)
                .unwrap()
        }
    }

    pub fn destroy_image_view(context: &VulkanContext, image_view: vk::ImageView) {
        unsafe {
            context.device.destroy_image_view(image_view, None);
        }
    }

    pub fn new(
        context: &VulkanContext,
        image: vk::Image,
        format: vk::Format,
        view_type: vk::ImageViewType,
        subresource_range: vk::ImageSubresourceRange,
    ) -> Self {
        let image_view =
            Self::create_image_view(context, image, format, view_type, subresource_range);

        Self { image_view }
    }

    pub fn destroy(&self, context: &VulkanContext) {
        Self::destroy_image_view(context, self.image_view);
    }
}
