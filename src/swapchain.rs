use std::ptr;

use super::vulkan_context::VulkanContext;
use ash::vk;

use super::framebuffer::Framebuffer;
use super::image::Image;
use super::image_view::ImageView;
use super::render_pass::RenderPass;

pub struct Swapchain {
    pub swapchain: vk::SwapchainKHR,
    pub images: Vec<vk::Image>,
    pub image_views: Vec<vk::ImageView>,
    pub extent: vk::Extent2D,
    pub format: vk::SurfaceFormatKHR,
    pub present_mode: vk::PresentModeKHR,

    pub depth_image: vk::Image,
    pub depth_image_memory: vk::DeviceMemory,
    pub depth_image_view: vk::ImageView,
    pub depth_format: vk::Format,

    pub framebuffers: Vec<vk::Framebuffer>,

    pub sharing_mode: vk::SharingMode,
    pub graphics_queue_index: u32,
    pub present_queue_index: u32,
}

impl Swapchain {
    pub fn new(
        context: &VulkanContext,
        surface: vk::SurfaceKHR,
        format: vk::SurfaceFormatKHR,
        present_mode: vk::PresentModeKHR,
        extent: vk::Extent2D,
        depth_format: vk::Format,
        sharing_mode: vk::SharingMode,
        graphics_queue_index: u32,
        present_queue_index: u32,
        render_pass: &RenderPass,
    ) -> Self {
        let queue_family_indices: [u32; 2] = [graphics_queue_index, present_queue_index];

        let (swapchain, images) = Self::create_swapchain(
            context,
            surface,
            format,
            present_mode,
            extent,
            sharing_mode,
            &queue_family_indices,
            vk::SwapchainKHR::null(),
        );

        let image_views = Self::create_image_views(context, &images, format.format);

        let (depth_image, depth_image_memory, depth_image_view) = Self::create_depth_resources(
            context,
            &images,
            extent,
            depth_format,
            sharing_mode,
            &queue_family_indices,
        );

        let framebuffers =
            Self::create_framebuffers(context, extent, &image_views, depth_image_view, render_pass);

        Self {
            swapchain,
            images,
            image_views,
            extent,
            format,
            present_mode,
            depth_image,
            depth_image_memory,
            depth_image_view,
            depth_format,
            framebuffers,
            sharing_mode,
            graphics_queue_index,
            present_queue_index,
        }
    }

    pub fn destroy(&self, context: &VulkanContext) {
        for framebuffer in self.framebuffers.iter() {
            Framebuffer::destroy_framebuffer(&context, *framebuffer);
        }

        for image_view in self.image_views.iter() {
            ImageView::destroy_image_view(context, *image_view);
        }

        Self::destroy_swapchain(context, self.swapchain);

        ImageView::destroy_image_view(context, self.depth_image_view);
        Image::destroy_image(&context, self.depth_image);
        context.free_memory(self.depth_image_memory);
    }

    pub fn recreate(
        &mut self,
        context: &VulkanContext,
        surface: vk::SurfaceKHR,
        extent: vk::Extent2D,
        render_pass: &RenderPass,
    ) {
        let queue_family_indices: [u32; 2] = [self.graphics_queue_index, self.present_queue_index];

        self.extent = extent;

        Self::destroy_swapchain(context, self.swapchain);

        (self.swapchain, self.images) = Self::create_swapchain(
            context,
            surface,
            self.format,
            self.present_mode,
            self.extent,
            self.sharing_mode,
            &queue_family_indices,
            vk::SwapchainKHR::null(), // TODO: use
        );

        for image_view in self.image_views.iter() {
            ImageView::destroy_image_view(context, *image_view);
        }

        self.image_views = Self::create_image_views(context, &self.images, self.format.format);

        ImageView::destroy_image_view(context, self.depth_image_view);
        Image::destroy_image(&context, self.depth_image);
        context.free_memory(self.depth_image_memory);

        (
            self.depth_image,
            self.depth_image_memory,
            self.depth_image_view,
        ) = Self::create_depth_resources(
            context,
            &self.images,
            self.extent,
            self.depth_format,
            self.sharing_mode,
            &queue_family_indices,
        );

        for framebuffer in self.framebuffers.iter() {
            Framebuffer::destroy_framebuffer(&context, *framebuffer);
        }

        self.framebuffers = Self::create_framebuffers(
            context,
            self.extent,
            &self.image_views,
            self.depth_image_view,
            render_pass,
        );
    }

    fn create_depth_resources(
        context: &VulkanContext,
        images: &Vec<vk::Image>,
        extent: vk::Extent2D,
        depth_format: vk::Format,
        sharing_mode: vk::SharingMode,
        queue_family_indices: &[u32],
    ) -> (vk::Image, vk::DeviceMemory, vk::ImageView) {
        let (depth_image, depth_image_memory) = Image::create_image(
            &context,
            extent.width,
            extent.height,
            1,
            1,
            1,
            depth_format,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            sharing_mode,
            &queue_family_indices,
        );

        let depth_image_view = ImageView::create_image_view(
            context,
            depth_image,
            depth_format,
            vk::ImageViewType::TYPE_2D,
            vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::DEPTH,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            },
        );

        (depth_image, depth_image_memory, depth_image_view)
    }

    fn create_image_views(
        context: &VulkanContext,
        images: &Vec<vk::Image>,
        format: vk::Format,
    ) -> Vec<vk::ImageView> {
        let mut image_views: Vec<vk::ImageView> = Vec::with_capacity(images.len());

        for image in images.iter() {
            image_views.push(ImageView::create_image_view(
                context,
                *image,
                format,
                vk::ImageViewType::TYPE_2D,
                vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
            ));
        }

        image_views
    }

    fn create_framebuffers(
        context: &VulkanContext,
        extent: vk::Extent2D,
        image_views: &Vec<vk::ImageView>,
        depth_image_view: vk::ImageView,
        render_pass: &RenderPass,
    ) -> Vec<vk::Framebuffer> {
        let mut framebuffers = Vec::with_capacity(image_views.len());

        for image_view in image_views.iter() {
            let mut attachments: [vk::ImageView; 2] = [vk::ImageView::null(); 2];

            attachments[render_pass.color_attachment_index as usize] = *image_view;
            attachments[render_pass.depth_attachment_index as usize] = depth_image_view;

            framebuffers.push(Framebuffer::create_framebuffer(
                &context,
                render_pass.render_pass,
                &attachments,
                extent,
                1,
            ));
        }

        framebuffers
    }

    pub fn create_swapchain(
        context: &VulkanContext,
        surface: vk::SurfaceKHR,
        format: vk::SurfaceFormatKHR,
        present_mode: vk::PresentModeKHR,
        extent: vk::Extent2D,
        sharing_mode: vk::SharingMode,
        queue_family_indices: &[u32],
        old_swapchain: vk::SwapchainKHR,
    ) -> (vk::SwapchainKHR, Vec<vk::Image>) {
        let mut image_count: u32 = context.swapchain_details.capabilities.min_image_count + 1;
        if image_count > context.swapchain_details.capabilities.max_image_count {
            image_count = context.swapchain_details.capabilities.min_image_count;
        }

        let create_info = vk::SwapchainCreateInfoKHR {
            s_type: vk::StructureType::SWAPCHAIN_CREATE_INFO_KHR,
            p_next: ptr::null(),
            min_image_count: image_count,
            image_format: format.format,
            flags: vk::SwapchainCreateFlagsKHR::empty(),
            surface: surface,
            image_color_space: format.color_space,
            image_extent: extent,
            image_array_layers: 1,
            image_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
            pre_transform: context.swapchain_details.capabilities.current_transform,
            composite_alpha: vk::CompositeAlphaFlagsKHR::OPAQUE,
            present_mode: present_mode,
            clipped: vk::TRUE,
            old_swapchain: old_swapchain,
            image_sharing_mode: sharing_mode,
            queue_family_index_count: queue_family_indices.len() as u32,
            p_queue_family_indices: queue_family_indices.as_ptr(),
            ..Default::default()
        };

        let swapchain: vk::SwapchainKHR = unsafe {
            context
                .swapchain_loader
                .create_swapchain(&create_info, None)
                .unwrap()
        };

        let swap_chain_images: Vec<vk::Image> = unsafe {
            context
                .swapchain_loader
                .get_swapchain_images(swapchain)
                .unwrap()
        };

        (swapchain, swap_chain_images)
    }

    pub fn destroy_swapchain(context: &VulkanContext, swapchain: vk::SwapchainKHR) {
        unsafe {
            context.swapchain_loader.destroy_swapchain(swapchain, None);
        }
    }
}
