#![feature(allocator_api)]
#![allow(dead_code, unused_variables)]
use ::image::{EncodableLayout, ImageReader};
use ash::{khr, vk};
use nalgebra_glm as glm;
use std::{
    cell::RefCell,
    ffi::{c_char, c_void},
    fs, ptr,
    rc::Rc,
    slice,
    time::{Duration, SystemTime},
    u32,
};
use winit::raw_window_handle::HasWindowHandle;

#[cfg(feature = "validation")]
use ash::ext;

mod instance;
#[cfg(feature = "validation")]
use instance::check_validation_layer_support;

mod vulkan_context;
use vulkan_context::{QueueFamilyIndices, SwapChainSupportDetails, VulkanContext};

mod queue;
use queue::Queue;

mod swapchain;
use swapchain::Swapchain;

mod descriptor_set_layout;
use descriptor_set_layout::{DescriptorSetLayout, create_descriptor_image_sampler_write};
use descriptor_set_layout::{DescriptorSetLayoutBuilder, create_descriptor_uniform_buffer_write};

mod descriptor_pool;
use descriptor_pool::DescriptorPool;

mod render_pass;
use render_pass::RenderPass;

mod buffer;
use buffer::Buffer;

mod command_pool;
use command_pool::CommandPool;

mod shader_module;
use shader_module::ShaderModule;

mod image;
use image::Image;

mod pipeline_layout;
use pipeline_layout::PipelineLayout;

mod graphics_pipeline;
use graphics_pipeline::GraphicsPipeline;

mod compute_pipeline;
use compute_pipeline::ComputePipeline;

mod image_view;
use image_view::ImageView;

mod framebuffer;

mod command_buffer;
use command_buffer::CommandBuffer;

mod vertex;
use vertex::Vertex;

mod frames_in_flight;
use frames_in_flight::FramesInFlight;

mod cube;
use cube::{INDICES, MyVertex, VERTICES};

mod sampler;
use sampler::Sampler;

mod semaphore;
use semaphore::Semaphore;

mod fence;
use fence::Fence;

mod vulkan_allocator;
use vulkan_allocator::VulkanAllocator;

struct TransformationData {
    proj_view: glm::Mat4,
    model: glm::Mat4,
}

const MAX_FRAMES_IN_FLIGHT: usize = 1;

struct Renderer {
    entry: ash::Entry,
    vulkan_allocator: Rc<RefCell<VulkanAllocator>>,
    context: VulkanContext,

    #[cfg(feature = "validation")]
    debug_utils_loader: ext::debug_utils::Instance,

    #[cfg(feature = "validation")]
    debug_messager: vk::DebugUtilsMessengerEXT,

    surface_loader: khr::surface::Instance,
    surface: vk::SurfaceKHR,

    graphics_queue: Queue,
    present_queue: Queue,
    transfer_queue: Queue,
    compute_queue: Queue,

    swapchain: Swapchain,

    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,

    uniform_buffers: Vec<Buffer>,
    sampler: vk::Sampler,

    render_pass: RenderPass,

    pipeline_layout: vk::PipelineLayout,
    graphics_pipeline: GraphicsPipeline,
    // compute_pipeline: ComputePipeline,
    vertex_buffer: Buffer,
    index_buffer: Buffer,

    texture_image: Image,
    texture_image_view: ImageView,

    graphics_command_pool: CommandPool,
    draw_command_buffers: Vec<vk::CommandBuffer>,

    shader_storage_buffers: Vec<Buffer>,

    transfer_command_pool: CommandPool,
    compute_command_pool: CommandPool,

    frames: FramesInFlight,
    start_time: Duration,
}

impl Renderer {
    fn score_physical_device(
        properties: vk::PhysicalDeviceProperties,
        features: vk::PhysicalDeviceFeatures,
        queue_family_indices: &QueueFamilyIndices,
        swapchain_details: &SwapChainSupportDetails,
    ) -> u32 {
        let is_swapchain_adequate =
            !swapchain_details.formats.is_empty() && !swapchain_details.present_modes.is_empty();

        let base = !queue_family_indices.graphics.is_empty()
            && !queue_family_indices.present.is_empty()
            && !queue_family_indices.compute.is_empty()
            && is_swapchain_adequate
            && features.sampler_anisotropy == vk::TRUE;

        if !base {
            return 0;
        }

        return 1
            + !queue_family_indices.transfer_only.is_empty() as u32
            + !queue_family_indices.compute_only.is_empty() as u32;
    }

    fn choose_swap_surface_format(available: &Vec<vk::SurfaceFormatKHR>) -> vk::SurfaceFormatKHR {
        for format in available {
            if format.format == vk::Format::B8G8R8A8_SRGB
                && format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            {
                return *format;
            }
        }

        available[0]
    }

    fn choose_swap_present_mode(available: &Vec<vk::PresentModeKHR>) -> vk::PresentModeKHR {
        for mode in available {
            if *mode == vk::PresentModeKHR::MAILBOX {
                return *mode;
            }
        }

        available[0]
    }

    fn choose_swap_extent(
        width: u32,
        height: u32,
        capabilities: &vk::SurfaceCapabilitiesKHR,
    ) -> vk::Extent2D {
        if capabilities.current_extent.width != u32::MAX {
            return capabilities.current_extent;
        }

        vk::Extent2D {
            width: width.clamp(
                capabilities.min_image_extent.width,
                capabilities.max_image_extent.width,
            ),
            height: height.clamp(
                capabilities.min_image_extent.height,
                capabilities.max_image_extent.height,
            ),
        }
    }

    fn find_depth_format(context: &VulkanContext) -> vk::Format {
        let depth_format = context.find_supported_format(
            &vec![
                vk::Format::D32_SFLOAT,
                vk::Format::D32_SFLOAT_S8_UINT,
                vk::Format::D24_UNORM_S8_UINT,
            ],
            vk::ImageTiling::OPTIMAL,
            vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT,
        );

        if depth_format == vk::Format::UNDEFINED {
            panic!("failed to find suitable depth format");
        }

        depth_format
    }

    fn create_surface(
        entry: &ash::Entry,
        instance: &ash::Instance,
        window: &winit::window::Window,
    ) -> vk::SurfaceKHR {
        use winit::raw_window_handle::RawWindowHandle;

        let handle: RawWindowHandle = window.window_handle().unwrap().as_raw();

        #[cfg(target_os = "windows")]
        if let RawWindowHandle::Win32(handle) = handle {
            let surface_create_info = vk::Win32SurfaceCreateInfoKHR {
                s_type: vk::StructureType::WIN32_SURFACE_CREATE_INFO_KHR,
                p_next: ptr::null(),
                flags: vk::Win32SurfaceCreateFlagsKHR::empty(),
                hwnd: isize::from(handle.hwnd),
                hinstance: isize::from(handle.hinstance.unwrap()),
                ..Default::default()
            };

            let win32_surface_loader = khr::win32_surface::Instance::new(&entry, &instance);
            return unsafe {
                win32_surface_loader
                    .create_win32_surface(&surface_create_info, None)
                    .unwrap()
            };
        }

        panic!("expected a win32 window");
    }

    fn new(window: &winit::window::Window) -> Self {
        let entry: ash::Entry = unsafe { ash::Entry::load().unwrap() };

        #[cfg(feature = "validation")]
        if !check_validation_layer_support(&entry) {
            panic!("requested validation layers but not supported");
        }

        let instance = instance::create_instance(&entry);

        #[cfg(feature = "validation")]
        let debug_utils_loader = ext::debug_utils::Instance::new(&entry, &instance);

        #[cfg(feature = "validation")]
        let debug_messager = instance::create_debug_messenger(&debug_utils_loader);

        let surface_loader = khr::surface::Instance::new(&entry, &instance);
        let surface = Self::create_surface(&entry, &instance, window);

        let vulkan_allocator = Rc::new(RefCell::new(VulkanAllocator::new()));
        // let allocation_callbacks = vulkan_allocator.get_allocation_callbacks();

        const REQUIRED_DEVICE_EXTENSIONS: [*const c_char; 1] = [ash::khr::swapchain::NAME.as_ptr()];
        let context = VulkanContext::new(
            &entry,
            instance,
            &surface_loader,
            surface,
            &REQUIRED_DEVICE_EXTENSIONS,
            Self::score_physical_device,
            &vulkan_allocator,
        );

        let graphics_queue = Queue::new(
            &context,
            *context.queue_family_indices.graphics.iter().next().unwrap(),
            0,
        );

        let present_queue = Queue::new(
            &context,
            *context.queue_family_indices.present.iter().next().unwrap(),
            0,
        );

        let transfer_queue = if !context.queue_family_indices.transfer_only.is_empty() {
            Queue::new(
                &context,
                *context
                    .queue_family_indices
                    .transfer_only
                    .iter()
                    .next()
                    .unwrap(),
                0,
            )
        } else {
            graphics_queue
        };

        let compute_queue = if !context.queue_family_indices.compute_only.is_empty() {
            Queue::new(
                &context,
                *context
                    .queue_family_indices
                    .transfer_only
                    .iter()
                    .next()
                    .unwrap(),
                0,
            )
        } else {
            Queue::new(
                &context,
                *context.queue_family_indices.compute.iter().next().unwrap(),
                0,
            )
        };

        let swapchain_details = VulkanContext::query_swapchain_support(
            &surface_loader,
            surface,
            context.physical_device,
        );
        let swapchain_format = Self::choose_swap_surface_format(&swapchain_details.formats);
        let swapchain_present_mode: vk::PresentModeKHR =
            Self::choose_swap_present_mode(&swapchain_details.present_modes);

        let size: winit::dpi::PhysicalSize<u32> = window.inner_size();
        let swapchain_extent =
            Self::choose_swap_extent(size.width, size.height, &swapchain_details.capabilities);

        let depth_format = Self::find_depth_format(&context);

        let graphics_command_pool = CommandPool::new(
            &context,
            vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
            graphics_queue.queue,
            graphics_queue.family_index,
            &vulkan_allocator,
        );

        let initial_transfer_command_buffer = CommandBuffer::new(
            &context,
            graphics_command_pool.command_pool,
            vk::CommandBufferLevel::PRIMARY,
        );

        initial_transfer_command_buffer
            .begin(&context, vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        let (vertex_buffer, vertex_staging_buffer) = Buffer::create_vertex_buffer(
            &context,
            &VERTICES,
            initial_transfer_command_buffer.command_buffer,
            vk::SharingMode::EXCLUSIVE,
            &[],
            &vulkan_allocator,
        );

        let (index_buffer, index_staging_buffer) = Buffer::create_index_buffer(
            &context,
            &INDICES,
            initial_transfer_command_buffer.command_buffer,
            vk::SharingMode::EXCLUSIVE,
            &[],
            &vulkan_allocator,
        );

        let image = ImageReader::open("image2.png")
            .unwrap()
            .decode()
            .unwrap()
            .into_rgba8();
        let pixels: &[u8] = image.as_bytes();

        let (texture_image, texture_image_staging_buffer) = Image::create_texture_image(
            &context,
            pixels,
            image.width(),
            image.height(),
            vk::Format::R8G8B8A8_SRGB,
            vk::ImageTiling::OPTIMAL,
            initial_transfer_command_buffer.command_buffer,
            vk::SharingMode::EXCLUSIVE,
            &[],
            &vulkan_allocator,
        );
        let texture_image_view = ImageView::new(
            &context,
            texture_image.image,
            vk::Format::R8G8B8A8_SRGB,
            vk::ImageViewType::TYPE_2D,
            vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            },
            &vulkan_allocator,
        );

        initial_transfer_command_buffer.end(&context);
        graphics_command_pool.submit(
            &context,
            slice::from_ref(&initial_transfer_command_buffer.command_buffer),
            &[],
            &[],
            &[],
            vk::Fence::null(),
        );
        graphics_command_pool.wait_queue(&context);
        graphics_command_pool.free_command_buffers(
            &context,
            slice::from_ref(&initial_transfer_command_buffer.command_buffer),
        );

        vertex_staging_buffer.destroy(&context, &vulkan_allocator);
        index_staging_buffer.destroy(&context, &vulkan_allocator);
        texture_image_staging_buffer.destroy(&context, &vulkan_allocator);

        let uniform_buffer_binding: u32 = 0;
        let uniform_sampler_binding: u32 = 1;
        let descriptor_set_layout_builder = DescriptorSetLayoutBuilder::new()
            .add_uniform_buffer(uniform_buffer_binding, 1, vk::ShaderStageFlags::VERTEX)
            .add_image_sampler(uniform_sampler_binding, 1, vk::ShaderStageFlags::FRAGMENT);

        let descriptor_set_layout = DescriptorSetLayout::create_descriptor_set_layout(
            &context,
            &descriptor_set_layout_builder.bindings,
            &vulkan_allocator,
        );

        let max_sets = MAX_FRAMES_IN_FLIGHT as u32;
        let pool_sizes = descriptor_set_layout_builder.calculate_pool_sizes(max_sets);
        let descriptor_pool = DescriptorPool::create_descriptor_pool(
            &context,
            &pool_sizes,
            max_sets,
            &vulkan_allocator,
        );

        let descriptor_set_layouts: Vec<vk::DescriptorSetLayout> =
            vec![descriptor_set_layout; max_sets as usize];
        let descriptor_sets = DescriptorPool::create_descriptor_sets(
            &context,
            descriptor_pool,
            &descriptor_set_layouts,
        );

        let uniform_buffers = Self::create_uniform_buffers(&context, &vulkan_allocator);
        let sampler = Self::create_texture_sampler(&context, &vulkan_allocator);

        for i in 0..max_sets as usize {
            let buffer_info = vk::DescriptorBufferInfo {
                buffer: uniform_buffers[i].buffer,
                offset: 0,
                range: vk::WHOLE_SIZE,
            };

            let buffer_write = create_descriptor_uniform_buffer_write(
                descriptor_sets[i],
                &buffer_info,
                uniform_buffer_binding,
                0,
                1,
            );

            let image_info = vk::DescriptorImageInfo {
                sampler: sampler,
                image_view: texture_image_view.image_view,
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            };

            let image_write = create_descriptor_image_sampler_write(
                descriptor_sets[i],
                &image_info,
                uniform_sampler_binding,
                0,
                1,
            );

            let descriptor_writes: [vk::WriteDescriptorSet; 2] = [buffer_write, image_write];

            DescriptorPool::write_descriptors(&context, &descriptor_writes, &[]);
        }

        let render_pass = Self::create_render_pass(
            &context,
            swapchain_format.format,
            depth_format,
            &vulkan_allocator,
        );

        let swapchain = Swapchain::new(
            &context,
            surface,
            swapchain_format,
            swapchain_present_mode,
            swapchain_extent,
            depth_format,
            &swapchain_details,
            if graphics_queue.family_index == present_queue.family_index {
                vk::SharingMode::EXCLUSIVE
            } else {
                vk::SharingMode::CONCURRENT
            },
            graphics_queue.family_index,
            present_queue.family_index,
            &render_pass,
            &vulkan_allocator,
        );

        let vertex_shader = ShaderModule::create_shader_module(
            &context,
            &fs::read("res/shader.vert.spv").unwrap(),
            &vulkan_allocator,
        );
        let fragment_shader = ShaderModule::create_shader_module(
            &context,
            &fs::read("res/shader.frag.spv").unwrap(),
            &vulkan_allocator,
        );

        let pipeline_layout = PipelineLayout::create_pipeline_layout(
            &context,
            slice::from_ref(&descriptor_set_layout),
            &[],
            &vulkan_allocator,
        );

        let graphics_pipeline = GraphicsPipeline::new(
            &context,
            pipeline_layout,
            vertex_shader,
            fragment_shader,
            swapchain_extent,
            render_pass.render_pass,
            0,
            MyVertex::get_binding_description(),
            &MyVertex::get_attribute_description(),
        );

        let draw_command_buffers = CommandBuffer::create_command_buffers(
            &context,
            graphics_command_pool.command_pool,
            vk::CommandBufferLevel::PRIMARY,
            MAX_FRAMES_IN_FLIGHT as u32,
        );

        let transfer_command_pool = CommandPool::new(
            &context,
            vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
            transfer_queue.queue,
            transfer_queue.family_index,
            &vulkan_allocator,
        );

        let shader_storage_buffers = vec![]; //Self::create_shader_storage_buffers(&context);
        let compute_shader = ShaderModule::create_shader_module(
            &context,
            &fs::read("res/shader.comp.spv").unwrap(),
            &vulkan_allocator,
        );
        // let compute_pipeline: ComputePipeline = ComputePipeline::new();

        let compute_command_pool = CommandPool::new(
            &context,
            vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
            compute_queue.queue,
            compute_queue.family_index,
            &vulkan_allocator,
        );

        let frames = FramesInFlight::new(&context, MAX_FRAMES_IN_FLIGHT, &vulkan_allocator);

        Self {
            entry,
            vulkan_allocator,
            context,

            #[cfg(feature = "validation")]
            debug_utils_loader,

            #[cfg(feature = "validation")]
            debug_messager,

            surface_loader,
            surface,
            graphics_queue,
            present_queue,
            transfer_queue,
            compute_queue,
            swapchain,
            descriptor_set_layout,
            descriptor_pool,
            descriptor_sets,
            uniform_buffers,
            sampler,
            render_pass,
            pipeline_layout,
            graphics_pipeline,
            graphics_command_pool,
            draw_command_buffers,
            vertex_buffer,
            index_buffer,
            texture_image,
            texture_image_view,
            transfer_command_pool,
            compute_command_pool,
            shader_storage_buffers,
            frames,
            start_time: SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap(),
        }
    }

    fn create_texture_sampler(
        context: &VulkanContext,
        vulkan_allocator: &Rc<RefCell<VulkanAllocator>>,
    ) -> vk::Sampler {
        let properties: vk::PhysicalDeviceProperties = unsafe {
            context
                .instance
                .get_physical_device_properties(context.physical_device)
        };

        let create_info = vk::SamplerCreateInfo {
            s_type: vk::StructureType::SAMPLER_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::SamplerCreateFlags::empty(),
            mag_filter: vk::Filter::LINEAR,
            min_filter: vk::Filter::LINEAR,
            mipmap_mode: vk::SamplerMipmapMode::LINEAR,
            address_mode_u: vk::SamplerAddressMode::REPEAT,
            address_mode_v: vk::SamplerAddressMode::REPEAT,
            address_mode_w: vk::SamplerAddressMode::REPEAT,
            mip_lod_bias: properties.limits.max_sampler_lod_bias,
            anisotropy_enable: vk::TRUE,
            max_anisotropy: properties.limits.max_sampler_anisotropy,
            compare_enable: vk::FALSE,
            compare_op: vk::CompareOp::ALWAYS,
            min_lod: 0.0,
            max_lod: 0.0,
            border_color: vk::BorderColor::INT_OPAQUE_BLACK,
            unnormalized_coordinates: vk::FALSE,
            ..Default::default()
        };

        unsafe {
            context
                .device
                .create_sampler(
                    &create_info,
                    Some(&vulkan_allocator.borrow_mut().get_allocation_callbacks()),
                )
                .unwrap()
        }
    }

    fn recreate_swapchain(&mut self, size: winit::dpi::PhysicalSize<u32>) {
        self.context.wait_idle();

        let swapchain_details = VulkanContext::query_swapchain_support(
            &self.surface_loader,
            self.surface,
            self.context.physical_device,
        );
        let swapchain_format = Self::choose_swap_surface_format(&swapchain_details.formats);
        let swapchain_present_mode: vk::PresentModeKHR =
            Self::choose_swap_present_mode(&swapchain_details.present_modes);
        let swapchain_extent =
            Self::choose_swap_extent(size.width, size.height, &swapchain_details.capabilities);

        if swapchain_format != self.swapchain.format {
            self.render_pass
                .destroy(&self.context, &self.vulkan_allocator);
            self.render_pass = Self::create_render_pass(
                &self.context,
                swapchain_format.format,
                self.swapchain.depth_format,
                &self.vulkan_allocator,
            );

            GraphicsPipeline::destroy_pipeline(&self.context, self.graphics_pipeline.pipeline);
            self.graphics_pipeline = GraphicsPipeline::new(
                &self.context,
                self.pipeline_layout,
                self.graphics_pipeline.vertex_shader,
                self.graphics_pipeline.fragment_shader,
                swapchain_extent,
                self.render_pass.render_pass,
                0,
                MyVertex::get_binding_description(),
                &MyVertex::get_attribute_description(),
            );
        }

        self.swapchain.recreate(
            &self.context,
            self.surface,
            swapchain_format,
            swapchain_present_mode,
            swapchain_extent,
            &swapchain_details,
            &self.render_pass,
            &self.vulkan_allocator,
        );
    }

    fn update_uniform_buffers(&self) {
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap();

        let ubo = TransformationData {
            proj_view: glm::perspective(
                self.swapchain.extent.width as f32 / self.swapchain.extent.height as f32,
                glm::pi::<f32>() / 4.0,
                0.01,
                1000.0,
            ) * glm::look_at(
                &glm::vec3(0.0, 0.0, -10.5),
                &glm::vec3(0.0, 0.0, 0.0),
                &glm::vec3(0.0, 1.0, 0.0),
            ),
            model: glm::rotate(
                &glm::rotate(
                    &glm::identity::<f32, 4>(),
                    (now - self.start_time).as_millis() as f32 / 1000.0,
                    &glm::vec3(1.0, 0.0, 0.0),
                ),
                (now - self.start_time).as_millis() as f32 / 1000.0,
                &glm::vec3(0.0, 1.0, 0.0),
            ),
        };

        unsafe {
            self.uniform_buffers[self.frames.curr_frame]
                .mapped_memory
                .copy_from_nonoverlapping(
                    &ubo as *const TransformationData as *const c_void,
                    size_of::<TransformationData>(),
                );
        }
    }

    fn draw_frame(&mut self, window: &winit::window::Window, window_resized: bool) {
        let size = window.inner_size();

        if window_resized {
            self.recreate_swapchain(size);
            return;
        }

        unsafe {
            self.context
                .device
                .wait_for_fences(
                    std::slice::from_ref(&self.frames.current_frame().in_flight_fence),
                    true,
                    u64::MAX,
                )
                .unwrap();
        }

        let result = unsafe {
            self.context.swapchain_loader.acquire_next_image(
                self.swapchain.swapchain,
                u64::MAX,
                self.frames.current_frame().swap_image_available_semaphore,
                vk::Fence::null(),
            )
        };

        let image_index: u32;

        // NOTE: if suboptimal finish this frame's submission so calling waitIdle will wait for image_available_semaphore to finish and recreate swapchain.
        // TODO: it might be better to just submit an empty command buffer waiting on image_available_semaphore so I don't need to finish the frame.
        let mut acquire_suboptimal = false;

        match result {
            Ok(value) => {
                if value.1 {
                    acquire_suboptimal = true;
                }

                image_index = value.0;
            }
            Err(error) => {
                if error == vk::Result::ERROR_OUT_OF_DATE_KHR {
                    self.recreate_swapchain(size);
                    return;
                }

                panic!("failed to acquire swap chain image!");
            }
        }

        unsafe {
            self.context
                .device
                .reset_fences(std::slice::from_ref(
                    &self.frames.current_frame().in_flight_fence,
                ))
                .unwrap();
        }

        self.record_commands(image_index);
        self.update_uniform_buffers();

        let wait_stages: [vk::PipelineStageFlags; 1] =
            [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];

        self.graphics_command_pool.submit(
            &self.context,
            slice::from_ref(&self.draw_command_buffers[self.frames.curr_frame]),
            slice::from_ref(&self.frames.current_frame().swap_image_available_semaphore),
            &wait_stages,
            slice::from_ref(&self.frames.current_frame().render_finished_semaphore),
            self.frames.current_frame().in_flight_fence,
        );

        let present_info = vk::PresentInfoKHR {
            s_type: vk::StructureType::PRESENT_INFO_KHR,
            p_next: ptr::null(),
            wait_semaphore_count: 1,
            p_wait_semaphores: &self.frames.current_frame().render_finished_semaphore,
            swapchain_count: 1,
            p_swapchains: &self.swapchain.swapchain,
            p_image_indices: &image_index,
            p_results: ptr::null_mut(),
            ..Default::default()
        };

        let result = unsafe {
            self.context
                .swapchain_loader
                .queue_present(self.present_queue.queue, &present_info)
        };

        if result.is_err() && result.unwrap_err() == vk::Result::ERROR_OUT_OF_DATE_KHR
            || acquire_suboptimal
            || result.is_ok() && result.unwrap()
        {
            self.recreate_swapchain(size);
            return;
        } else if result.is_err() {
            panic!("failed to acquire swap chain image!");
        }

        self.frames.step();
    }

    fn record_commands(&self, swap_image_index: u32) {
        let cmd = CommandBuffer {
            command_buffer: self.draw_command_buffers[self.frames.curr_frame],
        };

        cmd.reset(&self.context);
        cmd.begin(&self.context, vk::CommandBufferUsageFlags::empty());

        let clear_values: [vk::ClearValue; 2] = [
            vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 1.0],
                },
            },
            vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 1.0,
                    stencil: 0,
                },
            },
        ];

        let render_area = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: self.swapchain.extent,
        };

        cmd.begin_render_pass(
            &self.context,
            self.render_pass.render_pass,
            self.swapchain.framebuffers[swap_image_index as usize],
            &render_area,
            &clear_values,
            vk::SubpassContents::INLINE,
        );

        cmd.bind_pipeline(
            &self.context,
            vk::PipelineBindPoint::GRAPHICS,
            self.graphics_pipeline.pipeline,
        );

        cmd.bind_vertex_buffers(
            &self.context,
            0,
            slice::from_ref(&self.vertex_buffer.buffer),
            slice::from_ref(&0),
        );
        cmd.bind_index_buffer(
            &self.context,
            self.index_buffer.buffer,
            0,
            vk::IndexType::UINT16,
        );

        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: self.swapchain.extent.width as f32,
            height: self.swapchain.extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };

        let scissor = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: self.swapchain.extent,
        };

        cmd.set_viewports_scissors(
            &self.context,
            slice::from_ref(&viewport),
            slice::from_ref(&scissor),
        );

        unsafe {
            self.context.device.cmd_bind_descriptor_sets(
                cmd.command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                slice::from_ref(&self.descriptor_sets[self.frames.curr_frame]),
                &[],
            );
            self.context.device.cmd_draw_indexed(
                cmd.command_buffer,
                INDICES.len() as u32,
                1,
                0,
                0,
                0,
            );
        }

        cmd.end_render_pass(&self.context);
        cmd.end(&self.context);
    }

    fn create_render_pass(
        context: &VulkanContext,
        swapchain_format: vk::Format,
        depth_image_format: vk::Format,
        vulkan_allocator: &Rc<RefCell<VulkanAllocator>>,
    ) -> RenderPass {
        let color_attachment = vk::AttachmentDescription {
            format: swapchain_format,
            flags: vk::AttachmentDescriptionFlags::empty(),
            samples: vk::SampleCountFlags::TYPE_1, // no multisampling for now
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::STORE,
            // not using stencil buffer for now
            stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
            stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
            initial_layout: vk::ImageLayout::UNDEFINED,
            final_layout: vk::ImageLayout::PRESENT_SRC_KHR,
        };

        let depth_attachment = vk::AttachmentDescription {
            format: depth_image_format,
            flags: vk::AttachmentDescriptionFlags::empty(),
            samples: vk::SampleCountFlags::TYPE_1, // no multisampling for now
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::DONT_CARE,
            // not using stencil buffer for now
            stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
            stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
            initial_layout: vk::ImageLayout::UNDEFINED,
            final_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        };

        let color_attachment_ref = vk::AttachmentReference {
            attachment: 0,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        };

        let depth_attachment_ref = vk::AttachmentReference {
            attachment: 1,
            layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        };

        let subpass: vk::SubpassDescription<'_> = vk::SubpassDescription {
            flags: vk::SubpassDescriptionFlags::empty(),
            pipeline_bind_point: vk::PipelineBindPoint::GRAPHICS,
            p_color_attachments: &color_attachment_ref,
            color_attachment_count: 1,

            // only using color attachments for now
            p_input_attachments: ptr::null(),
            p_depth_stencil_attachment: &depth_attachment_ref,
            p_resolve_attachments: ptr::null(),
            p_preserve_attachments: ptr::null(),
            input_attachment_count: 0,
            preserve_attachment_count: 0,
            ..Default::default()
        };

        let dependency = vk::SubpassDependency {
            src_subpass: vk::SUBPASS_EXTERNAL,
            dst_subpass: 0,
            src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
            dst_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
            src_access_mask: vk::AccessFlags::empty(),
            dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            dependency_flags: vk::DependencyFlags::empty(),
        };

        let attachments: [vk::AttachmentDescription; 2] = [color_attachment, depth_attachment];

        RenderPass::new(
            &context,
            &attachments,
            slice::from_ref(&subpass),
            slice::from_ref(&dependency),
            color_attachment_ref.attachment,
            depth_attachment_ref.attachment,
            vulkan_allocator,
        )
    }

    fn create_uniform_buffers(
        context: &VulkanContext,
        vulkan_allocator: &Rc<RefCell<VulkanAllocator>>,
    ) -> Vec<Buffer> {
        let buffer_size: vk::DeviceSize = size_of::<TransformationData>() as u64;
        let mut uniform_buffers: Vec<Buffer> = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);

        for i in 0..MAX_FRAMES_IN_FLIGHT {
            let (buffer_memory, buffer) = Buffer::create_buffer(
                &context,
                buffer_size,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                vk::SharingMode::EXCLUSIVE,
                &[],
                &vulkan_allocator,
            );

            uniform_buffers.push(Buffer {
                buffer,
                memory: buffer_memory,
                mapped_memory: context.map_memory(buffer_memory, 0, buffer_size),
            });
        }

        uniform_buffers
    }

    fn create_shader_storage_buffers(
        context: &VulkanContext,
        vulkan_allocator: &Rc<RefCell<VulkanAllocator>>,
    ) -> Vec<Buffer> {
        let buffer_size: vk::DeviceSize = size_of::<TransformationData>() as u64;
        let mut ssbos: Vec<Buffer> = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);

        for i in 0..MAX_FRAMES_IN_FLIGHT {
            let (buffer_memory, buffer) = Buffer::create_buffer(
                &context,
                0,
                vk::BufferUsageFlags::VERTEX_BUFFER
                    | vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_DST,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
                vk::SharingMode::EXCLUSIVE,
                &[],
                &vulkan_allocator,
            );

            ssbos.push(Buffer {
                buffer,
                memory: buffer_memory,
                mapped_memory: ptr::null_mut(),
            });
        }

        ssbos
    }

    fn generate_particles() {
        // let particles: Vec<Particle> = Vec::with_capacity(100);
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        #[cfg(feature = "validation")]
        unsafe {
            self.debug_utils_loader
                .destroy_debug_utils_messenger(self.debug_messager, None);
        }

        self.context.wait_idle();

        for i in 0..MAX_FRAMES_IN_FLIGHT as usize {
            Semaphore::destroy_semaphore(
                &self.context,
                self.frames.frames[i].render_finished_semaphore,
                &self.vulkan_allocator,
            );
            Semaphore::destroy_semaphore(
                &self.context,
                self.frames.frames[i].swap_image_available_semaphore,
                &self.vulkan_allocator,
            );
            Fence::destroy_fence(
                &self.context,
                self.frames.frames[i].in_flight_fence,
                &self.vulkan_allocator,
            );

            self.uniform_buffers[i].destroy(&self.context, &self.vulkan_allocator);
        }

        Sampler::destroy_sampler(&self.context, self.sampler, &self.vulkan_allocator);

        DescriptorPool::destroy_descriptor_pool(
            &self.context,
            self.descriptor_pool,
            &self.vulkan_allocator,
        );
        DescriptorSetLayout::destroy_descriptor_set_layout(
            &self.context,
            self.descriptor_set_layout,
            &self.vulkan_allocator,
        );

        PipelineLayout::destroy_pipeline_layout(
            &self.context,
            self.pipeline_layout,
            &self.vulkan_allocator,
        );
        self.graphics_pipeline
            .destroy(&self.context, &self.vulkan_allocator);
        self.render_pass
            .destroy(&self.context, &self.vulkan_allocator);

        self.vertex_buffer
            .destroy(&self.context, &self.vulkan_allocator);
        self.index_buffer
            .destroy(&self.context, &self.vulkan_allocator);

        self.texture_image_view
            .destroy(&self.context, &self.vulkan_allocator);
        self.texture_image
            .destroy(&self.context, &self.vulkan_allocator);

        self.graphics_command_pool
            .destroy(&self.context, &self.vulkan_allocator);
        self.transfer_command_pool
            .destroy(&self.context, &self.vulkan_allocator);
        self.compute_command_pool
            .destroy(&self.context, &self.vulkan_allocator);

        self.swapchain
            .destroy(&self.context, &self.vulkan_allocator);
        unsafe { self.surface_loader.destroy_surface(self.surface, None) }
        self.context.destroy_device();
        self.context.destroy_instance();
    }
}

#[derive(Default)]
struct App {
    window: Option<winit::window::Window>,
    renderer: Option<Renderer>,
    window_resized: bool,
}

impl winit::application::ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let window = event_loop
            .create_window(winit::window::Window::default_attributes())
            .unwrap();

        self.renderer = Some(Renderer::new(&window));
        self.window = Some(window);
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        match event {
            winit::event::WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            winit::event::WindowEvent::Resized(size) => {
                if size.width != 0 && size.height != 0 {
                    self.window_resized = true;
                }
            }
            winit::event::WindowEvent::RedrawRequested => {
                self.renderer
                    .as_mut()
                    .unwrap()
                    .draw_frame(self.window.as_ref().unwrap(), self.window_resized);

                self.window.as_ref().unwrap().request_redraw();

                if self.window_resized {
                    self.window_resized = false;
                }
            }
            _ => (),
        }
    }
}

fn main() {
    env_logger::init();

    let event_loop = winit::event_loop::EventLoop::new().unwrap();
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);

    let mut app: App = Default::default();
    event_loop.run_app(&mut app).unwrap();
}
