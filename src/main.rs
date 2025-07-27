#![allow(dead_code, unused_variables)]
use std::{ffi::c_void, fs, ptr, slice, time::{Duration, SystemTime}};
use nalgebra_glm as glm;
use ash::{khr, vk};

#[cfg(feature="validation")]
use ash::ext;

mod instance;
#[cfg(feature="validation")]
use instance::check_validation_layer_support;

mod vulkan_context;
use vulkan_context::VulkanContext;

mod vertex;
use vertex::Vertex;

mod descriptor_set_layout;
use descriptor_set_layout::{create_descriptor_uniform_buffer_write, DescriptorSetLayoutBuilder};
use winit::raw_window_handle::HasWindowHandle;

mod command;
use command::Command;

struct MyVertex {
    pos: glm::Vec3,
    tex_uv: glm::Vec2
}

struct TransformationData {
    proj_view: glm::Mat4,
    model: glm::Mat4
}

impl Vertex for MyVertex {
    fn get_binding_description() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription {
            binding: 0,
            stride: size_of::<MyVertex>() as u32,
            input_rate: vk::VertexInputRate::VERTEX
        }
    }

    fn get_attribute_description() -> Vec<vk::VertexInputAttributeDescription> {
        let attribute_description_pos = vk::VertexInputAttributeDescription {
            location: 0,
            binding: 0,
            format: vk::Format::R32G32B32_SFLOAT,
            offset: 0
        };

        let attribute_description_color = vk::VertexInputAttributeDescription {
            location: 1,
            binding: 0,
            format: vk::Format::R32G32_SFLOAT,
            offset: 3 * size_of::<f32>() as u32
        };

        vec![attribute_description_pos, attribute_description_color]
    }
}

const VERTICES: [MyVertex; 24] = [
    // Front face
    MyVertex { pos: glm::Vec3::new(-1.0, -1.0,  1.0), tex_uv: glm::Vec2::new(0.0, 0.0) },
    MyVertex { pos: glm::Vec3::new( 1.0, -1.0,  1.0), tex_uv: glm::Vec2::new(1.0, 0.0) },
    MyVertex { pos: glm::Vec3::new( 1.0,  1.0,  1.0), tex_uv: glm::Vec2::new(1.0, 1.0) },
    MyVertex { pos: glm::Vec3::new(-1.0,  1.0,  1.0), tex_uv: glm::Vec2::new(0.0, 1.0) },

    // Back face
    MyVertex { pos: glm::Vec3::new(-1.0, -1.0, -1.0), tex_uv: glm::Vec2::new(0.0, 0.0) },
    MyVertex { pos: glm::Vec3::new( 1.0, -1.0, -1.0), tex_uv: glm::Vec2::new(1.0, 0.0) },
    MyVertex { pos: glm::Vec3::new( 1.0,  1.0, -1.0), tex_uv: glm::Vec2::new(1.0, 1.0) },
    MyVertex { pos: glm::Vec3::new(-1.0,  1.0, -1.0), tex_uv: glm::Vec2::new(0.0, 1.0) },

    // Left face
    MyVertex { pos: glm::Vec3::new(-1.0, -1.0, -1.0), tex_uv: glm::Vec2::new(0.0, 0.0) },
    MyVertex { pos: glm::Vec3::new(-1.0, -1.0,  1.0), tex_uv: glm::Vec2::new(1.0, 0.0) },
    MyVertex { pos: glm::Vec3::new(-1.0,  1.0,  1.0), tex_uv: glm::Vec2::new(1.0, 1.0) },
    MyVertex { pos: glm::Vec3::new(-1.0,  1.0, -1.0), tex_uv: glm::Vec2::new(0.0, 1.0) },

    // Right face
    MyVertex { pos: glm::Vec3::new( 1.0, -1.0, -1.0), tex_uv: glm::Vec2::new(0.0, 0.0) },
    MyVertex { pos: glm::Vec3::new( 1.0, -1.0,  1.0), tex_uv: glm::Vec2::new(1.0, 0.0) },
    MyVertex { pos: glm::Vec3::new( 1.0,  1.0,  1.0), tex_uv: glm::Vec2::new(1.0, 1.0) },
    MyVertex { pos: glm::Vec3::new( 1.0,  1.0, -1.0), tex_uv: glm::Vec2::new(0.0, 1.0) },

    // Top face
    MyVertex { pos: glm::Vec3::new(-1.0,  1.0, -1.0), tex_uv: glm::Vec2::new(0.0, 0.0) },
    MyVertex { pos: glm::Vec3::new(-1.0,  1.0,  1.0), tex_uv: glm::Vec2::new(1.0, 0.0) },
    MyVertex { pos: glm::Vec3::new( 1.0,  1.0,  1.0), tex_uv: glm::Vec2::new(1.0, 1.0) },
    MyVertex { pos: glm::Vec3::new( 1.0,  1.0, -1.0), tex_uv: glm::Vec2::new(0.0, 1.0) },

    // Bottom face
    MyVertex { pos: glm::Vec3::new(-1.0, -1.0, -1.0), tex_uv: glm::Vec2::new(0.0, 0.0) },
    MyVertex { pos: glm::Vec3::new(-1.0, -1.0,  1.0), tex_uv: glm::Vec2::new(1.0, 0.0) },
    MyVertex { pos: glm::Vec3::new( 1.0, -1.0,  1.0), tex_uv: glm::Vec2::new(1.0, 1.0) },
    MyVertex { pos: glm::Vec3::new( 1.0, -1.0, -1.0), tex_uv: glm::Vec2::new(0.0, 1.0) },
];

const INDICES: [u16; 36] = [
    // Front face
    0, 1, 2,
    2, 3, 0,

    // Back face
    4, 5, 6,
    6, 7, 4,

    // Left face
    8, 9, 10,
    10, 11, 8,

    // Right face
    12, 13, 14,
    14, 15, 12,

    // Top face
    16, 17, 18,
    18, 19, 16,

    // Bottom face
    20, 21, 22,
    22, 23, 20,
];

const MAX_FRAMES_IN_FLIGHT: usize = 3;

struct Renderer {
    entry: ash::Entry,
    context: VulkanContext,
    
    #[cfg(feature="validation")]
    debug_utils_loader: ext::debug_utils::Instance,
    
    #[cfg(feature="validation")]
    debug_messager: vk::DebugUtilsMessengerEXT,
    
    surface_loader: khr::surface::Instance,
    surface: vk::SurfaceKHR,

    swapchain: vk::SwapchainKHR,
    swapchain_images: Vec<vk::Image>
    swapchain_image_views: Vec<vk::ImageView>,
    swapchain_extent: vk::Extent2D,
    swapchain_format: vk::SurfaceFormatKHR,
    swapchain_present_mode: vk::PresentModeKHR,

    depth_image: vk::Image,
    depth_image_memory: vk::DeviceMemory,
    depth_image_view: vk::ImageView,

    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,

    uniform_buffers: Vec<vk::Buffer>,
    uniform_buffers_memory: Vec<vk::DeviceMemory>,
    uniform_buffers_mapped: Vec<*mut c_void>,

    render_pass: vk::RenderPass,
    swapchain_framebuffers: Vec<vk::Framebuffer>,
    
    pipeline_layout: vk::PipelineLayout,
    graphics_pipeline: vk::Pipeline,
    
    vertex_buffer: vk::Buffer,
    vertex_buffer_memory: vk::DeviceMemory,
    index_buffer: vk::Buffer,
    index_buffer_memory: vk::DeviceMemory,
    
    draw_command_pool: vk::CommandPool,
    draw_command_buffers: Vec<vk::CommandBuffer>,

    swap_image_available_semaphores: Vec<vk::Semaphore>,
    render_finished_semaphores: Vec<vk::Semaphore>,
    in_flight_fences: Vec<vk::Fence>,

    current_frame: usize,
    start_time: Duration
}

impl Renderer {
    fn choose_swap_surface_format(available: &Vec<vk::SurfaceFormatKHR>) -> vk::SurfaceFormatKHR {
        for format in available {
            if format.format == vk::Format::B8G8R8A8_SRGB && format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR {
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

    fn choose_swap_extent(width: u32, height: u32, capabilities: &vk::SurfaceCapabilitiesKHR) -> vk::Extent2D {
        if capabilities.current_extent.width != u32::MAX {
            return capabilities.current_extent;
        }

        vk::Extent2D {
            width: width.clamp(capabilities.min_image_extent.width, capabilities.max_image_extent.width),
            height: height.clamp(capabilities.min_image_extent.height, capabilities.max_image_extent.height)
        }
    }

    fn find_depth_format(context: &VulkanContext) -> vk::Format {
        let depth_format = context.find_supported_format(
            &vec![vk::Format::D32_SFLOAT, vk::Format::D32_SFLOAT_S8_UINT, vk::Format::D24_UNORM_S8_UINT],
            vk::ImageTiling::OPTIMAL,
            vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT
        );

        if depth_format == vk::Format::UNDEFINED {
            panic!("failed to find suitable depth format");
        }

        depth_format
    }

    fn new(window: &winit::window::Window) -> Self {
        let entry: ash::Entry = unsafe {
            ash::Entry::load().unwrap()
        };

        #[cfg(feature="validation")]
        if !check_validation_layer_support(&entry) {
            panic!("requested validation layers but not supported");
        }

        let instance = instance::create_instance(&entry);
        
        #[cfg(feature="validation")]
        let debug_utils_loader = ext::debug_utils::Instance::new(&entry, &instance);

        #[cfg(feature="validation")]
        let debug_messager = instance::create_debug_messenger(&debug_utils_loader);

        let surface_loader = khr::surface::Instance::new(&entry, &instance);
        let surface = Self::create_surface(&entry, &instance, window);

        let mut context = VulkanContext::new(&entry, instance, &surface_loader, surface);
        context.init();

        let size: winit::dpi::PhysicalSize<u32> = window.inner_size();
        let swapchain_format = Self::choose_swap_surface_format(&context.swapchain_details.formats);
        let swapchain_present_mode = Self::choose_swap_present_mode(&context.swapchain_details.present_modes);
        let swapchain_extent = Self::choose_swap_extent(size.width, size.height, &context.swapchain_details.capabilities);

        let (swapchain, swapchain_images) = context.create_swapchain(surface, swapchain_format, swapchain_present_mode, swapchain_extent);
        let mut swapchain_image_views: Vec<vk::ImageView> = Vec::with_capacity(swapchain_images.len());

        for image in swapchain_images.iter() {
            swapchain_image_views.push(context.create_image_view(*image, swapchain_format.format, vk::ImageAspectFlags::COLOR));
        }

        let depth_format = Self::find_depth_format(&context);
        let (depth_image, depth_image_memory) = context.create_image(swapchain_extent.width, swapchain_extent.height, depth_format, vk::ImageTiling::OPTIMAL, vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT, vk::MemoryPropertyFlags::DEVICE_LOCAL);
        let depth_image_view = context.create_image_view(depth_image, depth_format, vk::ImageAspectFlags::DEPTH);

        let uniform_buffer_binding: u32 = 0;
        let descriptor_set_layout_builder = DescriptorSetLayoutBuilder::new()
            .add_uniform_buffer(uniform_buffer_binding, 1, vk::ShaderStageFlags::VERTEX);

        let descriptor_set_layout = context.create_descriptor_set_layout(&descriptor_set_layout_builder.bindings);
        
        let max_sets = MAX_FRAMES_IN_FLIGHT as u32;
        let pool_sizes = descriptor_set_layout_builder.calculate_pool_sizes(max_sets); 
        let descriptor_pool = context.create_descriptor_pool(&pool_sizes, max_sets);

        let descriptor_set_layouts: Vec<vk::DescriptorSetLayout> = vec![descriptor_set_layout; max_sets as usize];
        let descriptor_sets = context.create_descriptor_sets(descriptor_pool, &descriptor_set_layouts);
        
        let (uniform_buffers, uniform_buffers_memory, uniform_buffers_mapped) = Self::create_uniform_buffers(&context);
        
        for i in 0..max_sets as usize {
            let buffer_info = vk::DescriptorBufferInfo {
                buffer: uniform_buffers[i],
                offset: 0,
                range: vk::WHOLE_SIZE
            };
        
            let buffer_write = create_descriptor_uniform_buffer_write(descriptor_sets[i], &buffer_info, uniform_buffer_binding, 0, 1);

            let descriptor_writes = [buffer_write];

            context.write_descriptors(&descriptor_writes, &[]);
        }

        let (render_pass, color_attachment_index, depth_attachment_index) = Self::create_render_pass(&context, swapchain_format.format, depth_format);
        let swapchain_framebuffers = Self::create_swapchain_framebuffers(&context, &swapchain_image_views, depth_image_view, render_pass, color_attachment_index, depth_attachment_index, swapchain_extent);
        
        let vertex_shader = context.create_shader_module(&fs::read("res/shader.vert.spv").unwrap());
        let fragment_shader = context.create_shader_module(&fs::read("res/shader.frag.spv").unwrap());

        let pipeline_layout = context.create_pipeline_layout(slice::from_ref(&descriptor_set_layout), &[]);
        let graphics_pipeline = Self::create_graphics_pipeline(&context, pipeline_layout, vertex_shader, fragment_shader, swapchain_extent, render_pass);
        
        context.destroy_shader_module(vertex_shader);
        context.destroy_shader_module(fragment_shader);

        let (vertex_buffer_memory, vertex_buffer) = context.create_vertex_buffer(&VERTICES);
        let (index_buffer_memory, index_buffer) = context.create_index_buffer(&INDICES);

        let draw_command_pool = context.create_command_pool(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let draw_command_buffers = context.create_command_buffers(draw_command_pool, vk::CommandBufferLevel::PRIMARY, MAX_FRAMES_IN_FLIGHT as u32);

        let swap_image_available_semaphores = context.create_semaphores(MAX_FRAMES_IN_FLIGHT as u32);
        let render_finished_semaphores = context.create_semaphores(MAX_FRAMES_IN_FLIGHT as u32);
        let in_flight_fences = context.create_fences(MAX_FRAMES_IN_FLIGHT as u32);

        Self {
            entry,
            context,
            
            #[cfg(feature="validation")]
            debug_utils_loader,
            
            #[cfg(feature="validation")]
            debug_messager,

            surface_loader,
            surface,
            swapchain,
            swapchain_image_views,
            swapchain_images,
            swapchain_extent,
            swapchain_format,
            swapchain_present_mode,
            depth_image,
            depth_image_memory,
            depth_image_view,
            descriptor_set_layout,
            descriptor_pool,
            descriptor_sets,
            uniform_buffers,
            uniform_buffers_memory,
            uniform_buffers_mapped,
            render_pass,
            swapchain_framebuffers,
            pipeline_layout,
            graphics_pipeline,
            draw_command_pool,
            draw_command_buffers,
            vertex_buffer,
            vertex_buffer_memory,
            index_buffer,
            index_buffer_memory,

            swap_image_available_semaphores,
            render_finished_semaphores,
            in_flight_fences,

            current_frame: 0,
            start_time: SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap()
        }
    }

    fn recreate_swapchain(&mut self, size: winit::dpi::PhysicalSize<u32>) {
        unsafe {
            self.device.device_wait_idle().unwrap();
        }
        
        self.cleanup_swapchain();
        (self.swapchain, self.swapchain_images) = self.context.create_swapchain(self.surface, self.swapchain_format, self.swapchain_present_mode, self.swapchain_extent);

        self.swapchain_image_views.clear();

        for image in self.swapchain_images.iter() {
            self.swapchain_image_views.push(self.context.create_image_view(*image, self.swapchain_format.format, vk::ImageAspectFlags::COLOR));
        }

        (self.depth_image, self.depth_image_memory) = self.context.create_image(self.swapchain_extent.width, self.swapchain_extent.height, self.depth_format, vk::ImageTiling::OPTIMAL, vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT, vk::MemoryPropertyFlags::DEVICE_LOCAL);
        self.depth_image_view = self.context.create_image_view(self.depth_image, self.depth_format, vk::ImageAspectFlags::DEPTH);

        self.swapchain_framebuffers = Self::create_swapchain_framebuffers(self.context, &self.swapchain_image_views, self.depth_image_view, self.render_pass, self.swapchain_extent);
    }

    fn update_uniform_buffers(&self) {
        let now = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap();

        let ubo = TransformationData {
            proj_view: glm::perspective(self.swapchain_extent.width as f32 / self.swapchain_extent.height as f32, glm::pi::<f32>() / 4.0, 0.01, 1000.0) * glm::look_at(&glm::vec3(0.0, 0.0, -10.5), &glm::vec3(0.0, 0.0, 0.0), &glm::vec3(0.0, 1.0, 0.0)),
            model: glm::rotate(&glm::rotate(&glm::identity::<f32, 4>(), (now - self.start_time).as_millis() as f32 / 1000.0, &glm::vec3(1.0, 0.0, 0.0)), (now - self.start_time).as_millis() as f32 / 1000.0, &glm::vec3(0.0, 1.0, 0.0)),
        };
        
        unsafe {
            self.uniform_buffers_mapped[self.current_frame].copy_from_nonoverlapping(&ubo as *const TransformationData as *const c_void, size_of::<TransformationData>());
        }
    }

    fn draw_frame(&mut self, window: &winit::window::Window, window_resized: bool) {
        let mut image_index: u32 = 0;

        unsafe {
            self.device.wait_for_fences(std::slice::from_ref(&self.in_flight_fences[self.current_frame]), true, u64::MAX).unwrap();
            self.device.reset_fences(std::slice::from_ref(&self.in_flight_fences[self.current_frame])).unwrap();

            let result = self.swapchain_loader.acquire_next_image(self.swapchain, u64::MAX, self.swap_image_available_semaphores[self.current_frame], vk::Fence::null());

            match result {
                Ok(value) => {
                    image_index = value.0;
                }
                Err(error) => {
                    if error == vk::Result::ERROR_OUT_OF_DATE_KHR || window_resized {
                        let size = window.inner_size();
                        self.recreate_swapchain(size);
                        return;
                    }
                    else if error != vk::Result::SUCCESS && error != vk::Result::SUBOPTIMAL_KHR {
                        panic!("failed to acquire swap chain image!");
                    }
                }
            }
        }
        
        self.record(image_index);
        self.update_uniform_buffers();

        let wait_stages: [vk::PipelineStageFlags; 1] = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];

        let submit_info = vk::SubmitInfo {
            s_type: vk::StructureType::SUBMIT_INFO,
            p_next: ptr::null(),
            wait_semaphore_count: 1,
            p_wait_semaphores: &self.swap_image_available_semaphores[self.current_frame],
            p_wait_dst_stage_mask: wait_stages.as_ptr(),
            command_buffer_count: 1,
            p_command_buffers: &self.command_buffers[self.current_frame],
            signal_semaphore_count: 1,
            p_signal_semaphores: &self.render_finished_semaphores[self.current_frame],
            ..Default::default()
        };

        unsafe {
            self.device.queue_submit(self.graphics_queue, std::slice::from_ref(&submit_info), self.in_flight_fences[self.current_frame]).unwrap();
        };

        let present_info = vk::PresentInfoKHR {
            s_type: vk::StructureType::PRESENT_INFO_KHR,
            p_next: ptr::null(),
            wait_semaphore_count: 1,
            p_wait_semaphores: &self.render_finished_semaphores[self.current_frame],
            swapchain_count: 1,
            p_swapchains: &self.swapchain,
            p_image_indices: &image_index,
            p_results: ptr::null_mut(),
            ..Default::default()
        };

        unsafe {
            let result = self.swapchain_loader.queue_present(self.present_queue, &present_info);
            
            if result.is_err() {
                let error: vk::Result = result.unwrap_err();

                if error == vk::Result::ERROR_OUT_OF_DATE_KHR || window_resized {
                    let size = window.inner_size();
                    self.recreate_swapchain(size);
                }
                else if error != vk::Result::SUCCESS && error != vk::Result::SUBOPTIMAL_KHR {
                    panic!("failed to acquire swap chain image!");
                }
            }
        };

        self.current_frame = (self.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    fn record(&self, swap_image_index: u32) {
        let cmd = Command {
            context: &self.context,
            command_buffer: self.draw_command_buffers[self.current_frame]
        };

        cmd.reset();
        cmd.begin();

        let clear_values: [vk::ClearValue; 2] = [
            vk::ClearValue {color: vk::ClearColorValue {float32: [0.0, 0.0, 0.0, 1.0]}},
            vk::ClearValue {depth_stencil: vk::ClearDepthStencilValue {depth: 1.0, stencil: 0}}
        ];

        let render_area = vk::Rect2D {
            offset: vk::Offset2D {x: 0, y: 0},
            extent: self.swapchain_extent
        };

        cmd.begin_render_pass(
            self.render_pass,
            self.swapchain_framebuffers[swap_image_index as usize],
            &render_area,
            &clear_values,
            vk::SubpassContents::INLINE
        );

        cmd.bind_pipeline(vk::PipelineBindPoint::GRAPHICS, self.graphics_pipeline);
        
        cmd.bind_vertex_buffers(0, slice::from_ref(&self.vertex_buffer), slice::from_ref(&0));
        cmd.bind_index_buffer(self.index_buffer, 0, vk::IndexType::UINT16);

        let viewport = vk::Viewport{
            x: 0.0,
            y: 0.0,
            width: self.swapchain_extent.width as f32,
            height: self.swapchain_extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0
        };

        let scissor = vk::Rect2D {
            offset: vk::Offset2D {x: 0, y: 0},
            extent: self.swapchain_extent,
        };

        cmd.set_viewports_scissors(slice::from_ref(&viewport), slice::from_ref(&scissor));
        
        unsafe {
            self.context.device.cmd_bind_descriptor_sets(cmd.command_buffer, vk::PipelineBindPoint::GRAPHICS, self.pipeline_layout, 0, slice::from_ref(&self.descriptor_sets[self.current_frame]), &[]);
            self.context.device.cmd_draw_indexed(cmd.command_buffer, INDICES.len() as u32, 1, 0, 0, 0);
        }

        cmd.end_render_pass();
        cmd.end();
    }

    fn create_surface(entry: &ash::Entry, instance: &ash::Instance, window: &winit::window::Window) -> vk::SurfaceKHR {
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
                win32_surface_loader.create_win32_surface(&surface_create_info, None).unwrap()
            }
        }

        panic!("expected a win32 window");
    }

    fn create_render_pass(context: &VulkanContext, swapchain_format: vk::Format, depth_image_format: vk::Format) -> (vk::RenderPass, u32, u32) {
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
            final_layout: vk::ImageLayout::PRESENT_SRC_KHR
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
            final_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
        };

        let color_attachment_ref = vk::AttachmentReference {
            attachment: 0,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
        };

        let depth_attachment_ref = vk::AttachmentReference {
            attachment: 1,
            layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
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
            src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
            dst_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
            src_access_mask: vk::AccessFlags::empty(),
            dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            dependency_flags: vk::DependencyFlags::empty()
        };

        let attachments: [vk::AttachmentDescription; 2] = [color_attachment, depth_attachment];

        (context.create_render_pass(&attachments, slice::from_ref(&subpass), slice::from_ref(&dependency)), color_attachment_ref.attachment, depth_attachment_ref.attachment)
    }

     fn create_graphics_pipeline(context: &VulkanContext, pipeline_layout: vk::PipelineLayout, vertex_shader: vk::ShaderModule, fragment_shader: vk::ShaderModule, swapchain_extent: vk::Extent2D, render_pass: vk::RenderPass) -> vk::Pipeline {
        let vert_shader_stage_create_info = vk::PipelineShaderStageCreateInfo {
            s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
            p_name: c"main".as_ptr(),
            p_next: ptr::null(),
            flags: vk::PipelineShaderStageCreateFlags::empty(),
            stage: vk::ShaderStageFlags::VERTEX,
            module: vertex_shader,
            p_specialization_info: ptr::null(), // for constexpr's in shader
            ..Default::default()
        };

        let frag_shader_stage_create_info = vk::PipelineShaderStageCreateInfo {
            s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
            p_name: c"main".as_ptr(),
            p_next: ptr::null(),
            flags: vk::PipelineShaderStageCreateFlags::empty(),
            stage: vk::ShaderStageFlags::FRAGMENT,
            module: fragment_shader,
            p_specialization_info: ptr::null(), // for constexpr's in shader
            ..Default::default()
        };

        let shader_stage_create_infos: [vk::PipelineShaderStageCreateInfo; 2] = [vert_shader_stage_create_info, frag_shader_stage_create_info];

        
        let vertex_binding_description: vk::VertexInputBindingDescription = MyVertex::get_binding_description();
        let vertex_attribute_description: Vec<vk::VertexInputAttributeDescription> = MyVertex::get_attribute_description();

        let vertex_input_state_create_info = vk::PipelineVertexInputStateCreateInfo {
            s_type: vk::StructureType::PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::PipelineVertexInputStateCreateFlags::empty(),
            vertex_attribute_description_count: vertex_attribute_description.len() as u32,
            p_vertex_attribute_descriptions: vertex_attribute_description.as_ptr(),
            vertex_binding_description_count: 1,
            p_vertex_binding_descriptions: &vertex_binding_description,
            ..Default::default()
        };

        let input_assembly_state_create_info = vk::PipelineInputAssemblyStateCreateInfo {
            s_type: vk::StructureType::PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            flags: vk::PipelineInputAssemblyStateCreateFlags::empty(),
            p_next: ptr::null(),
            primitive_restart_enable: vk::FALSE,
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            ..Default::default()
        };

        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: swapchain_extent.width as f32,
            height: swapchain_extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };

        let scissor = vk::Rect2D {
            offset: vk::Offset2D {x: 0, y: 0},
            extent: swapchain_extent,
        };
        
        let viewport_state_create_info = vk::PipelineViewportStateCreateInfo {
            s_type: vk::StructureType::PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::PipelineViewportStateCreateFlags::empty(),
            scissor_count: 1,
            p_scissors: &scissor,
            viewport_count: 1,
            p_viewports: &viewport,
            ..Default::default()
        };

        let rasterization_state_create_info = vk::PipelineRasterizationStateCreateInfo {
            s_type: vk::StructureType::PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::PipelineRasterizationStateCreateFlags::empty(),
            depth_clamp_enable: vk::FALSE,
            cull_mode: vk::CullModeFlags::NONE,
            front_face: vk::FrontFace::COUNTER_CLOCKWISE,
            line_width: 1.0,
            polygon_mode: vk::PolygonMode::FILL,
            rasterizer_discard_enable: vk::FALSE,
            depth_bias_clamp: 0.0,
            depth_bias_constant_factor: 0.0,
            depth_bias_enable: vk::FALSE,
            depth_bias_slope_factor: 0.0,
            ..Default::default()
        };

        // no multisampling for now
        let multisample_state_create_info = vk::PipelineMultisampleStateCreateInfo {
            s_type: vk::StructureType::PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            flags: vk::PipelineMultisampleStateCreateFlags::empty(),
            p_next: ptr::null(),
            rasterization_samples: vk::SampleCountFlags::TYPE_1,
            sample_shading_enable: vk::FALSE,
            min_sample_shading: 0.0,
            p_sample_mask: ptr::null(),
            alpha_to_one_enable: vk::FALSE,
            alpha_to_coverage_enable: vk::FALSE,
            ..Default::default()
        };

        let stencil_state = vk::StencilOpState {
            fail_op: vk::StencilOp::KEEP,
            pass_op: vk::StencilOp::KEEP,
            depth_fail_op: vk::StencilOp::KEEP,
            compare_op: vk::CompareOp::ALWAYS,
            compare_mask: 0,
            write_mask: 0,
            reference: 0,
        };

        let depth_stencil_state_create_info = vk::PipelineDepthStencilStateCreateInfo {
            s_type: vk::StructureType::PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::PipelineDepthStencilStateCreateFlags::empty(),
            depth_test_enable: vk::TRUE,
            depth_write_enable: vk::TRUE,
            depth_compare_op: vk::CompareOp::LESS,
            depth_bounds_test_enable: vk::FALSE,
            stencil_test_enable: vk::FALSE,
            front: stencil_state,
            back: stencil_state,
            max_depth_bounds: 1.0,
            min_depth_bounds: 0.0,
            ..Default::default()
        };

        let color_blend_attachment_states = [vk::PipelineColorBlendAttachmentState {
            blend_enable: vk::FALSE,
            color_write_mask: vk::ColorComponentFlags::R | vk::ColorComponentFlags::G | vk::ColorComponentFlags::B | vk::ColorComponentFlags::A,
            src_color_blend_factor: vk::BlendFactor::ONE,
            dst_color_blend_factor: vk::BlendFactor::ZERO,
            color_blend_op: vk::BlendOp::ADD,
            src_alpha_blend_factor: vk::BlendFactor::ONE,
            dst_alpha_blend_factor: vk::BlendFactor::ZERO,
            alpha_blend_op: vk::BlendOp::ADD,
        }];

        let color_blend_state_create_info = vk::PipelineColorBlendStateCreateInfo {
            s_type: vk::StructureType::PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::PipelineColorBlendStateCreateFlags::empty(),
            logic_op_enable: vk::FALSE,
            logic_op: vk::LogicOp::COPY,
            attachment_count: color_blend_attachment_states.len() as u32,
            p_attachments: color_blend_attachment_states.as_ptr(),
            blend_constants: [0.0, 0.0, 0.0, 0.0],
            ..Default::default()
        };

        let dynamic_states: [vk::DynamicState; 2] = [
            vk::DynamicState::VIEWPORT,
            vk::DynamicState::SCISSOR
        ];

        let dynamic_state_create_info = vk::PipelineDynamicStateCreateInfo {
            s_type: vk::StructureType::PIPELINE_DYNAMIC_STATE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::PipelineDynamicStateCreateFlags::empty(),
            dynamic_state_count: dynamic_states.len() as u32,
            p_dynamic_states: dynamic_states.as_ptr(),
            ..Default::default()
        };

        let graphics_pipeline_create_info = vk::GraphicsPipelineCreateInfo {
            s_type: vk::StructureType::GRAPHICS_PIPELINE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::PipelineCreateFlags::empty(),
            stage_count: shader_stage_create_infos.len() as u32,
            p_stages: shader_stage_create_infos.as_ptr(),
            p_vertex_input_state: &vertex_input_state_create_info as *const vk::PipelineVertexInputStateCreateInfo,
            p_input_assembly_state: &input_assembly_state_create_info as *const vk::PipelineInputAssemblyStateCreateInfo,
            p_tessellation_state: ptr::null(),
            p_viewport_state: &viewport_state_create_info as *const vk::PipelineViewportStateCreateInfo,
            p_rasterization_state: &rasterization_state_create_info as *const vk::PipelineRasterizationStateCreateInfo,
            p_multisample_state: &multisample_state_create_info as *const vk::PipelineMultisampleStateCreateInfo,
            p_depth_stencil_state: &depth_stencil_state_create_info as *const vk::PipelineDepthStencilStateCreateInfo,
            p_color_blend_state: &color_blend_state_create_info as *const vk::PipelineColorBlendStateCreateInfo,
            p_dynamic_state: &dynamic_state_create_info as *const vk::PipelineDynamicStateCreateInfo,
            layout: pipeline_layout,
            render_pass: render_pass,
            subpass: 0, // index to graphics subpass
            base_pipeline_handle: vk::Pipeline::null(),
            base_pipeline_index: -1,
            ..Default::default()
        };

        let graphics_pipeline_create_infos: [vk::GraphicsPipelineCreateInfo; 1] = [graphics_pipeline_create_info];
        let graphics_pipelines: Vec<vk::Pipeline> = unsafe {
            context.device.create_graphics_pipelines(vk::PipelineCache::null(), 
            graphics_pipeline_create_infos.as_slice(),
            None).unwrap()
        };

        graphics_pipelines[0]
    }

    fn create_uniform_buffers(context: &VulkanContext) -> (Vec<vk::Buffer>, Vec<vk::DeviceMemory>, Vec<*mut c_void>) {
        let buffer_size: vk::DeviceSize = size_of::<TransformationData>() as u64;
        
        let mut uniform_buffers: Vec<vk::Buffer> = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
        let mut uniform_buffers_memory: Vec<vk::DeviceMemory> = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
        let mut uniform_buffers_mapped: Vec<*mut c_void> = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);

        for i in  0..MAX_FRAMES_IN_FLIGHT {
            let (buffer_memory, buffer) = context.create_buffer(buffer_size, vk::BufferUsageFlags::UNIFORM_BUFFER, vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT);
            uniform_buffers.push(buffer);
            uniform_buffers_memory.push(buffer_memory);
            uniform_buffers_mapped.push(context.map_memory(uniform_buffers_memory[i], 0, buffer_size));
        }

        (uniform_buffers, uniform_buffers_memory, uniform_buffers_mapped)
    }

    fn create_swapchain_framebuffers(context: &VulkanContext, swapchain_image_views: &Vec<vk::ImageView>, depth_image_view: vk::ImageView, render_pass: vk::RenderPass, color_attachment_index: u32, depth_attachment_index: u32, swapchain_extent: vk::Extent2D) -> Vec<vk::Framebuffer> {
        let mut swapchain_framebuffers: Vec<vk::Framebuffer> = Vec::with_capacity(swapchain_image_views.len());

        for image_view in swapchain_image_views {
            let mut attachments: [vk::ImageView; 2] = [vk::ImageView::null(); 2];
            
            attachments[color_attachment_index as usize] = *image_view;
            attachments[depth_attachment_index as usize] = depth_image_view;

            swapchain_framebuffers.push(context.create_framebuffer(render_pass, &attachments, swapchain_extent, 1));
        }

        swapchain_framebuffers
    }

}

impl Drop for Renderer {
    fn drop(&mut self) {
        #[cfg(feature="validation")]
        unsafe {
            self.debug_utils_loader.destroy_debug_utils_messenger(self.debug_messager, None);
        }
    }
}

#[derive(Default)]
struct App {
    window: Option<winit::window::Window>,
    renderer: Option<Renderer>,
    window_resized: bool
}

impl winit::application::ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let window = event_loop.create_window(winit::window::Window::default_attributes()).unwrap();
        
        self.renderer = Some(Renderer::new(&window));
        self.window = Some(window);
    }

    fn window_event(&mut self, event_loop: &winit::event_loop::ActiveEventLoop, _id: winit::window::WindowId, event: winit::event::WindowEvent) {
        match event {
            winit::event::WindowEvent::CloseRequested => {
                event_loop.exit();
            },
            winit::event::WindowEvent::Resized(_size) => {
                self.window_resized = true;
            }
            winit::event::WindowEvent::RedrawRequested => {
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
    let event_loop = winit::event_loop::EventLoop::new().unwrap();
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);
    
    let mut app: App = Default::default();
    event_loop.run_app(&mut app).unwrap();
}