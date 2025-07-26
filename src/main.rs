#![allow(dead_code, unused_variables)]
use std::{fs, ptr, slice};
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
use descriptor_set_layout::DescriptorSetLayoutBuilder;
use winit::raw_window_handle::HasWindowHandle;

struct MyVertex {
    pos: glm::Vec3,
    tex_uv: glm::Vec2
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

const MAX_FRAMES_IN_FLIGHT: u32 = 3;

struct Renderer {
    entry: ash::Entry,
    context: VulkanContext,
    
    #[cfg(feature="validation")]
    debug_utils_loader: ext::debug_utils::Instance,
    
    #[cfg(feature="validation")]
    debug_messager: vk::DebugUtilsMessengerEXT,
    
    surface_loader: khr::surface::Instance,
    surface: vk::SurfaceKHR
}

impl Renderer {
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

        let size = window.inner_size();
        let (swapchain, swapchain_images, swapchain_image_format, swapchain_extent) = context.create_swapchain(size.width, size.height, surface);
        
        // let (depth_image, depth_image_memory) = context.create_image(swapchain_extent.width, swapchain_extent.height, depth_format, vk::ImageTiling::OPTIMAL, vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT, vk::MemoryPropertyFlags::DEVICE_LOCAL);
        // let depth_image_view = context.create_image_view(depth_image, depth_format, vk::ImageAspectFlags::DEPTH);

        let vertex_buffer = context.create_vertex_buffer(&VERTICES);
        let index_buffer = context.create_index_buffer(&INDICES);

        let descriptor_set_layout_builder = DescriptorSetLayoutBuilder::new()
            .add_uniform_buffer(0, 1, vk::ShaderStageFlags::VERTEX);

        let descriptor_set_layout = context.create_descriptor_set_layout(&descriptor_set_layout_builder.bindings);
        let max_sets = MAX_FRAMES_IN_FLIGHT;
        let pool_sizes = descriptor_set_layout_builder.calculate_pool_sizes(max_sets);
        
        let descriptor_pool = context.create_descriptor_pool(&pool_sizes, max_sets);

        let descriptor_set_layouts: Vec<vk::DescriptorSetLayout> = vec![descriptor_set_layout; max_sets as usize];
        let descriptor_sets = context.create_descriptor_sets(descriptor_pool, &descriptor_set_layouts);
        
        let vertex_shader = context.create_shader_module(&fs::read("res/shader.vert.spv").unwrap());
        let fragment_shader = context.create_shader_module(&fs::read("res/shader.frag.spv").unwrap());

        // let render_pass = Self::create_render_pass(&context, swapchain_image_format, depth_image_format);

        Self {
            entry,
            context,
            
            #[cfg(feature="validation")]
            debug_utils_loader,
            
            #[cfg(feature="validation")]
            debug_messager,

            surface_loader,
            surface
        }
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

    fn create_render_pass(context: &VulkanContext, swapchain_image_format: vk::Format, depth_image_format: vk::Format) -> vk::RenderPass {
        let color_attachment = vk::AttachmentDescription {
            format: swapchain_image_format,
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

        context.create_render_pass(&attachments, slice::from_ref(&subpass), slice::from_ref(&dependency))
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