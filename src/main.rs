use std::collections::HashSet;
use std::ffi::{c_char, c_void, CStr, CString};
use std::fs::{self};
use std::ptr::{self};
use std::u64;
use std::time::{Duration, SystemTime};
use image::EncodableLayout;
use winit::raw_window_handle::HasWindowHandle;
use winit::{self};
use ash::{ext, khr, vk};
use log::{warn};
use nalgebra_glm as glm;

const REQUIRED_VALIDATION_LAYER_NAMES: [&'static str; 1] = ["VK_LAYER_KHRONOS_validation"];
const USE_VALIDATION_LAYERS: bool = true;

const REQUIRED_DEVICE_EXTENSIONS: [*const c_char; 1] = [
    ash::khr::swapchain::NAME.as_ptr()
];

const MAX_FRAMES_IN_FLIGHT: usize = 2;

unsafe extern "system" fn vulkan_debug_utils_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _p_user_data: *mut c_void,
) -> vk::Bool32 {
    let severity = match message_severity {
        vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => "[Verbose]",
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => "[Warning]",
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => "[Error]",
        vk::DebugUtilsMessageSeverityFlagsEXT::INFO => "[Info]",
        _ => "[Unknown]",
    };

    let types: &'static str = match message_type {
        vk::DebugUtilsMessageTypeFlagsEXT::GENERAL => "[General]",
        vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE => "[Performance]",
        vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION => "[Validation]",
        _ => "[Unknown]",
    };

    let message: &CStr = unsafe {
        CStr::from_ptr((*p_callback_data).p_message)
    };

    println!("[Debug]{}{}{:?}", severity, types, message);

    vk::FALSE
}

#[derive(Default)]
struct QueueFamilyIndices {
    graphics: Option<u32>,
    present: Option<u32>
}

#[derive(Default)]
struct SwapChainSupportDetails {
    capabilities: vk::SurfaceCapabilitiesKHR,
    formats: Vec<vk::SurfaceFormatKHR>,
    present_modes: Vec<vk::PresentModeKHR>
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct Vertex {
    pos: glm::Vec3,
    tex_uv: glm::Vec2
}

impl Vertex {
    fn get_binding_description() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription {
            binding: 0,
            stride: size_of::<Vertex>() as u32,
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


const VERTICES: [Vertex; 24] = [
    // Front face
    Vertex { pos: glm::Vec3::new(-1.0, -1.0,  1.0), tex_uv: glm::Vec2::new(0.0, 0.0) },
    Vertex { pos: glm::Vec3::new( 1.0, -1.0,  1.0), tex_uv: glm::Vec2::new(1.0, 0.0) },
    Vertex { pos: glm::Vec3::new( 1.0,  1.0,  1.0), tex_uv: glm::Vec2::new(1.0, 1.0) },
    Vertex { pos: glm::Vec3::new(-1.0,  1.0,  1.0), tex_uv: glm::Vec2::new(0.0, 1.0) },

    // Back face
    Vertex { pos: glm::Vec3::new(-1.0, -1.0, -1.0), tex_uv: glm::Vec2::new(0.0, 0.0) },
    Vertex { pos: glm::Vec3::new( 1.0, -1.0, -1.0), tex_uv: glm::Vec2::new(1.0, 0.0) },
    Vertex { pos: glm::Vec3::new( 1.0,  1.0, -1.0), tex_uv: glm::Vec2::new(1.0, 1.0) },
    Vertex { pos: glm::Vec3::new(-1.0,  1.0, -1.0), tex_uv: glm::Vec2::new(0.0, 1.0) },

    // Left face
    Vertex { pos: glm::Vec3::new(-1.0, -1.0, -1.0), tex_uv: glm::Vec2::new(0.0, 0.0) },
    Vertex { pos: glm::Vec3::new(-1.0, -1.0,  1.0), tex_uv: glm::Vec2::new(1.0, 0.0) },
    Vertex { pos: glm::Vec3::new(-1.0,  1.0,  1.0), tex_uv: glm::Vec2::new(1.0, 1.0) },
    Vertex { pos: glm::Vec3::new(-1.0,  1.0, -1.0), tex_uv: glm::Vec2::new(0.0, 1.0) },

    // Right face
    Vertex { pos: glm::Vec3::new( 1.0, -1.0, -1.0), tex_uv: glm::Vec2::new(0.0, 0.0) },
    Vertex { pos: glm::Vec3::new( 1.0, -1.0,  1.0), tex_uv: glm::Vec2::new(1.0, 0.0) },
    Vertex { pos: glm::Vec3::new( 1.0,  1.0,  1.0), tex_uv: glm::Vec2::new(1.0, 1.0) },
    Vertex { pos: glm::Vec3::new( 1.0,  1.0, -1.0), tex_uv: glm::Vec2::new(0.0, 1.0) },

    // Top face
    Vertex { pos: glm::Vec3::new(-1.0,  1.0, -1.0), tex_uv: glm::Vec2::new(0.0, 0.0) },
    Vertex { pos: glm::Vec3::new(-1.0,  1.0,  1.0), tex_uv: glm::Vec2::new(1.0, 0.0) },
    Vertex { pos: glm::Vec3::new( 1.0,  1.0,  1.0), tex_uv: glm::Vec2::new(1.0, 1.0) },
    Vertex { pos: glm::Vec3::new( 1.0,  1.0, -1.0), tex_uv: glm::Vec2::new(0.0, 1.0) },

    // Bottom face
    Vertex { pos: glm::Vec3::new(-1.0, -1.0, -1.0), tex_uv: glm::Vec2::new(0.0, 0.0) },
    Vertex { pos: glm::Vec3::new(-1.0, -1.0,  1.0), tex_uv: glm::Vec2::new(1.0, 0.0) },
    Vertex { pos: glm::Vec3::new( 1.0, -1.0,  1.0), tex_uv: glm::Vec2::new(1.0, 1.0) },
    Vertex { pos: glm::Vec3::new( 1.0, -1.0, -1.0), tex_uv: glm::Vec2::new(0.0, 1.0) },
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

#[allow(dead_code)]
struct TransformationData {
    proj_view: glm::Mat4,
    model: glm::Mat4,
    t: f32
}
struct Renderer {
    _entry: ash::Entry,
    instance: ash::Instance,

    debug_utils_loader: ext::debug_utils::Instance,
    debug_messager: vk::DebugUtilsMessengerEXT,

    surface_loader: khr::surface::Instance,
    surface: vk::SurfaceKHR,

    physical_device: vk::PhysicalDevice,
    device: ash::Device,

    queue_family_indices: QueueFamilyIndices,

    graphics_queue: vk::Queue,
    present_queue: vk::Queue,

    swapchain_loader: khr::swapchain::Device,
    swapchain: vk::SwapchainKHR,
    swapchain_images: Vec<vk::Image>,
    swapchain_image_format: vk::Format,
    swapchain_extent: vk::Extent2D,
    swapchain_image_views: Vec<vk::ImageView>,

    depth_image: vk::Image,
    depth_image_memory: vk::DeviceMemory,
    depth_image_view: vk::ImageView,

    uniform_buffers: Vec<vk::Buffer>,
    uniform_buffers_memory: Vec<vk::DeviceMemory>,
    uniform_buffers_mapped: Vec<*mut c_void>,

    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,

    render_pass: vk::RenderPass,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,

    swapchain_framebuffers: Vec<vk::Framebuffer>,
    single_time_command_pool: vk::CommandPool,
    
    texture_image: vk::Image,
    texture_image_memory: vk::DeviceMemory,
    texture_image_view: vk::ImageView,
    texture_sampler: vk::Sampler,

    vertex_buffer_memory: vk::DeviceMemory,
    vertex_buffer: vk::Buffer,
    index_buffer_memory: vk::DeviceMemory,
    index_buffer: vk::Buffer,
    
    command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,

    swap_image_available_semaphores: Vec<vk::Semaphore>,
    render_finished_semaphores: Vec<vk::Semaphore>,
    in_flight_fences: Vec<vk::Fence>,

    current_frame: usize,
    start_time: Duration
} 

impl Renderer {
    fn new(window: &winit::window::Window) -> Self {
        let entry: ash::Entry = unsafe {
            ash::Entry::load().unwrap()
        };
        
        if USE_VALIDATION_LAYERS && !Renderer::check_validation_layer_support(&entry) {
            panic!("requested validation layers but not supported");
        }

        
        let size = window.inner_size();

        let instance = Renderer::create_instance(&entry);
        let debug_utils_loader = ext::debug_utils::Instance::new(&entry, &instance);
        let debug_messager = Renderer::setup_debug_messenger(&debug_utils_loader);
        let surface_loader = khr::surface::Instance::new(&entry, &instance);
        let surface = Renderer::create_surface(&entry, &instance, &window);
        let (physical_device, queue_family_indices) = Renderer::pick_physical_device(&instance, &surface_loader, surface);
        let device = Renderer::create_logical_device(&instance, physical_device, &queue_family_indices);
        let (graphics_queue, present_queue) = Renderer::get_queue_handles(&device, &queue_family_indices);
        let swapchain_loader = khr::swapchain::Device::new(&instance, &device);
        let (swapchain, swapchain_images, swapchain_image_format, swapchain_extent) = Renderer::create_swapchain(&swapchain_loader, size.width, size.height, &surface_loader, surface, physical_device, &queue_family_indices);
        let swapchain_image_views = Renderer::create_image_views(&device, &swapchain_images, swapchain_image_format);
        let depth_image_format = Renderer::find_depth_format(&instance, physical_device);
        let (depth_image, depth_image_memory, depth_image_view) = Renderer::create_depth_resources(&instance, &device, physical_device, swapchain_extent, depth_image_format);
        let single_time_command_pool = Renderer::create_command_pool(&device, vk::CommandPoolCreateFlags::TRANSIENT, &queue_family_indices);
        let (uniform_buffers, uniform_buffers_memory, uniform_buffers_mapped) = Renderer::create_uniform_buffers(&instance, &device, physical_device);
        let (texture_image, texture_image_memory) = Renderer::create_texture_image(&instance, &device, physical_device, single_time_command_pool, graphics_queue);
        let texture_image_view = Renderer::create_texture_image_view(&device, texture_image);
        let texture_sampler = Renderer::create_texture_sampler(&instance, &device, physical_device);
        let descriptor_set_layout = Renderer::create_descriptor_set_layout(&device);
        let descriptor_pool = Renderer::create_descriptor_pool(&device);
        let descriptor_sets = Renderer::create_descriptor_sets(&device, descriptor_pool, &uniform_buffers, texture_sampler, texture_image_view, descriptor_set_layout);
        let render_pass = Renderer::create_render_pass(&device, swapchain_image_format, depth_image_format);
        let (pipeline, pipeline_layout) = Renderer::create_graphics_pipeline(&device, swapchain_extent, render_pass, descriptor_set_layout);
        let swapchain_framebuffers = Renderer::create_framebuffers(&device, &swapchain_image_views, depth_image_view, render_pass, swapchain_extent);
        let (vertex_buffer_memory, vertex_buffer) = Renderer::create_vertex_buffer(&instance, &device, physical_device, single_time_command_pool, graphics_queue);
        let (index_buffer_memory, index_buffer) = Renderer::create_index_buffer(&instance, &device, physical_device, single_time_command_pool, graphics_queue);
        let command_pool = Renderer::create_command_pool(&device, vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER, &queue_family_indices);
        let command_buffers = Renderer::create_command_buffers(&device, command_pool);
        let (swap_image_available_semaphores, render_finished_semaphores, in_flight_fences) = Renderer::create_sync_objects(&device);
        
        Renderer {
            _entry: entry,
            instance,
            debug_utils_loader,
            debug_messager,
            surface_loader,
            surface,
            physical_device,
            device,
            queue_family_indices,
            graphics_queue,
            present_queue,
            swapchain_loader,
            swapchain,
            swapchain_images,
            swapchain_image_format,
            swapchain_extent,
            swapchain_image_views,
            depth_image,
            depth_image_memory,
            depth_image_view,
            render_pass,
            texture_image,
            texture_image_memory,
            texture_image_view,
            texture_sampler,
            uniform_buffers,
            uniform_buffers_memory,
            uniform_buffers_mapped,
            descriptor_set_layout,
            descriptor_pool,
            descriptor_sets,
            pipeline,
            pipeline_layout,
            swapchain_framebuffers,
            single_time_command_pool,
            vertex_buffer_memory,
            vertex_buffer,
            index_buffer_memory,
            index_buffer,
            command_pool,
            command_buffers,
            swap_image_available_semaphores,
            render_finished_semaphores,
            in_flight_fences,
            current_frame: 0,
            start_time: SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap()
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


            self.device.reset_command_buffer(self.command_buffers[self.current_frame], vk::CommandBufferResetFlags::empty()).unwrap();
            self.record_command_buffer(image_index);
        }

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

    fn setup_debug_messenger(debug_utils_loader: &ext::debug_utils::Instance) -> vk::DebugUtilsMessengerEXT {
        if USE_VALIDATION_LAYERS == false {
            return vk::DebugUtilsMessengerEXT::null();
        }

        let mut debug_create_info = vk::DebugUtilsMessengerCreateInfoEXT{..Default::default()};
        populate_debug_messenger_create_info(&mut debug_create_info);

        unsafe {
            debug_utils_loader
                .create_debug_utils_messenger(&debug_create_info, None)
                .unwrap()
        }
    }

    fn check_validation_layer_support(entry: &ash::Entry) -> bool {
        let supported_layers: Vec<vk::LayerProperties> = unsafe {
            entry.enumerate_instance_layer_properties().unwrap()
        };

        if supported_layers.len() <= 0 {
            warn!("no supported layers found");
            return false;
        }

        for required_layer_name in REQUIRED_VALIDATION_LAYER_NAMES.iter() {
            let mut is_layer_found = false;

            for layer in supported_layers.iter() {
                let layer_name: &str = unsafe {
                    CStr::from_ptr(layer.layer_name.as_ptr()).to_str().unwrap()
                };
                
                if layer_name == *required_layer_name {
                    is_layer_found = true;
                    break;
                }
            }

            if is_layer_found == false {
                return false;
            }
        }

        true
    }

    fn create_instance(entry: &ash::Entry) -> ash::Instance {
        let application_name: CString = CString::new("app name").unwrap();
        let engine_name: CString = CString::new("engine name").unwrap();

        let app_info = vk::ApplicationInfo {
            s_type: vk::StructureType::APPLICATION_INFO,
            p_next: ptr::null(),
            p_application_name: application_name.as_ptr(),
            application_version: vk::make_api_version(0, 1, 0, 0),
            p_engine_name: engine_name.as_ptr(),
            engine_version: 0,
            api_version: vk::API_VERSION_1_0,
            ..Default::default()
        };

        #[cfg(target_os = "windows")]
        let mut extension_names: Vec<*const c_char> = vec![
            khr::surface::NAME.as_ptr(),
            khr::win32_surface::NAME.as_ptr(),
        ];

        if USE_VALIDATION_LAYERS {
            extension_names.push(ext::debug_utils::NAME.as_ptr());
        }

        let cstr_required_validation_layer_names: Vec<CString> = REQUIRED_VALIDATION_LAYER_NAMES
            .iter()
            .map(|layer_name: &&str| CString::new(*layer_name).unwrap())
            .collect();

        let p_required_validation_layer_names: Vec<*const i8> = cstr_required_validation_layer_names
            .iter()
            .map(|layer_name: &CString| layer_name.as_ptr())
            .collect();

        let mut create_info = vk::InstanceCreateInfo {
            s_type: vk::StructureType::INSTANCE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::InstanceCreateFlags::empty(),
            p_application_info: &app_info,
            pp_enabled_extension_names: extension_names.as_ptr(),
            pp_enabled_layer_names: ptr::null(),
            enabled_extension_count: extension_names.len() as u32,
            enabled_layer_count: 0,
            ..Default::default()
        };

        let mut debug_create_info = vk::DebugUtilsMessengerCreateInfoEXT{..Default::default()};
        if USE_VALIDATION_LAYERS {
            populate_debug_messenger_create_info(&mut debug_create_info);
            create_info.p_next = &debug_create_info as *const vk::DebugUtilsMessengerCreateInfoEXT as *const c_void;
            create_info.pp_enabled_layer_names = p_required_validation_layer_names.as_ptr();
            create_info.enabled_layer_count = p_required_validation_layer_names.len() as u32;
        }
   
        unsafe {
            entry.create_instance(&create_info, None).unwrap()
        }
    }

    fn pick_physical_device(instance: &ash::Instance, surface_loader: &khr::surface::Instance, surface: vk::SurfaceKHR) -> (vk::PhysicalDevice, QueueFamilyIndices) {
        let devices: Vec<vk::PhysicalDevice> = unsafe {
            instance.enumerate_physical_devices().unwrap()
        };

        if devices.len() <= 0 {
            panic!("no devices found");
        }

        for device in devices {
            let queue_family_indices: QueueFamilyIndices = Renderer::find_queue_families(instance, surface_loader, surface, device);

            if Renderer::is_device_suitable(&instance, surface_loader, surface, device, &queue_family_indices) {
                return (device, queue_family_indices);
            }
        }

        panic!("failed to find suitable device");
    }

    fn is_device_suitable(instance: &ash::Instance, surface_loader: &khr::surface::Instance, surface: vk::SurfaceKHR, device: vk::PhysicalDevice, queue_family_indices: &QueueFamilyIndices) -> bool {
        let _properties: vk::PhysicalDeviceProperties = unsafe {
            instance.get_physical_device_properties(device)
        };

        let features: vk::PhysicalDeviceFeatures = unsafe {
            instance.get_physical_device_features(device)
        };

        let is_extensions_supported: bool = Renderer::check_device_extension_support(instance, device);
        let mut is_swapchain_adequate: bool = false;

        if is_extensions_supported {
            let swapchain: SwapChainSupportDetails = Renderer::query_swapchain_support(surface_loader, surface, device);
            is_swapchain_adequate = !swapchain.formats.is_empty() && !swapchain.present_modes.is_empty();
        }

        queue_family_indices.graphics.is_some() && queue_family_indices.present.is_some() && is_extensions_supported && is_swapchain_adequate && features.sampler_anisotropy == vk::TRUE
    }

    fn find_queue_families(instance: &ash::Instance, surface_loader: &khr::surface::Instance, surface: vk::SurfaceKHR, device: vk::PhysicalDevice) -> QueueFamilyIndices {
        let queue_families: Vec<vk::QueueFamilyProperties> = unsafe {
            instance.get_physical_device_queue_family_properties(device)
        };

        let mut indices: QueueFamilyIndices = Default::default();

        let mut i: u32 = 0;
        for family in queue_families {
            if family.queue_count == 0 {
                continue;
            }

            if family.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                indices.graphics = Some(i);
            }

            let present_support: bool = unsafe {
                surface_loader.get_physical_device_surface_support(device, i, surface).unwrap()
            };

            if present_support {
                indices.present = Some(i);
            }

            if indices.graphics.is_some() && indices.present.is_some() {
                break;
            }

            i += 1;
        }
        
        indices
    }

    fn create_logical_device(instance: &ash::Instance, physical_device: vk::PhysicalDevice, queue_family_indices: &QueueFamilyIndices) -> ash::Device {
        let queue_priority: f32 = 1.0;

        
        let mut queue_create_infos: Vec<vk::DeviceQueueCreateInfo> = vec![];
        let mut unique_indices: HashSet<u32> = HashSet::new();
        unique_indices.insert(queue_family_indices.graphics.unwrap());
        unique_indices.insert(queue_family_indices.present.unwrap());

        for queue_family_index in unique_indices {
            let queue_create_info = vk::DeviceQueueCreateInfo {
                s_type: vk::StructureType::DEVICE_QUEUE_CREATE_INFO,
                p_next: ptr::null(),
                flags: vk::DeviceQueueCreateFlags::empty(),
                queue_family_index: queue_family_index,
                queue_count: 1,
                p_queue_priorities: &queue_priority,
                ..Default::default()
            };

            queue_create_infos.push(queue_create_info);
        }

        let physical_device_features: vk::PhysicalDeviceFeatures = vk::PhysicalDeviceFeatures {
            sampler_anisotropy: vk::TRUE,
            ..Default::default()
        };

        let create_info = vk::DeviceCreateInfo {
            s_type: vk::StructureType::DEVICE_CREATE_INFO,
            p_next: ptr::null(),
            p_queue_create_infos: queue_create_infos.as_ptr(),
            flags: vk::DeviceCreateFlags::empty(),
            queue_create_info_count: queue_create_infos.len() as u32,
            enabled_extension_count: REQUIRED_DEVICE_EXTENSIONS.len() as u32,
            pp_enabled_extension_names: REQUIRED_DEVICE_EXTENSIONS.as_ptr(),
            p_enabled_features: &physical_device_features,
            ..Default::default()
        };

        unsafe {
            instance.create_device(physical_device, &create_info, None).unwrap()
        }
    }

    fn get_queue_handles(device: &ash::Device, queue_family_indices: &QueueFamilyIndices) -> (vk::Queue, vk::Queue) {
        unsafe {
            (
                device.get_device_queue(queue_family_indices.graphics.unwrap(), 0),
                device.get_device_queue(queue_family_indices.present.unwrap(), 0)
            )
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

    fn check_device_extension_support(instance: &ash::Instance, device: vk::PhysicalDevice) -> bool {
        let supported_extensions: Vec<vk::ExtensionProperties> = unsafe {
            instance.enumerate_device_extension_properties(device).unwrap()
        };

        let mut required_extensions: HashSet<&CStr> = HashSet::new();
        for extension in REQUIRED_DEVICE_EXTENSIONS {
            required_extensions.insert(unsafe {
                CStr::from_ptr(extension)
            });
        }

        for extension in supported_extensions {
            let name: &CStr = unsafe {
                CStr::from_ptr(extension.extension_name.as_ptr())
            };
            required_extensions.remove(name);
        }
        
        required_extensions.len() == 0
    }

    
    fn query_swapchain_support(surface_loader: &khr::surface::Instance, surface: vk::SurfaceKHR, device: vk::PhysicalDevice) -> SwapChainSupportDetails {
        SwapChainSupportDetails {
            capabilities: unsafe {
                surface_loader.get_physical_device_surface_capabilities(device, surface).unwrap()
            },
            formats: unsafe {
                surface_loader.get_physical_device_surface_formats(device, surface).unwrap()
            },
            present_modes: unsafe {
                surface_loader.get_physical_device_surface_present_modes(device, surface).unwrap()
            }
        }
    }

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

    fn create_swapchain(swapchain_loader: &khr::swapchain::Device, width: u32, height: u32, surface_loader: &khr::surface::Instance, surface: vk::SurfaceKHR, device: vk::PhysicalDevice, queue_family_indices: &QueueFamilyIndices) -> (vk::SwapchainKHR, Vec<vk::Image>, vk::Format, vk::Extent2D) {
        let swapchain: SwapChainSupportDetails = Renderer::query_swapchain_support(surface_loader, surface, device);

        let surface_format: vk::SurfaceFormatKHR = Renderer::choose_swap_surface_format(&swapchain.formats);
        let present_mode: vk::PresentModeKHR = Renderer::choose_swap_present_mode(&swapchain.present_modes);
        let extent: vk::Extent2D = Renderer::choose_swap_extent(width, height, &swapchain.capabilities);
         
        let mut image_count: u32 = swapchain.capabilities.min_image_count + 1;
        if image_count > swapchain.capabilities.max_image_count {
            image_count = swapchain.capabilities.min_image_count;
        }

        let mut create_info = vk::SwapchainCreateInfoKHR {
            s_type: vk::StructureType::SWAPCHAIN_CREATE_INFO_KHR,
            p_next: ptr::null(),
            min_image_count: image_count,
            image_format: surface_format.format,
            flags: vk::SwapchainCreateFlagsKHR::empty(),
            surface: surface,
            image_color_space: surface_format.color_space,
            image_extent: extent,
            image_array_layers: 1,
            image_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
            image_sharing_mode: vk::SharingMode::EXCLUSIVE,
            queue_family_index_count: 0,
            p_queue_family_indices: ptr::null(),
            pre_transform: swapchain.capabilities.current_transform,
            composite_alpha: vk::CompositeAlphaFlagsKHR::OPAQUE,
            present_mode: present_mode,
            clipped: vk::TRUE,
            old_swapchain: vk::SwapchainKHR::null(), // for now assume we will only create one swapchain
            ..Default::default()
        };

        let indices: [u32; 2] = [queue_family_indices.graphics.unwrap(), queue_family_indices.present.unwrap()];

        if queue_family_indices.graphics != queue_family_indices.present {
            create_info.image_sharing_mode = vk::SharingMode::CONCURRENT;
            create_info.queue_family_index_count = indices.len() as u32;
            create_info.p_queue_family_indices = indices.as_ptr();
        } 

        let swapchain: vk::SwapchainKHR = unsafe {
            swapchain_loader.create_swapchain(&create_info, None).unwrap()
        };

        let swap_chain_images: Vec<vk::Image> = unsafe {
            swapchain_loader.get_swapchain_images(swapchain).unwrap()
        };

        let swapchain_image_format = surface_format.format;
        let swapchain_extent = extent;
        
        (swapchain, swap_chain_images, swapchain_image_format, swapchain_extent)
    }

    fn create_image_views(device: &ash::Device, swapchain_images: &Vec<vk::Image>, swapchain_image_format: vk::Format) -> Vec<vk::ImageView> {
        let mut swapchain_image_views: Vec<vk::ImageView> = Vec::with_capacity(swapchain_images.len());

        for image in swapchain_images.iter() {
            swapchain_image_views.push(Renderer::create_image_view(device, *image, swapchain_image_format, vk::ImageAspectFlags::COLOR));
        }

        swapchain_image_views
    }
    
    fn create_render_pass(device: &ash::Device, swapchain_image_format: vk::Format, depth_image_format: vk::Format) -> vk::RenderPass {
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

        let subpass = vk::SubpassDescription {
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
            src_access_mask: vk::AccessFlags::empty(),
            dst_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
            dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            dependency_flags: vk::DependencyFlags::empty()
        };

        let attachments: [vk::AttachmentDescription; 2] = [color_attachment, depth_attachment];

        let render_pass_create_info = vk::RenderPassCreateInfo {
            s_type: vk::StructureType::RENDER_PASS_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::RenderPassCreateFlags::empty(),
            attachment_count: attachments.len() as u32,
            p_attachments: attachments.as_ptr(),
            subpass_count: 1,
            p_subpasses: &subpass,
            p_dependencies: &dependency,
            dependency_count: 1,
            ..Default::default()
        };

        unsafe {
            device.create_render_pass(&render_pass_create_info, None).unwrap()
        }
    }

    fn create_graphics_pipeline(device: &ash::Device, swapchain_extent: vk::Extent2D, render_pass: vk::RenderPass, descriptor_set_layout: vk::DescriptorSetLayout) -> (vk::Pipeline, vk::PipelineLayout) {
        let vert_bytes: Vec<u8> = fs::read("src/shader.vert.spv").unwrap();
        let frag_bytes: Vec<u8> = fs::read("src/shader.frag.spv").unwrap();

        let vert_shader: vk::ShaderModule = Renderer::create_shader_module(device, &vert_bytes);
        let frag_shader: vk::ShaderModule = Renderer::create_shader_module(device, &frag_bytes);

        let vert_shader_stage_create_info = vk::PipelineShaderStageCreateInfo {
            s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
            p_name: c"main".as_ptr(),
            p_next: ptr::null(),
            flags: vk::PipelineShaderStageCreateFlags::empty(),
            stage: vk::ShaderStageFlags::VERTEX,
            module: vert_shader,
            p_specialization_info: ptr::null(), // for constexpr's in shader
            ..Default::default()
        };

        let frag_shader_stage_create_info = vk::PipelineShaderStageCreateInfo {
            s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
            p_name: c"main".as_ptr(),
            p_next: ptr::null(),
            flags: vk::PipelineShaderStageCreateFlags::empty(),
            stage: vk::ShaderStageFlags::FRAGMENT,
            module: frag_shader,
            p_specialization_info: ptr::null(), // for constexpr's in shader
            ..Default::default()
        };

        let shader_stage_create_infos: [vk::PipelineShaderStageCreateInfo; 2] = [vert_shader_stage_create_info, frag_shader_stage_create_info];

        
        let vertex_binding_description: vk::VertexInputBindingDescription = Vertex::get_binding_description();
        let vertex_attribute_description: Vec<vk::VertexInputAttributeDescription> = Vertex::get_attribute_description();

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

        let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo {
            s_type: vk::StructureType::PIPELINE_LAYOUT_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::PipelineLayoutCreateFlags::empty(),
            set_layout_count: 1,
            p_set_layouts: &descriptor_set_layout,
            p_push_constant_ranges: ptr::null(),
            push_constant_range_count: 0,
            ..Default::default()
        };

        let pipeline_layout: vk::PipelineLayout = unsafe {
            device.create_pipeline_layout(&pipeline_layout_create_info, None).unwrap()
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
            device.create_graphics_pipelines(vk::PipelineCache::null(), 
            graphics_pipeline_create_infos.as_slice(),
            None).unwrap()
        };

        unsafe {
            device.destroy_shader_module(vert_shader, None);
            device.destroy_shader_module(frag_shader, None);
        }


        (graphics_pipelines[0], pipeline_layout)
    }

    fn create_shader_module(device: &ash::Device, bytes: &Vec<u8>) -> vk::ShaderModule {
        let create_info = vk::ShaderModuleCreateInfo {
            s_type: vk::StructureType::SHADER_MODULE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::ShaderModuleCreateFlags::empty(),
            code_size: bytes.len(),
            p_code: bytes.as_ptr() as *const u32,
            ..Default::default()
        };

        unsafe {
            device.create_shader_module(&create_info, None).unwrap()
        }
    }

    fn create_framebuffers(device: &ash::Device, swapchain_image_views: &Vec<vk::ImageView>, depth_image_view: vk::ImageView, render_pass: vk::RenderPass, swapchain_extent: vk::Extent2D) -> Vec<vk::Framebuffer> {
        let mut swapchain_framebuffers: Vec<vk::Framebuffer> = Vec::with_capacity(swapchain_image_views.len());

        for image_view in swapchain_image_views {
            let attachments: [vk::ImageView; 2] = [
                *image_view,
                depth_image_view
            ];

            let create_info = vk::FramebufferCreateInfo {
                s_type: vk::StructureType::FRAMEBUFFER_CREATE_INFO,
                p_next: ptr::null(),
                flags: vk::FramebufferCreateFlags::empty(),
                render_pass: render_pass,
                attachment_count: attachments.len() as u32,
                p_attachments: attachments.as_ptr(),
                width: swapchain_extent.width,
                height: swapchain_extent.height,
                layers: 1,
                ..Default::default()
            };

            swapchain_framebuffers.push(unsafe {
                device.create_framebuffer(&create_info, None).unwrap()
            });
        }

        swapchain_framebuffers
    }

    fn create_command_pool(device: &ash::Device, flags: vk::CommandPoolCreateFlags, queue_family_indices: &QueueFamilyIndices) -> vk::CommandPool {
        let create_info = vk::CommandPoolCreateInfo {
            s_type: vk::StructureType::COMMAND_POOL_CREATE_INFO,
            p_next: ptr::null(),
            flags: flags,
            queue_family_index: queue_family_indices.graphics.unwrap(),
            ..Default::default()
        };

        unsafe {
            device.create_command_pool(&create_info, None).unwrap()
        }
    }

    fn create_command_buffers(device: &ash::Device, command_pool: vk::CommandPool) -> Vec<vk::CommandBuffer> {
        let allocate_info = vk::CommandBufferAllocateInfo {
            s_type: vk::StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
            p_next: ptr::null(),
            command_pool: command_pool,
            level: vk::CommandBufferLevel::PRIMARY,
            command_buffer_count: MAX_FRAMES_IN_FLIGHT as u32,
            ..Default::default()
        };

        unsafe {
            device.allocate_command_buffers(&allocate_info).unwrap()
        }
    }

    fn record_command_buffer(&self, swap_image_index: u32) {
        let command_buffer_begin_info = vk::CommandBufferBeginInfo {
            s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
            p_next: ptr::null(),
            flags: vk::CommandBufferUsageFlags::empty(),
            p_inheritance_info: ptr::null(),
            ..Default::default()
        };

        unsafe {
            self.device.begin_command_buffer(self.command_buffers[self.current_frame], &command_buffer_begin_info).unwrap()
        }

        let clear_values: [vk::ClearValue; 2] = [
            vk::ClearValue {color: vk::ClearColorValue {float32: [0.0, 0.0, 0.0, 1.0]}},
            vk::ClearValue {depth_stencil: vk::ClearDepthStencilValue {depth: 1.0, stencil: 0}}
        ];

        let render_pass_begin_info = vk::RenderPassBeginInfo {
            s_type: vk::StructureType::RENDER_PASS_BEGIN_INFO,
            p_next: ptr::null(),
            render_pass: self.render_pass,
            framebuffer: self.swapchain_framebuffers[swap_image_index as usize],
            render_area: vk::Rect2D {
                offset: vk::Offset2D {x: 0, y: 0},
                extent: self.swapchain_extent
            },
            clear_value_count: clear_values.len() as u32,
            p_clear_values: clear_values.as_ptr(),
            ..Default::default()
        };

        unsafe {
            self.device.cmd_begin_render_pass(self.command_buffers[self.current_frame], &render_pass_begin_info, vk::SubpassContents::INLINE);
            self.device.cmd_bind_pipeline(self.command_buffers[self.current_frame], vk::PipelineBindPoint::GRAPHICS, self.pipeline);
            
            let offsets: [u64; 1] = [0];
            self.device.cmd_bind_vertex_buffers(self.command_buffers[self.current_frame], 0, std::slice::from_ref(&self.vertex_buffer), &offsets);
            self.device.cmd_bind_index_buffer(self.command_buffers[self.current_frame], self.index_buffer, 0, vk::IndexType::UINT16);

            let viewport = vk::Viewport{
                x: 0.0,
                y: 0.0,
                width: self.swapchain_extent.width as f32,
                height: self.swapchain_extent.height as f32,
                min_depth: 0.0,
                max_depth: 1.0
            };

            self.device.cmd_set_viewport(self.command_buffers[self.current_frame], 0, std::slice::from_ref(&viewport));

            let scissor = vk::Rect2D {
                offset: vk::Offset2D {x: 0, y: 0},
                extent: self.swapchain_extent,
            };

            self.device.cmd_set_scissor(self.command_buffers[self.current_frame], 0, std::slice::from_ref(&scissor));

            self.device.cmd_bind_descriptor_sets(self.command_buffers[self.current_frame], vk::PipelineBindPoint::GRAPHICS, self.pipeline_layout, 0, std::slice::from_ref(&self.descriptor_sets[self.current_frame]), &[]);
            self.device.cmd_draw_indexed(self.command_buffers[self.current_frame], INDICES.len() as u32, 1, 0, 0, 0);
            self.device.cmd_end_render_pass(self.command_buffers[self.current_frame]);
            self.device.end_command_buffer(self.command_buffers[self.current_frame]).unwrap();
        }
    }

    fn create_sync_objects(device: &ash::Device) -> (Vec<vk::Semaphore>, Vec<vk::Semaphore>, Vec<vk::Fence>) {
        let semamphore_create_info = vk::SemaphoreCreateInfo {
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

        let mut swap_image_available_semaphores: Vec<vk::Semaphore> = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
        let mut render_finished_semaphores: Vec<vk::Semaphore> = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
        let mut in_flight_fences: Vec<vk::Fence> = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
        

        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            unsafe {
                swap_image_available_semaphores.push(device.create_semaphore(&semamphore_create_info, None).unwrap());
                render_finished_semaphores.push(device.create_semaphore(&semamphore_create_info, None).unwrap());
                in_flight_fences.push(device.create_fence(&fence_create_info, None).unwrap());
            }
        }

        (swap_image_available_semaphores, render_finished_semaphores, in_flight_fences)
    }

    fn recreate_swapchain(&mut self, size: winit::dpi::PhysicalSize<u32>) {
        unsafe {
            self.device.device_wait_idle().unwrap();
        }

        self.cleanup_swapchain();
        (self.swapchain, self.swapchain_images, self.swapchain_image_format, self.swapchain_extent) = Renderer::create_swapchain(&self.swapchain_loader, size.width, size.height, &self.surface_loader, self.surface, self.physical_device, &self.queue_family_indices);
        self.swapchain_image_views = Renderer::create_image_views(&self.device, &self.swapchain_images, self.swapchain_image_format);
        self.swapchain_framebuffers = Renderer::create_framebuffers(&self.device, &self.swapchain_image_views, self.depth_image_view, self.render_pass, self.swapchain_extent);
    }

    fn cleanup_swapchain(&self) {
        unsafe {
            for framebuffer in self.swapchain_framebuffers.iter() {
                self.device.destroy_framebuffer(*framebuffer, None);
            }

            for image_view in self.swapchain_image_views.iter() {
                self.device.destroy_image_view(*image_view, None);
            }
            
            self.swapchain_loader.destroy_swapchain(self.swapchain, None);
        }
    }

    fn find_memory_type(instance: &ash::Instance, physical_device: vk::PhysicalDevice, suitable_memory_bits: u32, required_properties: vk::MemoryPropertyFlags) -> u32 {

        let memory_properties: vk::PhysicalDeviceMemoryProperties = unsafe {
            instance.get_physical_device_memory_properties(physical_device)
        };

        for i in 0..memory_properties.memory_type_count as usize {
            if suitable_memory_bits & (1 << i) != 0 && memory_properties.memory_types[i].property_flags.contains(required_properties) {
                return i as u32;
            }
        }

        panic!("failed to find suitable memory")
    }

    fn create_buffer(instance: &ash::Instance, device: &ash::Device, physical_device: vk::PhysicalDevice, size: vk::DeviceSize, usage: vk::BufferUsageFlags, properties: vk::MemoryPropertyFlags) -> (vk::DeviceMemory, vk::Buffer) {
        let create_info = vk::BufferCreateInfo {
            s_type: vk::StructureType::BUFFER_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::BufferCreateFlags::empty(),
            size: size,
            usage: usage,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            queue_family_index_count: 0,
            p_queue_family_indices: ptr::null(),
            ..Default::default()
        };

        let buffer: vk::Buffer = unsafe {
            device.create_buffer(&create_info, None).unwrap()
        };

        let mem_requirements: vk::MemoryRequirements = unsafe {
            device.get_buffer_memory_requirements(buffer)
        };

        let allocate_info = vk::MemoryAllocateInfo {
            s_type: vk::StructureType::MEMORY_ALLOCATE_INFO,
            p_next: ptr::null(),
            allocation_size: mem_requirements.size,
            memory_type_index: Renderer::find_memory_type(instance, physical_device, mem_requirements.memory_type_bits, properties),
            ..Default::default()
        }; 

        let buffer_memory: vk::DeviceMemory = unsafe {
            device.allocate_memory(&allocate_info, None).unwrap()
        };

        unsafe {    
            device.bind_buffer_memory(buffer, buffer_memory, 0).unwrap();
        };

        (buffer_memory, buffer)
    }

    fn create_vertex_buffer(instance: &ash::Instance, device: &ash::Device, physical_device: vk::PhysicalDevice, command_pool: vk::CommandPool, graphics_queue: vk::Queue) -> (vk::DeviceMemory, vk::Buffer) {
        let buffer_size: vk::DeviceSize = (size_of::<Vertex>() * VERTICES.len()) as u64;
        let (staging_buffer_memory, staging_buffer): (vk::DeviceMemory, vk::Buffer) = Renderer::create_buffer(instance, device, physical_device, buffer_size, vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::VERTEX_BUFFER, vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT);

        unsafe {
            let data_ptr: *mut Vertex = device.map_memory(staging_buffer_memory, 0, buffer_size, vk::MemoryMapFlags::empty()).unwrap() as *mut Vertex;
            data_ptr.copy_from_nonoverlapping(VERTICES.as_ptr(), VERTICES.len());
            device.unmap_memory(staging_buffer_memory);
        };

        let (vertex_buffer_memory, vertex_buffer): (vk::DeviceMemory, vk::Buffer) = Renderer::create_buffer(instance, device, physical_device, buffer_size, vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER, vk::MemoryPropertyFlags::DEVICE_LOCAL);

        Renderer::copy_buffer(device, staging_buffer, vertex_buffer, buffer_size, command_pool, graphics_queue);

        unsafe {
            device.destroy_buffer(staging_buffer, None);
            device.free_memory(staging_buffer_memory, None);
        }

        (vertex_buffer_memory, vertex_buffer)
    }

    fn create_index_buffer(instance: &ash::Instance, device: &ash::Device, physical_device: vk::PhysicalDevice, command_pool: vk::CommandPool, graphics_queue: vk::Queue) -> (vk::DeviceMemory, vk::Buffer) {
        let buffer_size: vk::DeviceSize = (size_of::<u16>() * INDICES.len()) as u64;
        let (staging_buffer_memory, staging_buffer): (vk::DeviceMemory, vk::Buffer) = Renderer::create_buffer(instance, device, physical_device, buffer_size, vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::INDEX_BUFFER, vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT);

        unsafe {
            let data_ptr: *mut u16 = device.map_memory(staging_buffer_memory, 0, buffer_size, vk::MemoryMapFlags::empty()).unwrap() as *mut u16;
            data_ptr.copy_from_nonoverlapping(INDICES.as_ptr(), INDICES.len());
            device.unmap_memory(staging_buffer_memory);
        };

        let (index_buffer_memory, index_buffer): (vk::DeviceMemory, vk::Buffer) = Renderer::create_buffer(instance, device, physical_device, buffer_size, vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER, vk::MemoryPropertyFlags::DEVICE_LOCAL);

        Renderer::copy_buffer(device, staging_buffer, index_buffer, buffer_size, command_pool, graphics_queue);

        unsafe {
            device.destroy_buffer(staging_buffer, None);
            device.free_memory(staging_buffer_memory, None);
        }

        (index_buffer_memory, index_buffer)
    }
    
    // graphics queue guarantees VK_TRANSFER_QUEUE
    fn copy_buffer(device: &ash::Device, src_buffer: vk::Buffer, dst_buffer: vk::Buffer, size: vk::DeviceSize, command_pool: vk::CommandPool, graphics_queue: vk::Queue) {
        let command_buffer: vk::CommandBuffer = Renderer::begin_single_time_commands(device, command_pool);
        let region = vk::BufferCopy {
            src_offset: 0,
            dst_offset: 0,
            size: size
        };
        
        unsafe {
            device.cmd_copy_buffer(command_buffer, src_buffer, dst_buffer, std::slice::from_ref(&region));
        };

        Renderer::end_single_time_commands(device, command_pool, command_buffer, graphics_queue);
    }

    fn create_descriptor_set_layout(device: &ash::Device) -> vk::DescriptorSetLayout {
        let ubo_layout_binding = vk::DescriptorSetLayoutBinding {
            binding: 0,
            descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::VERTEX,
            p_immutable_samplers: ptr::null(),
            ..Default::default()
        };

        let sampler_layout_binding = vk::DescriptorSetLayoutBinding {
            binding: 1,
            descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::FRAGMENT,
            p_immutable_samplers: ptr::null(),
            ..Default::default()
        };

        let bindings: [vk::DescriptorSetLayoutBinding; 2] = [ubo_layout_binding, sampler_layout_binding];

        let create_info = vk::DescriptorSetLayoutCreateInfo {
            s_type: vk::StructureType::DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::DescriptorSetLayoutCreateFlags::empty(),
            binding_count: bindings.len() as u32,
            p_bindings: bindings.as_ptr(),
            ..Default::default()
        };
        
        unsafe {
            device.create_descriptor_set_layout(&create_info, None).unwrap()
        }
    }

    fn create_uniform_buffers(instance: &ash::Instance, device: &ash::Device, physical_device: vk::PhysicalDevice) -> (Vec<vk::Buffer>, Vec<vk::DeviceMemory>, Vec<*mut c_void>) {
        let buffer_size: vk::DeviceSize = size_of::<TransformationData>() as u64;
        
        let mut uniform_buffers: Vec<vk::Buffer> = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
        let mut uniform_buffers_memory: Vec<vk::DeviceMemory> = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
        let mut uniform_buffers_mapped: Vec<*mut c_void> = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);

        for i in  0..MAX_FRAMES_IN_FLIGHT {
            let (buffer_memory, buffer) = Renderer::create_buffer(instance, device, physical_device, buffer_size, vk::BufferUsageFlags::UNIFORM_BUFFER, vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT);
            uniform_buffers.push(buffer);
            uniform_buffers_memory.push(buffer_memory);

            uniform_buffers_mapped.push(unsafe {
                device.map_memory(uniform_buffers_memory[i], 0, buffer_size, vk::MemoryMapFlags::empty()).unwrap()
            });
        }

        (uniform_buffers, uniform_buffers_memory, uniform_buffers_mapped)
    }

    fn update_uniform_buffers(&self) {
        let now = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap();

        let ubo = TransformationData {
            proj_view: glm::perspective(self.swapchain_extent.width as f32 / self.swapchain_extent.height as f32, glm::pi::<f32>() / 4.0, 0.01, 1000.0) * glm::look_at(&glm::vec3(0.0, 0.0, -10.5), &glm::vec3(0.0, 0.0, 0.0), &glm::vec3(0.0, 1.0, 0.0)),
            model: glm::rotate(&glm::rotate(&glm::identity::<f32, 4>(), (now - self.start_time).as_millis() as f32 / 1000.0, &glm::vec3(1.0, 0.0, 0.0)), (now - self.start_time).as_millis() as f32 / 1000.0, &glm::vec3(0.0, 1.0, 0.0)),
            t: (now - self.start_time).as_millis() as f32 / 1000.0
        };

        
        unsafe {
            self.uniform_buffers_mapped[self.current_frame].copy_from_nonoverlapping(&ubo as *const TransformationData as *const c_void, size_of::<TransformationData>());
        }
    }

    fn create_descriptor_pool(device: &ash::Device) -> vk::DescriptorPool {
        let ubo_pool_size = vk::DescriptorPoolSize {
            ty: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: MAX_FRAMES_IN_FLIGHT as u32
        };

        let sampler_pool_size = vk::DescriptorPoolSize {
            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: MAX_FRAMES_IN_FLIGHT as u32
        };

        let pool_sizes: [vk::DescriptorPoolSize; 2] = [ubo_pool_size, sampler_pool_size];

        let create_info = vk::DescriptorPoolCreateInfo {
            s_type: vk::StructureType::DESCRIPTOR_POOL_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::DescriptorPoolCreateFlags::empty(),
            pool_size_count: pool_sizes.len() as u32,
            p_pool_sizes: pool_sizes.as_ptr(),
            max_sets: MAX_FRAMES_IN_FLIGHT as u32,
            ..Default::default()
        };

        unsafe {
            device.create_descriptor_pool(&create_info, None).unwrap()
        }
    }

    fn create_descriptor_sets(device: &ash::Device, descriptor_pool: vk::DescriptorPool, uniform_buffers: &Vec<vk::Buffer>, sampler: vk::Sampler, texture_image_view: vk::ImageView, descriptor_set_layout: vk::DescriptorSetLayout) -> Vec<vk::DescriptorSet> {
        let layouts: Vec<vk::DescriptorSetLayout> = vec![descriptor_set_layout; MAX_FRAMES_IN_FLIGHT];
        
        let allocate_info = vk::DescriptorSetAllocateInfo {
            s_type: vk::StructureType::DESCRIPTOR_SET_ALLOCATE_INFO,
            p_next: ptr::null(),
            descriptor_pool: descriptor_pool,
            descriptor_set_count: MAX_FRAMES_IN_FLIGHT as u32,
            p_set_layouts: layouts.as_ptr(),
            ..Default::default()
        };

        let descriptor_sets: Vec<vk::DescriptorSet> = unsafe {
            device.allocate_descriptor_sets(&allocate_info).unwrap()
        };

        for i in 0..MAX_FRAMES_IN_FLIGHT {
            let buffer_info = vk::DescriptorBufferInfo {
                buffer: uniform_buffers[i],
                offset: 0,
                range: vk::WHOLE_SIZE
            };

            let image_info = vk::DescriptorImageInfo {
                sampler: sampler,
                image_view: texture_image_view,
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            };
            
            let buffer_descriptor_write = vk::WriteDescriptorSet {
                s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                p_next: ptr::null(),
                dst_set: descriptor_sets[i],
                dst_binding: 0,
                dst_array_element: 0,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                p_image_info: ptr::null(),
                p_buffer_info: &buffer_info,
                p_texel_buffer_view: ptr::null(),
                ..Default::default()
            };

            let image_descriptor_write = vk::WriteDescriptorSet {
                s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                p_next: ptr::null(),
                dst_set: descriptor_sets[i],
                dst_binding: 1,
                dst_array_element: 0,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                p_image_info: &image_info,
                p_buffer_info: ptr::null(),
                p_texel_buffer_view: ptr::null(),
                ..Default::default()
            };

            let descriptor_writes: [vk::WriteDescriptorSet; 2] = [buffer_descriptor_write, image_descriptor_write];

            unsafe {
                device.update_descriptor_sets(&descriptor_writes, &[]);
            }
        }

        descriptor_sets
    }

    fn create_texture_sampler(instance: &ash::Instance, device: &ash::Device, physical_device: vk::PhysicalDevice) -> vk::Sampler {
        let properties: vk::PhysicalDeviceProperties = unsafe {
            instance.get_physical_device_properties(physical_device)
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
            device.create_sampler(&create_info, None).unwrap()
        }
    }

    fn create_image_view(device: &ash::Device, image: vk::Image, format: vk::Format, aspect_mask: vk::ImageAspectFlags) -> vk::ImageView {
        let create_info = vk::ImageViewCreateInfo {
            s_type: vk::StructureType::IMAGE_VIEW_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::ImageViewCreateFlags::empty(),
            image: image,
            view_type: vk::ImageViewType::TYPE_2D,
            format: format,
            components: vk::ComponentMapping {
                r: vk::ComponentSwizzle::R,
                g: vk::ComponentSwizzle::G,
                b: vk::ComponentSwizzle::B,
                a: vk::ComponentSwizzle::A
            },
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: aspect_mask,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1
            },
            ..Default::default()
        };

        unsafe {
            device.create_image_view(&create_info, None).unwrap()
        }
    }

    fn create_texture_image_view(device: &ash::Device, texture_image: vk::Image) -> vk::ImageView {
        Renderer::create_image_view(device, texture_image, vk::Format::R8G8B8A8_SRGB, vk::ImageAspectFlags::COLOR)
    }

    fn create_texture_image(instance: &ash::Instance, device: &ash::Device, physical_device: vk::PhysicalDevice, single_time_command_pool: vk::CommandPool, graphics_queue: vk::Queue) -> (vk::Image, vk::DeviceMemory) {
        let image= image::ImageReader::open("image.png").unwrap().decode().unwrap().into_rgba8();
        let pixels: &[u8] = image.as_bytes();

        let buffer_size: vk::DeviceSize = (size_of::<u8>() * pixels.len()) as u64;

        let (staging_buffer_memory, staging_buffer): (vk::DeviceMemory, vk::Buffer) = Renderer::create_buffer(instance, device, physical_device, buffer_size, vk::BufferUsageFlags::TRANSFER_SRC, vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT);

        unsafe {
            let data_ptr: *mut u8 = device.map_memory(staging_buffer_memory, 0, buffer_size, vk::MemoryMapFlags::empty()).unwrap() as *mut u8;
            data_ptr.copy_from_nonoverlapping(pixels.as_ptr(), pixels.len());
            device.unmap_memory(staging_buffer_memory);
        };

        let (texture_image, image_memory) = Renderer::create_image(instance, device, physical_device, image.width(), image.height(), vk::Format::R8G8B8A8_SRGB, vk::ImageTiling::OPTIMAL, vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED, vk::MemoryPropertyFlags::DEVICE_LOCAL);
    
        Renderer::transition_image_layout(device, single_time_command_pool, graphics_queue, texture_image, vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL);
        Renderer::copy_buffer_to_image(device, single_time_command_pool, staging_buffer, graphics_queue, texture_image, image.width(), image.height());
        Renderer::transition_image_layout(device, single_time_command_pool, graphics_queue, texture_image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        
        unsafe {
            device.destroy_buffer(staging_buffer, None);
            device.free_memory(staging_buffer_memory, None);
        };


        (texture_image, image_memory)
    }

    fn create_image(instance: &ash::Instance, device: &ash::Device, physical_device: vk::PhysicalDevice, width: u32, height: u32, format: vk::Format, tiling: vk::ImageTiling, usage: vk::ImageUsageFlags, properties: vk::MemoryPropertyFlags) -> (vk::Image, vk::DeviceMemory) {
        let create_info = vk::ImageCreateInfo {
            s_type: vk::StructureType::IMAGE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::ImageCreateFlags::empty(),
            image_type: vk::ImageType::TYPE_2D,
            format: format,
            extent: vk::Extent3D {
                width: width,
                height: height,
                depth: 1
            },
            mip_levels: 1,
            array_layers: 1,
            samples: vk::SampleCountFlags::TYPE_1,
            tiling: tiling,
            usage: usage,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            queue_family_index_count: 0,
            p_queue_family_indices: ptr::null(),
            initial_layout: vk::ImageLayout::UNDEFINED,
            ..Default::default()
        };
        
        let image: vk::Image = unsafe {
            device.create_image(&create_info, None).unwrap()
        };

        let mem_requirements: vk::MemoryRequirements = unsafe {
            device.get_image_memory_requirements(image)
        };

        let allocate_info = vk::MemoryAllocateInfo {
            s_type: vk::StructureType::MEMORY_ALLOCATE_INFO,
            p_next: ptr::null(),
            allocation_size: mem_requirements.size,
            memory_type_index: Renderer::find_memory_type(instance, physical_device, mem_requirements.memory_type_bits, properties),
            ..Default::default()
        }; 

        let image_memory: vk::DeviceMemory = unsafe {
            device.allocate_memory(&allocate_info, None).unwrap()
        };

        unsafe {
            device.bind_image_memory(image, image_memory, 0).unwrap()
        };

        (image, image_memory)
    }
    
    fn transition_image_layout(device: &ash::Device, single_time_command_pool: vk::CommandPool, graphics_queue: vk::Queue, image: vk::Image, old_layout: vk::ImageLayout, new_layout: vk::ImageLayout) {
        let command_buffer = Renderer::begin_single_time_commands(device, single_time_command_pool);

        let mut barrier = vk::ImageMemoryBarrier {
            s_type: vk::StructureType::IMAGE_MEMORY_BARRIER,
            p_next: ptr::null(),
            src_access_mask: vk::AccessFlags::empty(),
            dst_access_mask: vk::AccessFlags::empty(),
            old_layout: old_layout,
            new_layout: new_layout,
            src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            image: image,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_array_layer: 0,
                base_mip_level: 0,
                level_count: 1,
                layer_count: 1
            },
            ..Default::default()
        };

        let source_stage: vk::PipelineStageFlags;
        let destination_stage: vk::PipelineStageFlags;

        if old_layout == vk::ImageLayout::UNDEFINED && new_layout == vk::ImageLayout::TRANSFER_DST_OPTIMAL {
            barrier.src_access_mask = vk::AccessFlags::empty();
            barrier.dst_access_mask = vk::AccessFlags::TRANSFER_WRITE;

            source_stage = vk::PipelineStageFlags::TOP_OF_PIPE;
            destination_stage = vk::PipelineStageFlags::TRANSFER;
        }
        else if old_layout == vk::ImageLayout::TRANSFER_DST_OPTIMAL && new_layout == vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL {
            barrier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
            barrier.dst_access_mask = vk::AccessFlags::SHADER_READ;

            source_stage = vk::PipelineStageFlags::TRANSFER;
            destination_stage = vk::PipelineStageFlags::FRAGMENT_SHADER;
        }
        else {
            panic!("unsupported layout transition!");
        }

        unsafe {
            device.cmd_pipeline_barrier(command_buffer, source_stage, destination_stage, vk::DependencyFlags::empty(), &[], &[], std::slice::from_ref(&barrier));
        }

        Renderer::end_single_time_commands(device, single_time_command_pool, command_buffer, graphics_queue);
    }

    fn begin_single_time_commands(device: &ash::Device, single_time_command_pool: vk::CommandPool) -> vk::CommandBuffer {
        let allocate_info = vk::CommandBufferAllocateInfo {
            s_type: vk::StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
            p_next: ptr::null(),
            command_pool: single_time_command_pool,
            level: vk::CommandBufferLevel::PRIMARY,
            command_buffer_count: 1,
            ..Default::default()
        };
        
        let command_buffer: vk::CommandBuffer = unsafe {
            device.allocate_command_buffers(&allocate_info).unwrap()[0]
        };

        let begin_info = vk::CommandBufferBeginInfo {
            s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
            p_next: ptr::null(),
            flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
            p_inheritance_info: ptr::null(),
            ..Default::default()
        };

        unsafe {
            device.begin_command_buffer(command_buffer, &begin_info).unwrap();
        };

        command_buffer
    }

    fn end_single_time_commands(device: &ash::Device, single_time_command_pool: vk::CommandPool, command_buffer: vk::CommandBuffer, graphics_queue: vk::Queue) {
        unsafe {
            device.end_command_buffer(command_buffer).unwrap();

            let submit_info: vk::SubmitInfo<'_> = vk::SubmitInfo {
                s_type: vk::StructureType::SUBMIT_INFO,
                p_next: ptr::null(),
                wait_semaphore_count: 0,
                p_wait_semaphores: ptr::null(),
                p_wait_dst_stage_mask: ptr::null(),
                command_buffer_count: 1,
                p_command_buffers: &command_buffer,
                signal_semaphore_count: 0,
                p_signal_semaphores: ptr::null(),
                ..Default::default()
            };

            device.queue_submit(graphics_queue, std::slice::from_ref(&submit_info), vk::Fence::null()).unwrap();
            device.queue_wait_idle(graphics_queue).unwrap();

            device.free_command_buffers(single_time_command_pool, std::slice::from_ref(&command_buffer));
        }
    }

    fn copy_buffer_to_image(device: &ash::Device, single_time_command_pool: vk::CommandPool, buffer: vk::Buffer, graphics_queue: vk::Queue, image: vk::Image, width: u32, height: u32) {
        let command_buffer: vk::CommandBuffer = Renderer::begin_single_time_commands(device, single_time_command_pool);
        let region = vk::BufferImageCopy {
            buffer_offset: 0,
            buffer_row_length: 0,
            buffer_image_height: 0,
            image_subresource: vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1
            },
            image_offset: vk::Offset3D {x: 0, y: 0, z: 0},
            image_extent: vk::Extent3D {width, height, depth: 1},

        };
        
        unsafe {
            device.cmd_copy_buffer_to_image(command_buffer, buffer, image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, std::slice::from_ref(&region));
        };

        Renderer::end_single_time_commands(device, single_time_command_pool, command_buffer, graphics_queue);
    }

    fn create_depth_resources(instance: &ash::Instance, device: &ash::Device, physical_device: vk::PhysicalDevice, swapchain_extent: vk::Extent2D, depth_format: vk::Format) -> (vk::Image, vk::DeviceMemory, vk::ImageView) {
        let (depth_image, depth_image_memory) = Renderer::create_image(instance, device, physical_device, swapchain_extent.width, swapchain_extent.height, depth_format, vk::ImageTiling::OPTIMAL, vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT, vk::MemoryPropertyFlags::DEVICE_LOCAL);
        let depth_image_view = Renderer::create_image_view(device, depth_image, depth_format, vk::ImageAspectFlags::DEPTH);
        
        (depth_image, depth_image_memory, depth_image_view)
    }

    fn find_supported_format(instance: &ash::Instance, physical_device: vk::PhysicalDevice, candidates: &Vec<vk::Format>, tiling: vk::ImageTiling, features: vk::FormatFeatureFlags) -> vk::Format {
        for format in candidates {
            let properties: vk::FormatProperties = unsafe {
                instance.get_physical_device_format_properties(physical_device, *format)
            };

            if tiling == vk::ImageTiling::LINEAR && properties.linear_tiling_features.contains(features) ||
                tiling == vk::ImageTiling::OPTIMAL && properties.optimal_tiling_features.contains(features) {
                return *format;
            }
        }

        vk::Format::UNDEFINED
    }

    fn find_depth_format(instance: &ash::Instance, physical_device: vk::PhysicalDevice) -> vk::Format {
        Renderer::find_supported_format(
            instance,
            physical_device,
            &vec![vk::Format::D32_SFLOAT, vk::Format::D32_SFLOAT_S8_UINT, vk::Format::D24_UNORM_S8_UINT],
            vk::ImageTiling::OPTIMAL,
            vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT
        )
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        if USE_VALIDATION_LAYERS {
            unsafe {
                self.debug_utils_loader.destroy_debug_utils_messenger(self.debug_messager, None);
            }
        }
        
        unsafe {
            self.device.device_wait_idle().unwrap();
            self.cleanup_swapchain();
            
            for i in 0..MAX_FRAMES_IN_FLIGHT as usize {
                self.device.destroy_semaphore(self.swap_image_available_semaphores[i], None);
                self.device.destroy_semaphore(self.render_finished_semaphores[i], None);
                self.device.destroy_fence(self.in_flight_fences[i], None);

                self.device.destroy_buffer(self.uniform_buffers[i], None);
                self.device.free_memory(self.uniform_buffers_memory[i], None);
            }
            
            self.device.destroy_image_view(self.depth_image_view, None);
            self.device.destroy_image(self.depth_image, None);
            self.device.free_memory(self.depth_image_memory, None);

            self.device.destroy_sampler(self.texture_sampler, None);
            self.device.destroy_image_view(self.texture_image_view, None);
            self.device.destroy_image(self.texture_image, None);
            self.device.free_memory(self.texture_image_memory, None);

            self.device.destroy_descriptor_pool(self.descriptor_pool, None);
            self.device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            
            self.device.destroy_command_pool(self.single_time_command_pool, None);
            self.device.destroy_buffer(self.vertex_buffer, None);
            self.device.free_memory(self.vertex_buffer_memory, None);
            self.device.destroy_buffer(self.index_buffer, None);
            self.device.free_memory(self.index_buffer_memory, None);

            self.device.destroy_command_pool(self.command_pool, None);
            
            self.device.destroy_pipeline_layout(self.pipeline_layout, None);
            self.device.destroy_pipeline(self.pipeline, None);
            self.device.destroy_render_pass(self.render_pass, None);

            self.device.destroy_device(None);
            self.surface_loader.destroy_surface(self.surface, None);
            
            self.instance.destroy_instance(None);
        }
    }
}

fn populate_debug_messenger_create_info(create_info: &mut vk::DebugUtilsMessengerCreateInfoEXT<'_>) {
    create_info.s_type = vk::StructureType::DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    create_info.p_next = ptr::null();

    create_info.flags = vk::DebugUtilsMessengerCreateFlagsEXT::empty();
    create_info.message_severity = 
            vk::DebugUtilsMessageSeverityFlagsEXT::WARNING |
            // vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE |
            // vk::DebugUtilsMessageSeverityFlagsEXT::INFO |
            vk::DebugUtilsMessageSeverityFlagsEXT::ERROR;

    create_info.message_type =
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL |
            vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE |
            vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION;

    create_info.pfn_user_callback = Some(vulkan_debug_utils_callback);
    create_info.p_user_data = ptr::null_mut();
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
                self.renderer.as_mut().unwrap().draw_frame(self.window.as_ref().unwrap(), self.window_resized);
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