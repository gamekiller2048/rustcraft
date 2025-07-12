use std::collections::HashSet;
use std::ffi::{c_char, c_void, CStr, CString};
use std::fs::{self};
use std::ptr::{self};
use ash::vk::RenderPassCreateFlags;
use winit::raw_window_handle::HasWindowHandle;
use winit::{self};
use ash::{ext, khr, vk};
use log::{warn};

const REQUIRED_VALIDATION_LAYER_NAMES: [&'static str; 1] = ["VK_LAYER_KHRONOS_validation"];
const USE_VALIDATION_LAYERS: bool = true;

const REQUIRED_DEVICE_EXTENSIONS: [*const c_char; 1] = [
    ash::khr::swapchain::NAME.as_ptr()
];

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
    let message = unsafe {
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

struct App {
    window: Option<winit::window::Window>,
    entry: ash::Entry,
    instance: ash::Instance,

    debug_utils_loader: ext::debug_utils::Instance,
    debug_messager: vk::DebugUtilsMessengerEXT,

    physical_device: vk::PhysicalDevice,
    device: Option<ash::Device>,

    surface_loader: khr::surface::Instance,
    surface: vk::SurfaceKHR,
    
    swapchain_loader: Option<khr::swapchain::Device>,
    swapchain: vk::SwapchainKHR,
    swapchain_images: Vec<vk::Image>,
    swapchain_image_format: vk::Format,
    swapchain_extent: vk::Extent2D,
    swapchain_image_views: Vec<vk::ImageView>,

    render_pass: vk::RenderPass,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout
} 

impl winit::application::ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let window = event_loop.create_window(winit::window::Window::default_attributes()).unwrap();
        
        self.surface = App::create_surface(&self.entry, &self.instance, &window);
        self.physical_device = App::pick_physical_device(&self.instance, &self.surface_loader, self.surface);
        self.device = Some(App::create_logical_device(&self.instance, &self.surface_loader, self.surface, self.physical_device));
        self.swapchain_loader = Some(khr::swapchain::Device::new(&self.instance, self.device.as_ref().unwrap()));
        (self.swapchain, self.swapchain_images, self.swapchain_image_format, self.swapchain_extent) = App::create_swapchain(&self.instance, self.swapchain_loader.as_ref().unwrap(), 100, 100, &self.surface_loader, self.surface, self.physical_device);
        self.swapchain_image_views = App::create_image_views(self.device.as_ref().unwrap(), &self.swapchain_images, self.swapchain_image_format);
        self.render_pass = App::create_render_pass(self.device.as_ref().unwrap(), self.swapchain_image_format);
        (self.pipeline, self.pipeline_layout) = App::create_graphics_pipeline(self.device.as_ref().unwrap(), self.swapchain_extent, self.render_pass);

        self.window = Some(window);
    }

    fn window_event(&mut self, event_loop: &winit::event_loop::ActiveEventLoop, _id: winit::window::WindowId, event: winit::event::WindowEvent) {
        match event {
            winit::event::WindowEvent::CloseRequested => {
                event_loop.exit();
            },
            winit::event::WindowEvent::RedrawRequested => {
                self.draw_frame();
                self.window.as_ref().unwrap().request_redraw();
            }
            _ => (),
        }
    }
}

impl App {
    fn new() -> Self {    
        let entry = unsafe {
            ash::Entry::load().unwrap()
        };
        
        if USE_VALIDATION_LAYERS && !App::check_validation_layer_support(&entry) {
            panic!("requested validation layers but not supported");
        }
        
        let instance = App::create_instance(&entry);
        let debug_utils_loader = ext::debug_utils::Instance::new(&entry, &instance);
        let debug_messager = App::setup_debug_messenger(&debug_utils_loader);
        let surface_loader = khr::surface::Instance::new(&entry, &instance);

        App {
            window: None,
            entry, instance,
            debug_utils_loader,
            debug_messager,
            physical_device: vk::PhysicalDevice::null(),
            device: None,
            surface_loader,
            surface: vk::SurfaceKHR::null(),
            swapchain_loader: None,
            swapchain: vk::SwapchainKHR::null(),
            swapchain_images: vec![],
            swapchain_image_format: vk::Format::UNDEFINED,
            swapchain_extent: vk::Extent2D {width: 0, height: 0},
            swapchain_image_views: vec![],
            render_pass: vk::RenderPass::null(),
            pipeline: vk::Pipeline::null(),
            pipeline_layout: vk::PipelineLayout::null(),
        }
    }

    fn draw_frame(&mut self) {
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
                let layer_name = unsafe {
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

        let app_info: vk::ApplicationInfo<'_> = vk::ApplicationInfo {
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

    fn pick_physical_device(instance: &ash::Instance, surface_loader: &khr::surface::Instance, surface: vk::SurfaceKHR) -> vk::PhysicalDevice {
        let devices: Vec<vk::PhysicalDevice> = unsafe {
            instance.enumerate_physical_devices().unwrap()
        };

        if devices.len() <= 0 {
            panic!("no devices found");
        }

        for device in devices {
            if App::is_device_suitable(&instance, surface_loader, surface, device) {
                return device;
            }
        }

        panic!("failed to find suitable device");
    }

    // must have: graphics and present queue + at least one format and present mode in swap chain
    fn is_device_suitable(instance: &ash::Instance, surface_loader: &khr::surface::Instance, surface: vk::SurfaceKHR, device: vk::PhysicalDevice) -> bool {
        let _properties: vk::PhysicalDeviceProperties = unsafe {
            instance.get_physical_device_properties(device)
        };

        let _features: vk::PhysicalDeviceFeatures = unsafe {
            instance.get_physical_device_features(device)
        };

        let indices: QueueFamilyIndices = App::find_queue_families(&instance, surface_loader, surface, device);

        let is_extensions_supported: bool = App::check_device_extension_support(instance, device);
        let mut is_swapchain_adequate: bool = false;

        if is_extensions_supported {
            let swapchain: SwapChainSupportDetails = App::query_swapchain_support(surface_loader, surface, device);
            is_swapchain_adequate = !swapchain.formats.is_empty() && !swapchain.present_modes.is_empty();
        }

        indices.graphics.is_some() && indices.present.is_some() && is_extensions_supported && is_swapchain_adequate
    }

    fn find_queue_families(instance: &ash::Instance, surface_loader: &khr::surface::Instance, surface: vk::SurfaceKHR, device: vk::PhysicalDevice) -> QueueFamilyIndices {
        let queue_families = unsafe {
            instance.get_physical_device_queue_family_properties(device)
        };

        let mut indices: QueueFamilyIndices = Default::default();

        let mut i: u32 = 0;
        for family in queue_families {
            if (family.queue_flags & vk::QueueFlags::GRAPHICS).is_empty() {
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

    fn create_logical_device(instance: &ash::Instance, surface_loader: &khr::surface::Instance, surface: vk::SurfaceKHR, physical_device: vk::PhysicalDevice) -> ash::Device {
        let indices: QueueFamilyIndices = App::find_queue_families(instance, surface_loader, surface, physical_device);
        let queue_priority: f32 = 1.0;

        
        let mut queue_create_infos: Vec<vk::DeviceQueueCreateInfo> = vec![];
        let queue_family_indices: [u32; 2] = [indices.graphics.unwrap(), indices.present.unwrap()];

        for queue_family_index in queue_family_indices {
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

        let physical_device_features = vk::PhysicalDeviceFeatures {..Default::default()};

        let device_create_info = vk::DeviceCreateInfo {
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
            instance.create_device(physical_device, &device_create_info, None).unwrap()
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
        let supported_extensions = unsafe {
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

    fn create_swapchain(instance: &ash::Instance, swapchain_loader: &khr::swapchain::Device, width: u32, height: u32, surface_loader: &khr::surface::Instance, surface: vk::SurfaceKHR, device: vk::PhysicalDevice) -> (vk::SwapchainKHR, Vec<vk::Image>, vk::Format, vk::Extent2D) {
        let swapchain: SwapChainSupportDetails = App::query_swapchain_support(surface_loader, surface, device);

        let surface_format: vk::SurfaceFormatKHR = App::choose_swap_surface_format(&swapchain.formats);
        let present_mode: vk::PresentModeKHR = App::choose_swap_present_mode(&swapchain.present_modes);
        let extent: vk::Extent2D = App::choose_swap_extent(width, height, &swapchain.capabilities);
         
        let mut image_count: u32 = swapchain.capabilities.min_image_count + 1;
        if image_count > swapchain.capabilities.max_image_count {
            image_count = swapchain.capabilities.min_image_count;
        }

        let mut swapchain_create_info = vk::SwapchainCreateInfoKHR {
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

        let indices: QueueFamilyIndices = App::find_queue_families(instance, surface_loader, surface, device);
        let queue_family_indices: [u32; 2] = [indices.graphics.unwrap(), indices.present.unwrap()];

        if indices.graphics != indices.present {
            swapchain_create_info.image_sharing_mode = vk::SharingMode::CONCURRENT;
            swapchain_create_info.queue_family_index_count = queue_family_indices.len() as u32;
            swapchain_create_info.p_queue_family_indices = queue_family_indices.as_ptr();
        } 

        let swapchain = unsafe {
            swapchain_loader.create_swapchain(&swapchain_create_info, None).unwrap()
        };

        let swap_chain_images: Vec<vk::Image> = unsafe {
            swapchain_loader.get_swapchain_images(swapchain).unwrap()
        };

        let swapchain_image_format = surface_format.format;
        let swapchain_extent = extent;
        
        (swapchain, swap_chain_images, swapchain_image_format, swapchain_extent)
    }

    fn create_image_views(device: &ash::Device, swapchain_images: &Vec<vk::Image>, swapchain_image_format: vk::Format) -> Vec<vk::ImageView> {
        let mut swapchain_image_views: Vec<vk::ImageView> = vec![];
        swapchain_image_views.reserve(swapchain_images.len());

        for i in 0..swapchain_images.len() {
            let image_view_create_info = vk::ImageViewCreateInfo {
                s_type: vk::StructureType::IMAGE_VIEW_CREATE_INFO,
                p_next: ptr::null(),
                flags: vk::ImageViewCreateFlags::empty(),
                image: swapchain_images[i],
                view_type: vk::ImageViewType::TYPE_2D,
                format: swapchain_image_format,
                components: vk::ComponentMapping {
                    r: vk::ComponentSwizzle::R,
                    g: vk::ComponentSwizzle::G,
                    b: vk::ComponentSwizzle::B,
                    a: vk::ComponentSwizzle::A
                },
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1
                },
                ..Default::default()
            };
            
            swapchain_image_views.push(unsafe {
                device.create_image_view(&image_view_create_info, None).unwrap()
            });
        }

        swapchain_image_views
    }
    
    fn create_render_pass(device: &ash::Device, swapchain_image_format: vk::Format) -> vk::RenderPass {
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

        let color_attachment_ref = vk::AttachmentReference {
            attachment: 0,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
        };

        let subpass = vk::SubpassDescription {
            flags: vk::SubpassDescriptionFlags::empty(),
            pipeline_bind_point: vk::PipelineBindPoint::GRAPHICS,
            p_color_attachments: &color_attachment_ref,
            color_attachment_count: 1,
            
            // only using color attachments for now
            p_input_attachments: ptr::null(),
            p_depth_stencil_attachment: ptr::null(),
            p_resolve_attachments: ptr::null(),
            p_preserve_attachments: ptr::null(),
            input_attachment_count: 0,
            preserve_attachment_count: 0,
            ..Default::default()            
        };

        let render_pass_create_info = vk::RenderPassCreateInfo {
            s_type: vk::StructureType::RENDER_PASS_CREATE_INFO,
            p_next: ptr::null(),
            flags: RenderPassCreateFlags::empty(),
            attachment_count: 1,
            p_attachments: &color_attachment,
            subpass_count: 1,
            p_subpasses: &subpass,
            p_dependencies: ptr::null(),
            dependency_count: 0,
            ..Default::default()
        };

        unsafe {
            device.create_render_pass(&render_pass_create_info, None).unwrap()
        }
    }

    fn create_graphics_pipeline(device: &ash::Device, swapchain_extent: vk::Extent2D, render_pass: vk::RenderPass) -> (vk::Pipeline, vk::PipelineLayout) {
        let vert_bytes: Vec<u8> = fs::read("src/shader.vert.spv").unwrap();
        let frag_bytes: Vec<u8> = fs::read("src/shader.frag.spv").unwrap();

        let vert_shader: vk::ShaderModule = App::create_shader_module(device, &vert_bytes);
        let frag_shader: vk::ShaderModule = App::create_shader_module(device, &frag_bytes);

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

        let vertex_input_state_create_info = vk::PipelineVertexInputStateCreateInfo {
            s_type: vk::StructureType::PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::PipelineVertexInputStateCreateFlags::empty(),
            vertex_attribute_description_count: 0,
            p_vertex_attribute_descriptions: ptr::null(),
            vertex_binding_description_count: 0,
            p_vertex_binding_descriptions: ptr::null(),
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
            cull_mode: vk::CullModeFlags::BACK,
            front_face: vk::FrontFace::CLOCKWISE,
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
            depth_test_enable: vk::FALSE,
            depth_write_enable: vk::FALSE,
            depth_compare_op: vk::CompareOp::LESS_OR_EQUAL,
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

        // no uniforms for now
        let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo {
            s_type: vk::StructureType::PIPELINE_LAYOUT_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::PipelineLayoutCreateFlags::empty(),
            set_layout_count: 0,
            p_set_layouts: ptr::null(),
            p_push_constant_ranges: ptr::null(),
            push_constant_range_count: 0,
            ..Default::default()
        };

        let pipeline_layout = unsafe {
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
        let graphics_pipelines = unsafe {
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
        let shader_module_create_info = vk::ShaderModuleCreateInfo {
            s_type: vk::StructureType::SHADER_MODULE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::ShaderModuleCreateFlags::empty(),
            code_size: bytes.len(),
            p_code: bytes.as_ptr() as *const u32,
            ..Default::default()
        };

        unsafe {
            device.create_shader_module(&shader_module_create_info, None).unwrap()
        }
    }
}

impl Drop for App {
    fn drop(&mut self) {
        if USE_VALIDATION_LAYERS {
            unsafe {
                self.debug_utils_loader.destroy_debug_utils_messenger(self.debug_messager, None);
            }
        }
        
        unsafe {
            self.device.as_ref().unwrap().destroy_pipeline_layout(self.pipeline_layout, None);
            self.device.as_ref().unwrap().destroy_pipeline(self.pipeline, None);
            self.device.as_ref().unwrap().destroy_render_pass(self.render_pass, None);
            
            self.swapchain_loader.as_ref().unwrap().destroy_swapchain(self.swapchain, None);
            for image_view in self.swapchain_image_views.iter() {
                self.device.as_ref().unwrap().destroy_image_view(*image_view, None);
            }

            self.device.as_ref().unwrap().destroy_device(None);
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

fn main() {
    let event_loop = winit::event_loop::EventLoop::new().unwrap();
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);
    
    let mut app: App = App::new();
    event_loop.run_app(&mut app).unwrap();
}