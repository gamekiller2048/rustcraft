use std::collections::HashSet;
use std::ffi::{c_char, CStr};
use std::{num, ptr};
use std::u64;
use ash::{khr, vk};

#[derive(Default)]
pub struct QueueFamilyIndices {
    pub graphics: Option<u32>,
    pub present: Option<u32>
}

#[derive(Default)]
pub struct SwapChainSupportDetails {
    pub capabilities: vk::SurfaceCapabilitiesKHR,
    pub formats: Vec<vk::SurfaceFormatKHR>,
    pub present_modes: Vec<vk::PresentModeKHR>
}

pub struct VulkanContext {
    pub instance: ash::Instance,

    pub physical_device: vk::PhysicalDevice,
    pub queue_family_indices: QueueFamilyIndices,
    pub swapchain_details: SwapChainSupportDetails,
    
    pub device: ash::Device,

    pub graphics_queue: vk::Queue,
    pub present_queue: vk::Queue,

    pub swapchain_loader: khr::swapchain::Device,

    pub single_time_command_pool: vk::CommandPool
}

const USE_VALIDATION_LAYERS: bool = true;
const REQUIRED_DEVICE_EXTENSIONS: [*const c_char; 1] = [
    ash::khr::swapchain::NAME.as_ptr()
];


impl VulkanContext {
    pub fn new(entry: &ash::Entry, instance: ash::Instance, surface_loader: &khr::surface::Instance, surface: vk::SurfaceKHR) -> Self {
        let (physical_device, queue_family_indices, swapchain_details) = Self::pick_physical_device(&instance, &surface_loader, surface);
        let device = Self::create_logical_device(&instance, physical_device, &queue_family_indices); 
        let (graphics_queue, present_queue) = Self::get_queue_handles(&device, &queue_family_indices);
        let swapchain_loader = khr::swapchain::Device::new(&instance, &device);

        Self {
            instance,
            physical_device,
            queue_family_indices,
            swapchain_details,
            device,
            graphics_queue,
            present_queue,
            swapchain_loader,
            single_time_command_pool: vk::CommandPool::null()
        }
    }

    pub fn init(&mut self) {
        self.single_time_command_pool = self.create_command_pool(vk::CommandPoolCreateFlags::TRANSIENT)
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

    fn is_device_suitable(instance: &ash::Instance, surface_loader: &khr::surface::Instance, surface: vk::SurfaceKHR, device: vk::PhysicalDevice, queue_family_indices: &QueueFamilyIndices, swapchain_details: &SwapChainSupportDetails) -> bool {
        let _properties: vk::PhysicalDeviceProperties = unsafe {
            instance.get_physical_device_properties(device)
        };

        let features: vk::PhysicalDeviceFeatures = unsafe {
            instance.get_physical_device_features(device)
        };

        let is_extensions_supported: bool = Self::check_device_extension_support(instance, device);
        let mut is_swapchain_adequate: bool = false;

        if is_extensions_supported {
            is_swapchain_adequate = !swapchain_details.formats.is_empty() && !swapchain_details.present_modes.is_empty();
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

    fn pick_physical_device(instance: &ash::Instance, surface_loader: &khr::surface::Instance, surface: vk::SurfaceKHR) -> (vk::PhysicalDevice, QueueFamilyIndices, SwapChainSupportDetails) {
        let devices: Vec<vk::PhysicalDevice> = unsafe {
            instance.enumerate_physical_devices().unwrap()
        };

        if devices.len() <= 0 {
            panic!("no devices found");
        }

        for device in devices {
            let queue_family_indices: QueueFamilyIndices = Self::find_queue_families(instance, surface_loader, surface, device);
            let swapchain_details = Self::query_swapchain_support(surface_loader, surface, device);

            if Self::is_device_suitable(&instance, surface_loader, surface, device, &queue_family_indices, &swapchain_details) {
                return (device, queue_family_indices, swapchain_details);
            }
        }

        panic!("failed to find suitable device");
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

    pub fn create_swapchain(&self, surface: vk::SurfaceKHR, format: vk::SurfaceFormatKHR, present_mode: vk::PresentModeKHR, extent: vk::Extent2D) -> (vk::SwapchainKHR, Vec<vk::Image>) {
        let mut image_count: u32 = self.swapchain_details.capabilities.min_image_count + 1;
        if image_count > self.swapchain_details.capabilities.max_image_count {
            image_count = self.swapchain_details.capabilities.min_image_count;
        }
        
        let mut create_info = vk::SwapchainCreateInfoKHR {
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
            image_sharing_mode: vk::SharingMode::EXCLUSIVE,
            queue_family_index_count: 0,
            p_queue_family_indices: ptr::null(),
            pre_transform: self.swapchain_details.capabilities.current_transform,
            composite_alpha: vk::CompositeAlphaFlagsKHR::OPAQUE,
            present_mode: present_mode,
            clipped: vk::TRUE,
            old_swapchain: vk::SwapchainKHR::null(), // for now assume we will only create one swapchain
            ..Default::default()
        };

        let indices: [u32; 2] = [self.queue_family_indices.graphics.unwrap(), self.queue_family_indices.present.unwrap()];

        if self.queue_family_indices.graphics != self.queue_family_indices.present {
            create_info.image_sharing_mode = vk::SharingMode::CONCURRENT;
            create_info.queue_family_index_count = indices.len() as u32;
            create_info.p_queue_family_indices = indices.as_ptr();
        } 

        let swapchain: vk::SwapchainKHR = unsafe {
            self.swapchain_loader.create_swapchain(&create_info, None).unwrap()
        };

        let swap_chain_images: Vec<vk::Image> = unsafe {
            self.swapchain_loader.get_swapchain_images(swapchain).unwrap()
        };
        
        (swapchain, swap_chain_images)
    }

    fn find_memory_type(&self, suitable_memory_bits: u32, required_properties: vk::MemoryPropertyFlags) -> u32 {

        let memory_properties: vk::PhysicalDeviceMemoryProperties = unsafe {
            self.instance.get_physical_device_memory_properties(self.physical_device)
        };

        for i in 0..memory_properties.memory_type_count as usize {
            if suitable_memory_bits & (1 << i) != 0 && memory_properties.memory_types[i].property_flags.contains(required_properties) {
                return i as u32;
            }
        }

        panic!("failed to find suitable memory")
    }

    pub fn create_command_pool(&self, flags: vk::CommandPoolCreateFlags) -> vk::CommandPool {
        let create_info = vk::CommandPoolCreateInfo {
            s_type: vk::StructureType::COMMAND_POOL_CREATE_INFO,
            p_next: ptr::null(),
            flags: flags,
            queue_family_index: self.queue_family_indices.graphics.unwrap(),
            ..Default::default()
        };

        unsafe {
            self.device.create_command_pool(&create_info, None).unwrap()
        }
    }

    fn begin_single_time_commands(&self) -> vk::CommandBuffer {
        let allocate_info = vk::CommandBufferAllocateInfo {
            s_type: vk::StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
            p_next: ptr::null(),
            command_pool: self.single_time_command_pool,
            level: vk::CommandBufferLevel::PRIMARY,
            command_buffer_count: 1,
            ..Default::default()
        };

        let command_buffer: vk::CommandBuffer = unsafe {
            self.device.allocate_command_buffers(&allocate_info).unwrap()[0]
        };

        let begin_info = vk::CommandBufferBeginInfo {
            s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
            p_next: ptr::null(),
            flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
            p_inheritance_info: ptr::null(),
            ..Default::default()
        };

        unsafe {
            self.device.begin_command_buffer(command_buffer, &begin_info).unwrap();
        };

        command_buffer
    }

    fn end_single_time_commands(&self, command_buffer: vk::CommandBuffer) {
        unsafe {
            self.device.end_command_buffer(command_buffer).unwrap();

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

            self.device.queue_submit(self.graphics_queue, std::slice::from_ref(&submit_info), vk::Fence::null()).unwrap();
            self.device.queue_wait_idle(self.graphics_queue).unwrap();

            self.device.free_command_buffers(self.single_time_command_pool, std::slice::from_ref(&command_buffer));
        }
    }


    pub fn create_buffer(&self, size: vk::DeviceSize, usage: vk::BufferUsageFlags, properties: vk::MemoryPropertyFlags) -> (vk::DeviceMemory, vk::Buffer) {
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
            self.device.create_buffer(&create_info, None).unwrap()
        };

        let mem_requirements: vk::MemoryRequirements = unsafe {
            self.device.get_buffer_memory_requirements(buffer)
        };

        let allocate_info = vk::MemoryAllocateInfo {
            s_type: vk::StructureType::MEMORY_ALLOCATE_INFO,
            p_next: ptr::null(),
            allocation_size: mem_requirements.size,
            memory_type_index: self.find_memory_type(mem_requirements.memory_type_bits, properties),
            ..Default::default()
        }; 

        let buffer_memory: vk::DeviceMemory = unsafe {
            self.device.allocate_memory(&allocate_info, None).unwrap()
        };

        unsafe {    
            self.device.bind_buffer_memory(buffer, buffer_memory, 0).unwrap();
        };

        (buffer_memory, buffer)
    }

    pub fn map_memory<T>(&self, memory: vk::DeviceMemory, offset: u64, size: vk::DeviceSize) -> *mut T {
        unsafe {
            self.device.map_memory(memory, offset, size, vk::MemoryMapFlags::empty()).unwrap() as *mut T
        }   
    }

    pub fn unmap<T>(&self, memory: vk::DeviceMemory) {
        unsafe {
            self.device.unmap_memory(memory);
        };
    }

    pub fn copy_buffer(&self, src_buffer: vk::Buffer, dst_buffer: vk::Buffer, size: vk::DeviceSize) {
        let command_buffer: vk::CommandBuffer = self.begin_single_time_commands();
        let region = vk::BufferCopy {
            src_offset: 0,
            dst_offset: 0,
            size: size
        };
        
        unsafe {
            self.device.cmd_copy_buffer(command_buffer, src_buffer, dst_buffer, std::slice::from_ref(&region));
        };

        self.end_single_time_commands(command_buffer);
    }

    pub fn create_vertex_buffer<T>(&self, vertices: &[T]) -> (vk::DeviceMemory, vk::Buffer) {
        let buffer_size: vk::DeviceSize = (size_of::<T>() * vertices.len()) as u64;
        let (staging_buffer_memory, staging_buffer): (vk::DeviceMemory, vk::Buffer) = self.create_buffer(buffer_size, vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::VERTEX_BUFFER, vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT);

        unsafe {
            let data_ptr: *mut T = self.device.map_memory(staging_buffer_memory, 0, buffer_size, vk::MemoryMapFlags::empty()).unwrap() as *mut T;
            data_ptr.copy_from_nonoverlapping(vertices.as_ptr(), vertices.len());
            self.device.unmap_memory(staging_buffer_memory);
        };

        let (vertex_buffer_memory, vertex_buffer): (vk::DeviceMemory, vk::Buffer) = self.create_buffer(buffer_size, vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER, vk::MemoryPropertyFlags::DEVICE_LOCAL);

        self.copy_buffer(staging_buffer, vertex_buffer, buffer_size);

        unsafe {
            self.device.destroy_buffer(staging_buffer, None);
            self.device.free_memory(staging_buffer_memory, None);
        }

        (vertex_buffer_memory, vertex_buffer)
    }

    pub fn create_index_buffer<T>(&self, indices: &[T]) -> (vk::DeviceMemory, vk::Buffer) {
        let buffer_size: vk::DeviceSize = (size_of::<T>() * indices.len()) as u64;
        let (staging_buffer_memory, staging_buffer): (vk::DeviceMemory, vk::Buffer) = self.create_buffer(buffer_size, vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::INDEX_BUFFER, vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT);

        unsafe {
            let data_ptr: *mut T = self.device.map_memory(staging_buffer_memory, 0, buffer_size, vk::MemoryMapFlags::empty()).unwrap() as *mut T;
            data_ptr.copy_from_nonoverlapping(indices.as_ptr(), indices.len());
            self.device.unmap_memory(staging_buffer_memory);
        };

        let (index_buffer_memory, index_buffer): (vk::DeviceMemory, vk::Buffer) = self.create_buffer(buffer_size, vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER, vk::MemoryPropertyFlags::DEVICE_LOCAL);

        self.copy_buffer(staging_buffer, index_buffer, buffer_size);

        unsafe {
            self.device.destroy_buffer(staging_buffer, None);
            self.device.free_memory(staging_buffer_memory, None);
        }

        (index_buffer_memory, index_buffer)
    }
    
    pub fn create_shader_module(&self, bytes: &Vec<u8>) -> vk::ShaderModule {
        let create_info = vk::ShaderModuleCreateInfo {
            s_type: vk::StructureType::SHADER_MODULE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::ShaderModuleCreateFlags::empty(),
            code_size: bytes.len(),
            p_code: bytes.as_ptr() as *const u32,
            ..Default::default()
        };

        unsafe {
            self.device.create_shader_module(&create_info, None).unwrap()
        }
    }

    pub fn destroy_shader_module(&self, shader: vk::ShaderModule) {
        unsafe {
            self.device.destroy_shader_module(shader, None);
        };
    }

    pub fn create_descriptor_pool(&self, pool_sizes: &[vk::DescriptorPoolSize], max_sets: u32) -> vk::DescriptorPool {
        let create_info = vk::DescriptorPoolCreateInfo {
            s_type: vk::StructureType::DESCRIPTOR_POOL_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::DescriptorPoolCreateFlags::empty(),
            pool_size_count: pool_sizes.len() as u32,
            p_pool_sizes: pool_sizes.as_ptr(),
            max_sets: max_sets,
            ..Default::default()
        };

        unsafe {
            self.device.create_descriptor_pool(&create_info, None).unwrap()
        }
    }

    pub fn create_descriptor_set_layout(&self, bindings: &[vk::DescriptorSetLayoutBinding]) -> vk::DescriptorSetLayout {
        let create_info = vk::DescriptorSetLayoutCreateInfo {
            s_type: vk::StructureType::DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::DescriptorSetLayoutCreateFlags::empty(),
            binding_count: bindings.len() as u32,
            p_bindings: bindings.as_ptr(),
            ..Default::default()
        };
        
        unsafe {
            self.device.create_descriptor_set_layout(&create_info, None).unwrap()
        }
    }

    pub fn create_descriptor_sets(&self, descriptor_pool: vk::DescriptorPool, layouts: &[vk::DescriptorSetLayout]) -> Vec<vk::DescriptorSet> {
        let allocate_info = vk::DescriptorSetAllocateInfo {
            s_type: vk::StructureType::DESCRIPTOR_SET_ALLOCATE_INFO,
            p_next: ptr::null(),
            descriptor_pool: descriptor_pool,
            descriptor_set_count: layouts.len() as u32,
            p_set_layouts: layouts.as_ptr(),
            ..Default::default()
        };

        unsafe {
            self.device.allocate_descriptor_sets(&allocate_info).unwrap()
        }
    }

    pub fn write_descriptors(&self, writes: &[vk::WriteDescriptorSet], copies: &[vk::CopyDescriptorSet]) {
        unsafe {
            self.device.update_descriptor_sets(&writes, copies);
        }
    }

    pub fn create_render_pass(&self, attachments: &[vk::AttachmentDescription], subpasses: &[vk::SubpassDescription], dependencies: &[vk::SubpassDependency]) -> vk::RenderPass {
        let render_pass_create_info = vk::RenderPassCreateInfo {
            s_type: vk::StructureType::RENDER_PASS_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::RenderPassCreateFlags::empty(),
            attachment_count: attachments.len() as u32,
            p_attachments: attachments.as_ptr(),
            subpass_count: subpasses.len() as u32,
            p_subpasses: subpasses.as_ptr(),
            p_dependencies: dependencies.as_ptr(),
            dependency_count: dependencies.len() as u32,
            ..Default::default()
        };

        unsafe {
            self.device.create_render_pass(&render_pass_create_info, None).unwrap()
        }
    }

    pub fn create_image_view(&self, image: vk::Image, format: vk::Format, aspect_mask: vk::ImageAspectFlags) -> vk::ImageView {
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
            self.device.create_image_view(&create_info, None).unwrap()
        }
    }

    pub fn create_image(&self, width: u32, height: u32, format: vk::Format, tiling: vk::ImageTiling, usage: vk::ImageUsageFlags, properties: vk::MemoryPropertyFlags) -> (vk::Image, vk::DeviceMemory) {
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
            self.device.create_image(&create_info, None).unwrap()
        };

        let mem_requirements: vk::MemoryRequirements = unsafe {
            self.device.get_image_memory_requirements(image)
        };

        let allocate_info = vk::MemoryAllocateInfo {
            s_type: vk::StructureType::MEMORY_ALLOCATE_INFO,
            p_next: ptr::null(),
            allocation_size: mem_requirements.size,
            memory_type_index: self.find_memory_type(mem_requirements.memory_type_bits, properties),
            ..Default::default()
        }; 

        let image_memory: vk::DeviceMemory = unsafe {
            self.device.allocate_memory(&allocate_info, None).unwrap()
        };

        unsafe {
            self.device.bind_image_memory(image, image_memory, 0).unwrap()
        };

        (image, image_memory)
    }

    pub fn create_command_buffers(&self, command_pool: vk::CommandPool, level: vk::CommandBufferLevel, num_buffers: u32) -> Vec<vk::CommandBuffer> {
        let allocate_info = vk::CommandBufferAllocateInfo {
            s_type: vk::StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
            p_next: ptr::null(),
            command_pool: command_pool,
            level: level,
            command_buffer_count: num_buffers,
            ..Default::default()
        };

        unsafe {
            self.device.allocate_command_buffers(&allocate_info).unwrap()
        }
    }
    
    pub fn find_supported_format(&self, candidates: &Vec<vk::Format>, tiling: vk::ImageTiling, features: vk::FormatFeatureFlags) -> vk::Format {
        for format in candidates {
            let properties: vk::FormatProperties = unsafe {
                self.instance.get_physical_device_format_properties(self.physical_device, *format)
            };

            if tiling == vk::ImageTiling::LINEAR && properties.linear_tiling_features.contains(features) ||
                tiling == vk::ImageTiling::OPTIMAL && properties.optimal_tiling_features.contains(features) {
                return *format;
            }
        }

        vk::Format::UNDEFINED
    }

    pub fn create_pipeline_layout(&self, descriptor_set_layouts: &[vk::DescriptorSetLayout], push_constants: &[vk::PushConstantRange]) -> vk::PipelineLayout {
        let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo {
            s_type: vk::StructureType::PIPELINE_LAYOUT_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::PipelineLayoutCreateFlags::empty(),
            set_layout_count: descriptor_set_layouts.len() as u32,
            p_set_layouts: descriptor_set_layouts.as_ptr(),
            p_push_constant_ranges: push_constants.as_ptr(),
            push_constant_range_count: push_constants.len() as u32,
            ..Default::default()
        };

        unsafe {
            self.device.create_pipeline_layout(&pipeline_layout_create_info, None).unwrap()
        }
    }

    pub fn create_framebuffer(&self, render_pass: vk::RenderPass, attachments: &[vk::ImageView], extent: vk::Extent2D, layers: u32) -> vk::Framebuffer {
        let create_info = vk::FramebufferCreateInfo {
            s_type: vk::StructureType::FRAMEBUFFER_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::FramebufferCreateFlags::empty(),
            render_pass: render_pass,
            attachment_count: attachments.len() as u32,
            p_attachments: attachments.as_ptr(),
            width: extent.width,
            height: extent.height,
            layers: layers,
            ..Default::default()
        };

        unsafe {
            self.device.create_framebuffer(&create_info, None).unwrap()
        }
    }

    pub fn create_semaphores(&self, num_semaphores: u32) -> Vec<vk::Semaphore> {
        let create_info = vk::SemaphoreCreateInfo {
            s_type: vk::StructureType::SEMAPHORE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::SemaphoreCreateFlags::empty(),
            ..Default::default()
        };

        let mut semaphores: Vec<vk::Semaphore> = Vec::with_capacity(num_semaphores as usize);
        
        for _ in 0..num_semaphores {
            semaphores.push(unsafe {
                self.device.create_semaphore(&create_info, None).unwrap()
            });
        }

        semaphores
    }

    pub fn create_fences(&self, num_fences: u32) -> Vec<vk::Fence> {
        let create_info = vk::FenceCreateInfo {
            s_type: vk::StructureType::FENCE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::FenceCreateFlags::SIGNALED,
            ..Default::default()
        };

        let mut fences: Vec<vk::Fence> = Vec::with_capacity(num_fences as usize);
        
        for _ in 0..num_fences {
            fences.push(unsafe {
                self.device.create_fence(&create_info, None).unwrap()
            });
        }

        fences
    }
}

impl Drop for VulkanContext {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_command_pool(self.single_time_command_pool, None);
            self.device.destroy_device(None);
        }
    }
}