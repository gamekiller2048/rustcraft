use ash::{khr, vk};
use std::collections::HashSet;
use std::ffi::{CStr, c_char};
use std::hash::Hash;
use std::ptr;
use std::u64;

#[derive(Default)]
pub struct QueueFamilyIndices {
    pub graphics: HashSet<u32>,
    pub present: HashSet<u32>,
    pub transfer: HashSet<u32>,
    pub transfer_only: HashSet<u32>,
    pub compute: HashSet<u32>,
}

#[derive(Default)]
pub struct SwapChainSupportDetails {
    pub capabilities: vk::SurfaceCapabilitiesKHR,
    pub formats: Vec<vk::SurfaceFormatKHR>,
    pub present_modes: Vec<vk::PresentModeKHR>,
}

pub struct VulkanContext {
    pub instance: ash::Instance,

    pub physical_device: vk::PhysicalDevice,
    pub queue_family_indices: QueueFamilyIndices,
    pub swapchain_details: SwapChainSupportDetails,

    pub device: ash::Device,
    pub swapchain_loader: khr::swapchain::Device,
}

pub type ScorePhysicalDeviceFn = fn(
    properties: vk::PhysicalDeviceProperties,
    features: vk::PhysicalDeviceFeatures,
    queue_family_indices: &QueueFamilyIndices,
    swapchain_details: &SwapChainSupportDetails,
) -> u32;

impl VulkanContext {
    pub fn new(
        entry: &ash::Entry,
        instance: ash::Instance,
        surface_loader: &khr::surface::Instance,
        surface: vk::SurfaceKHR,
        required_device_extensions: &[*const c_char],
        score_physical_device_fn: ScorePhysicalDeviceFn,
    ) -> Self {
        let (physical_device, queue_family_indices, swapchain_details) = Self::pick_physical_device(
            &instance,
            &surface_loader,
            surface,
            required_device_extensions,
            score_physical_device_fn,
        );
        let device = Self::create_logical_device(
            &instance,
            physical_device,
            &queue_family_indices,
            required_device_extensions,
        );
        let swapchain_loader = khr::swapchain::Device::new(&instance, &device);

        Self {
            instance,
            physical_device,
            queue_family_indices,
            swapchain_details,
            device,
            swapchain_loader,
        }
    }

    fn query_swapchain_support(
        surface_loader: &khr::surface::Instance,
        surface: vk::SurfaceKHR,
        device: vk::PhysicalDevice,
    ) -> SwapChainSupportDetails {
        SwapChainSupportDetails {
            capabilities: unsafe {
                surface_loader
                    .get_physical_device_surface_capabilities(device, surface)
                    .unwrap()
            },
            formats: unsafe {
                surface_loader
                    .get_physical_device_surface_formats(device, surface)
                    .unwrap()
            },
            present_modes: unsafe {
                surface_loader
                    .get_physical_device_surface_present_modes(device, surface)
                    .unwrap()
            },
        }
    }

    fn check_device_extension_support(
        instance: &ash::Instance,
        device: vk::PhysicalDevice,
        required_device_extensions: &[*const c_char],
    ) -> bool {
        let supported_extensions: Vec<vk::ExtensionProperties> = unsafe {
            instance
                .enumerate_device_extension_properties(device)
                .unwrap()
        };

        let mut required_extensions: HashSet<&CStr> = HashSet::new();
        for extension in required_device_extensions {
            required_extensions.insert(unsafe { CStr::from_ptr(*extension) });
        }

        for extension in supported_extensions {
            let name: &CStr = unsafe { CStr::from_ptr(extension.extension_name.as_ptr()) };
            required_extensions.remove(name);
        }

        required_extensions.len() == 0
    }

    fn find_queue_families(
        instance: &ash::Instance,
        surface_loader: &khr::surface::Instance,
        surface: vk::SurfaceKHR,
        device: vk::PhysicalDevice,
    ) -> QueueFamilyIndices {
        let queue_families: Vec<vk::QueueFamilyProperties> =
            unsafe { instance.get_physical_device_queue_family_properties(device) };

        let mut graphics: HashSet<u32> = HashSet::new();
        let mut present: HashSet<u32> = HashSet::new();
        let mut transfer: HashSet<u32> = HashSet::new();
        let mut transfer_only: HashSet<u32> = HashSet::new();
        let mut compute: HashSet<u32> = HashSet::new();

        let mut index: u32 = 0;

        for family in queue_families {
            if family.queue_count == 0 {
                continue;
            }

            if family.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                graphics.insert(index);
            }
            if family.queue_flags.contains(vk::QueueFlags::COMPUTE) {
                compute.insert(index);
            }

            if family.queue_flags.contains(vk::QueueFlags::TRANSFER) {
                if family.queue_flags & vk::QueueFlags::TRANSFER == vk::QueueFlags::empty() {
                    transfer_only.insert(index);
                }
                transfer.insert(index);
            }

            let present_support: bool = unsafe {
                surface_loader
                    .get_physical_device_surface_support(device, index, surface)
                    .unwrap()
            };

            if present_support {
                present.insert(index);
            }

            index += 1;
        }

        QueueFamilyIndices {
            graphics,
            present,
            transfer,
            transfer_only,
            compute,
        }
    }

    fn pick_physical_device(
        instance: &ash::Instance,
        surface_loader: &khr::surface::Instance,
        surface: vk::SurfaceKHR,
        required_device_extensions: &[*const c_char],
        score_physical_device_fn: ScorePhysicalDeviceFn,
    ) -> (
        vk::PhysicalDevice,
        QueueFamilyIndices,
        SwapChainSupportDetails,
    ) {
        let devices: Vec<vk::PhysicalDevice> =
            unsafe { instance.enumerate_physical_devices().unwrap() };

        if devices.len() <= 0 {
            panic!("no devices found");
        }

        let mut max_score: u32 = 0;
        let mut best = (
            vk::PhysicalDevice::null(),
            QueueFamilyIndices::default(),
            SwapChainSupportDetails::default(),
        );

        for device in devices {
            let queue_family_indices: QueueFamilyIndices =
                Self::find_queue_families(instance, surface_loader, surface, device);

            let swapchain_details = Self::query_swapchain_support(surface_loader, surface, device);
            let is_extensions_supported: bool =
                Self::check_device_extension_support(instance, device, required_device_extensions);

            let properties: vk::PhysicalDeviceProperties =
                unsafe { instance.get_physical_device_properties(device) };
            let features: vk::PhysicalDeviceFeatures =
                unsafe { instance.get_physical_device_features(device) };

            let score = score_physical_device_fn(
                properties,
                features,
                &queue_family_indices,
                &swapchain_details,
            );

            if score > max_score {
                max_score = score;
                best = (device, queue_family_indices, swapchain_details);
            }
        }

        if max_score == 0 {
            panic!("failed to find suitable device");
        }

        best
    }

    fn create_logical_device(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        queue_family_indices: &QueueFamilyIndices,
        required_device_extensions: &[*const c_char],
    ) -> ash::Device {
        let mut unique_indices: HashSet<u32> = HashSet::new();
        unique_indices.extend(&queue_family_indices.graphics);
        unique_indices.extend(&queue_family_indices.transfer);
        unique_indices.extend(&queue_family_indices.transfer_only);
        unique_indices.extend(&queue_family_indices.compute);

        // TODO: device creates all unique queues even though not needed / priority and queue count is not changable
        let queue_priority: f32 = 1.0;
        let mut queue_create_infos: Vec<vk::DeviceQueueCreateInfo> = vec![];

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
            enabled_extension_count: required_device_extensions.len() as u32,
            pp_enabled_extension_names: required_device_extensions.as_ptr(),
            p_enabled_features: &physical_device_features,
            ..Default::default()
        };

        unsafe {
            instance
                .create_device(physical_device, &create_info, None)
                .unwrap()
        }
    }

    fn find_memory_type(
        &self,
        suitable_memory_bits: u32,
        required_properties: vk::MemoryPropertyFlags,
    ) -> u32 {
        let memory_properties: vk::PhysicalDeviceMemoryProperties = unsafe {
            self.instance
                .get_physical_device_memory_properties(self.physical_device)
        };

        for i in 0..memory_properties.memory_type_count as usize {
            if suitable_memory_bits & (1 << i) != 0
                && memory_properties.memory_types[i]
                    .property_flags
                    .contains(required_properties)
            {
                return i as u32;
            }
        }

        panic!("failed to find suitable memory")
    }

    pub fn create_buffer(
        &self,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        properties: vk::MemoryPropertyFlags,
        sharing_mode: vk::SharingMode,
        queue_family_indices: &[u32],
    ) -> (vk::DeviceMemory, vk::Buffer) {
        let create_info = vk::BufferCreateInfo {
            s_type: vk::StructureType::BUFFER_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::BufferCreateFlags::empty(),
            size: size,
            usage: usage,
            sharing_mode: sharing_mode,
            queue_family_index_count: queue_family_indices.len() as u32,
            p_queue_family_indices: queue_family_indices.as_ptr(),
            ..Default::default()
        };

        let buffer: vk::Buffer = unsafe { self.device.create_buffer(&create_info, None).unwrap() };

        let mem_requirements: vk::MemoryRequirements =
            unsafe { self.device.get_buffer_memory_requirements(buffer) };

        let allocate_info = vk::MemoryAllocateInfo {
            s_type: vk::StructureType::MEMORY_ALLOCATE_INFO,
            p_next: ptr::null(),
            allocation_size: mem_requirements.size,
            memory_type_index: self.find_memory_type(mem_requirements.memory_type_bits, properties),
            ..Default::default()
        };

        let buffer_memory: vk::DeviceMemory =
            unsafe { self.device.allocate_memory(&allocate_info, None).unwrap() };

        unsafe {
            self.device
                .bind_buffer_memory(buffer, buffer_memory, 0)
                .unwrap();
        };

        (buffer_memory, buffer)
    }

    pub fn map_memory<T>(
        &self,
        memory: vk::DeviceMemory,
        offset: u64,
        size: vk::DeviceSize,
    ) -> *mut T {
        unsafe {
            self.device
                .map_memory(memory, offset, size, vk::MemoryMapFlags::empty())
                .unwrap() as *mut T
        }
    }

    pub fn unmap<T>(&self, memory: vk::DeviceMemory) {
        unsafe {
            self.device.unmap_memory(memory);
        };
    }

    pub fn copy_buffer(
        &self,
        src_buffer: vk::Buffer,
        dst_buffer: vk::Buffer,
        size: vk::DeviceSize,
        transfer_command_buffer: vk::CommandBuffer,
    ) {
        let region = vk::BufferCopy {
            src_offset: 0,
            dst_offset: 0,
            size: size,
        };

        unsafe {
            self.device.cmd_copy_buffer(
                transfer_command_buffer,
                src_buffer,
                dst_buffer,
                std::slice::from_ref(&region),
            );
        };

        // let mut barrier = vk::BufferMemoryBarrier {
        //     s_type: vk::StructureType::IMAGE_MEMORY_BARRIER,
        //     p_next: ptr::null(),
        //     src_access_mask: vk::AccessFlags::empty(),
        //     dst_access_mask: vk::AccessFlags::empty(),
        //     old_layout: old_layout,
        //     new_layout: new_layout,
        //     src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
        //     dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED
        // };
    }

    pub fn create_vertex_buffer<T>(
        &self,
        vertices: &[T],
        transfer_command_buffer: vk::CommandBuffer,
        sharing_mode: vk::SharingMode,
        queue_family_indices: &[u32],
    ) -> (vk::DeviceMemory, vk::Buffer) {
        let buffer_size: vk::DeviceSize = (size_of::<T>() * vertices.len()) as u64;
        let (staging_buffer_memory, staging_buffer): (vk::DeviceMemory, vk::Buffer) = self
            .create_buffer(
                buffer_size,
                vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::VERTEX_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                vk::SharingMode::EXCLUSIVE,
                &[],
            );

        unsafe {
            let data_ptr: *mut T = self
                .device
                .map_memory(
                    staging_buffer_memory,
                    0,
                    buffer_size,
                    vk::MemoryMapFlags::empty(),
                )
                .unwrap() as *mut T;
            data_ptr.copy_from_nonoverlapping(vertices.as_ptr(), vertices.len());
            self.device.unmap_memory(staging_buffer_memory);
        };

        let (vertex_buffer_memory, vertex_buffer): (vk::DeviceMemory, vk::Buffer) = self
            .create_buffer(
                buffer_size,
                vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
                sharing_mode,
                queue_family_indices,
            );

        self.copy_buffer(
            staging_buffer,
            vertex_buffer,
            buffer_size,
            transfer_command_buffer,
        );

        unsafe {
            self.device.destroy_buffer(staging_buffer, None);
            self.device.free_memory(staging_buffer_memory, None);
        }

        (vertex_buffer_memory, vertex_buffer)
    }

    pub fn create_index_buffer<T>(
        &self,
        indices: &[T],
        transfer_command_buffer: vk::CommandBuffer,
        sharing_mode: vk::SharingMode,
        queue_family_indices: &[u32],
    ) -> (vk::DeviceMemory, vk::Buffer) {
        let buffer_size: vk::DeviceSize = (size_of::<T>() * indices.len()) as u64;
        let (staging_buffer_memory, staging_buffer): (vk::DeviceMemory, vk::Buffer) = self
            .create_buffer(
                buffer_size,
                vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::INDEX_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                vk::SharingMode::EXCLUSIVE,
                &[],
            );

        unsafe {
            let data_ptr: *mut T = self
                .device
                .map_memory(
                    staging_buffer_memory,
                    0,
                    buffer_size,
                    vk::MemoryMapFlags::empty(),
                )
                .unwrap() as *mut T;
            data_ptr.copy_from_nonoverlapping(indices.as_ptr(), indices.len());
            self.device.unmap_memory(staging_buffer_memory);
        };

        let (index_buffer_memory, index_buffer): (vk::DeviceMemory, vk::Buffer) = self
            .create_buffer(
                buffer_size,
                vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
                sharing_mode,
                queue_family_indices,
            );

        self.copy_buffer(
            staging_buffer,
            index_buffer,
            buffer_size,
            transfer_command_buffer,
        );

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
            self.device
                .create_shader_module(&create_info, None)
                .unwrap()
        }
    }

    pub fn destroy_shader_module(&self, shader: vk::ShaderModule) {
        unsafe {
            self.device.destroy_shader_module(shader, None);
        };
    }

    pub fn create_descriptor_pool(
        &self,
        pool_sizes: &[vk::DescriptorPoolSize],
        max_sets: u32,
    ) -> vk::DescriptorPool {
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
            self.device
                .create_descriptor_pool(&create_info, None)
                .unwrap()
        }
    }

    pub fn create_descriptor_set_layout(
        &self,
        bindings: &[vk::DescriptorSetLayoutBinding],
    ) -> vk::DescriptorSetLayout {
        let create_info = vk::DescriptorSetLayoutCreateInfo {
            s_type: vk::StructureType::DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::DescriptorSetLayoutCreateFlags::empty(),
            binding_count: bindings.len() as u32,
            p_bindings: bindings.as_ptr(),
            ..Default::default()
        };

        unsafe {
            self.device
                .create_descriptor_set_layout(&create_info, None)
                .unwrap()
        }
    }

    pub fn create_descriptor_sets(
        &self,
        descriptor_pool: vk::DescriptorPool,
        layouts: &[vk::DescriptorSetLayout],
    ) -> Vec<vk::DescriptorSet> {
        let allocate_info = vk::DescriptorSetAllocateInfo {
            s_type: vk::StructureType::DESCRIPTOR_SET_ALLOCATE_INFO,
            p_next: ptr::null(),
            descriptor_pool: descriptor_pool,
            descriptor_set_count: layouts.len() as u32,
            p_set_layouts: layouts.as_ptr(),
            ..Default::default()
        };

        unsafe {
            self.device
                .allocate_descriptor_sets(&allocate_info)
                .unwrap()
        }
    }

    pub fn write_descriptors(
        &self,
        writes: &[vk::WriteDescriptorSet],
        copies: &[vk::CopyDescriptorSet],
    ) {
        unsafe {
            self.device.update_descriptor_sets(&writes, copies);
        }
    }

    pub fn create_render_pass(
        &self,
        attachments: &[vk::AttachmentDescription],
        subpasses: &[vk::SubpassDescription],
        dependencies: &[vk::SubpassDependency],
    ) -> vk::RenderPass {
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
            self.device
                .create_render_pass(&render_pass_create_info, None)
                .unwrap()
        }
    }

    pub fn create_image_view(
        &self,
        image: vk::Image,
        format: vk::Format,
        aspect_mask: vk::ImageAspectFlags,
    ) -> vk::ImageView {
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
                a: vk::ComponentSwizzle::A,
            },
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: aspect_mask,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            },
            ..Default::default()
        };

        unsafe { self.device.create_image_view(&create_info, None).unwrap() }
    }

    pub fn destroy_image_view(&self, image_view: vk::ImageView) {
        unsafe { self.device.destroy_image_view(image_view, None) };
    }

    pub fn destroy_image(&self, image: vk::Image) {
        unsafe { self.device.destroy_image(image, None) };
    }

    pub fn free_memory(&self, memory: vk::DeviceMemory) {
        unsafe { self.device.free_memory(memory, None) };
    }

    pub fn destroy_semaphore(&self, semaphore: vk::Semaphore) {
        unsafe { self.device.destroy_semaphore(semaphore, None) };
    }

    pub fn destroy_fence(&self, fence: vk::Fence) {
        unsafe { self.device.destroy_fence(fence, None) };
    }



    pub fn create_image(
        &self,
        width: u32,
        height: u32,
        format: vk::Format,
        tiling: vk::ImageTiling,
        usage: vk::ImageUsageFlags,
        properties: vk::MemoryPropertyFlags,
        sharing_mode: vk::SharingMode,
        queue_family_indices: &[u32],
    ) -> (vk::Image, vk::DeviceMemory) {
        let create_info = vk::ImageCreateInfo {
            s_type: vk::StructureType::IMAGE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::ImageCreateFlags::empty(),
            image_type: vk::ImageType::TYPE_2D,
            format: format,
            extent: vk::Extent3D {
                width: width,
                height: height,
                depth: 1,
            },
            mip_levels: 1,
            array_layers: 1,
            samples: vk::SampleCountFlags::TYPE_1,
            tiling: tiling,
            usage: usage,
            sharing_mode: sharing_mode,
            queue_family_index_count: queue_family_indices.len() as u32,
            p_queue_family_indices: queue_family_indices.as_ptr(),
            initial_layout: vk::ImageLayout::UNDEFINED,
            ..Default::default()
        };

        let image: vk::Image = unsafe { self.device.create_image(&create_info, None).unwrap() };

        let mem_requirements: vk::MemoryRequirements =
            unsafe { self.device.get_image_memory_requirements(image) };

        let allocate_info = vk::MemoryAllocateInfo {
            s_type: vk::StructureType::MEMORY_ALLOCATE_INFO,
            p_next: ptr::null(),
            allocation_size: mem_requirements.size,
            memory_type_index: self.find_memory_type(mem_requirements.memory_type_bits, properties),
            ..Default::default()
        };

        let image_memory: vk::DeviceMemory =
            unsafe { self.device.allocate_memory(&allocate_info, None).unwrap() };

        unsafe {
            self.device
                .bind_image_memory(image, image_memory, 0)
                .unwrap()
        };

        (image, image_memory)
    }

    pub fn create_command_buffers(
        &self,
        command_pool: vk::CommandPool,
        level: vk::CommandBufferLevel,
        num_buffers: u32,
    ) -> Vec<vk::CommandBuffer> {
        let allocate_info = vk::CommandBufferAllocateInfo {
            s_type: vk::StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
            p_next: ptr::null(),
            command_pool: command_pool,
            level: level,
            command_buffer_count: num_buffers,
            ..Default::default()
        };

        unsafe {
            self.device
                .allocate_command_buffers(&allocate_info)
                .unwrap()
        }
    }

    pub fn find_supported_format(
        &self,
        candidates: &Vec<vk::Format>,
        tiling: vk::ImageTiling,
        features: vk::FormatFeatureFlags,
    ) -> vk::Format {
        for format in candidates {
            let properties: vk::FormatProperties = unsafe {
                self.instance
                    .get_physical_device_format_properties(self.physical_device, *format)
            };

            if tiling == vk::ImageTiling::LINEAR
                && properties.linear_tiling_features.contains(features)
                || tiling == vk::ImageTiling::OPTIMAL
                    && properties.optimal_tiling_features.contains(features)
            {
                return *format;
            }
        }

        vk::Format::UNDEFINED
    }

    pub fn create_pipeline_layout(
        &self,
        descriptor_set_layouts: &[vk::DescriptorSetLayout],
        push_constants: &[vk::PushConstantRange],
    ) -> vk::PipelineLayout {
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
            self.device
                .create_pipeline_layout(&pipeline_layout_create_info, None)
                .unwrap()
        }
    }

    pub fn create_framebuffer(
        &self,
        render_pass: vk::RenderPass,
        attachments: &[vk::ImageView],
        extent: vk::Extent2D,
        layers: u32,
    ) -> vk::Framebuffer {
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

        unsafe { self.device.create_framebuffer(&create_info, None).unwrap() }
    }

    pub fn destroy_framebuffer(&self, framebuffer: vk::Framebuffer) {
        unsafe { self.device.destroy_framebuffer(framebuffer, None) };
    }

    pub fn destroy_descriptor_pool(&self, descriptor_pool: vk::DescriptorPool) {
        unsafe { self.device.destroy_descriptor_pool(descriptor_pool, None) };
    }

    pub fn destroy_descriptor_set_layout(&self, descriptor_set_layout: vk::DescriptorSetLayout) {
        unsafe {
            self.device
                .destroy_descriptor_set_layout(descriptor_set_layout, None)
        };
    }

    pub fn destroy_pipeline_layout(&self, pipeline_layout: vk::PipelineLayout) {
        unsafe { self.device.destroy_pipeline_layout(pipeline_layout, None) };
    }

    pub fn destroy_pipeline(&self, pipeline: vk::Pipeline) {
        unsafe { self.device.destroy_pipeline(pipeline, None) };
    }

    pub fn destroy_render_pass(&self, render_pass: vk::RenderPass) {
        unsafe { self.device.destroy_render_pass(render_pass, None) };
    }

    pub fn destroy_buffer(&self, buffer: vk::Buffer) {
        unsafe { self.device.destroy_buffer(buffer, None) };
    }

    pub fn wait_idle(&self) {
        unsafe { self.device.device_wait_idle().unwrap() };
    }

    pub fn create_texture_image(
        &self,
        pixels: &[u8],
        width: u32,
        height: u32,
        format: vk::Format,
        tiling: vk::ImageTiling,
        transfer_command_buffer: vk::CommandBuffer,
        sharing_mode: vk::SharingMode,
        queue_family_indices: &[u32],
    ) -> (vk::Image, vk::DeviceMemory) {
        let buffer_size: vk::DeviceSize = (size_of::<u8>() * pixels.len()) as u64;

        let (staging_buffer_memory, staging_buffer): (vk::DeviceMemory, vk::Buffer) = self
            .create_buffer(
                buffer_size,
                vk::BufferUsageFlags::TRANSFER_SRC,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                vk::SharingMode::EXCLUSIVE,
                &[],
            );

        unsafe {
            let data_ptr: *mut u8 = self
                .device
                .map_memory(
                    staging_buffer_memory,
                    0,
                    buffer_size,
                    vk::MemoryMapFlags::empty(),
                )
                .unwrap() as *mut u8;
            data_ptr.copy_from_nonoverlapping(pixels.as_ptr(), pixels.len());
            self.device.unmap_memory(staging_buffer_memory);
        };

        let (texture_image, image_memory) = self.create_image(
            width,
            height,
            format,
            tiling,
            vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            sharing_mode,
            queue_family_indices,
        );

        self.transition_image_layout(
            texture_image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            transfer_command_buffer,
        );
        self.copy_buffer_to_image(
            staging_buffer,
            texture_image,
            width,
            height,
            transfer_command_buffer,
        );
        self.transition_image_layout(
            texture_image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            transfer_command_buffer,
        );

        unsafe {
            self.device.destroy_buffer(staging_buffer, None);
            self.device.free_memory(staging_buffer_memory, None);
        };

        (texture_image, image_memory)
    }

    pub fn transition_image_layout(
        &self,
        image: vk::Image,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
        transfer_command_buffer: vk::CommandBuffer,
    ) {
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
                layer_count: 1,
            },
            ..Default::default()
        };

        let source_stage: vk::PipelineStageFlags;
        let destination_stage: vk::PipelineStageFlags;

        if old_layout == vk::ImageLayout::UNDEFINED
            && new_layout == vk::ImageLayout::TRANSFER_DST_OPTIMAL
        {
            barrier.src_access_mask = vk::AccessFlags::empty();
            barrier.dst_access_mask = vk::AccessFlags::TRANSFER_WRITE;

            source_stage = vk::PipelineStageFlags::TOP_OF_PIPE;
            destination_stage = vk::PipelineStageFlags::TRANSFER;
        } else if old_layout == vk::ImageLayout::TRANSFER_DST_OPTIMAL
            && new_layout == vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
        {
            barrier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
            barrier.dst_access_mask = vk::AccessFlags::SHADER_READ;

            source_stage = vk::PipelineStageFlags::TRANSFER;
            destination_stage = vk::PipelineStageFlags::FRAGMENT_SHADER;
        } else {
            panic!("unsupported layout transition!");
        }

        unsafe {
            self.device.cmd_pipeline_barrier(
                transfer_command_buffer,
                source_stage,
                destination_stage,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                std::slice::from_ref(&barrier),
            );
        }
    }

    pub fn copy_buffer_to_image(
        &self,
        buffer: vk::Buffer,
        image: vk::Image,
        width: u32,
        height: u32,
        transfer_command_buffer: vk::CommandBuffer,
    ) {
        let region = vk::BufferImageCopy {
            buffer_offset: 0,
            buffer_row_length: 0,
            buffer_image_height: 0,
            image_subresource: vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            },
            image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
            image_extent: vk::Extent3D {
                width,
                height,
                depth: 1,
            },
        };

        unsafe {
            self.device.cmd_copy_buffer_to_image(
                transfer_command_buffer,
                buffer,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                std::slice::from_ref(&region),
            );
        };
    }

    pub fn destroy_device(&self) {
        unsafe { self.device.destroy_device(None) };
    }

    pub fn destroy_instance(&self) {
        unsafe { self.instance.destroy_instance(None) };
    }
}
