use ash::{khr, vk};
use log::debug;
use std::cell::RefCell;
use std::collections::HashSet;
use std::ffi::{CStr, c_char};
use std::rc::Rc;
use std::{ptr, u64};

use crate::vulkan_allocator::VulkanAllocator;

#[derive(Default, Debug)]
pub struct QueueFamilyIndices {
    pub graphics: HashSet<u32>,
    pub present: HashSet<u32>,
    pub transfer: HashSet<u32>,
    pub transfer_only: HashSet<u32>,
    pub compute: HashSet<u32>,
    pub compute_only: HashSet<u32>,
}

#[derive(Default, Debug)]
pub struct SwapChainSupportDetails {
    pub capabilities: vk::SurfaceCapabilitiesKHR,
    pub formats: Vec<vk::SurfaceFormatKHR>,
    pub present_modes: Vec<vk::PresentModeKHR>,
}

pub struct VulkanContext {
    pub instance: ash::Instance,

    pub physical_device: vk::PhysicalDevice,
    pub queue_family_indices: QueueFamilyIndices,

    pub device: ash::Device,
    pub swapchain_loader: khr::swapchain::Device,
    pub allocator: Rc<RefCell<VulkanAllocator>>,
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
        allocator: &Rc<RefCell<VulkanAllocator>>,
    ) -> Self {
        let (physical_device, queue_family_indices) = Self::pick_physical_device(
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
            allocator,
        );
        let swapchain_loader = khr::swapchain::Device::new(&instance, &device);

        Self {
            instance,
            physical_device,
            queue_family_indices,
            device,
            swapchain_loader,
            allocator: allocator.clone(),
        }
    }

    pub fn query_swapchain_support(
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
        let mut compute_only: HashSet<u32> = HashSet::new();

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
                if family.queue_flags & vk::QueueFlags::COMPUTE == vk::QueueFlags::empty() {
                    compute_only.insert(index);
                }
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
            compute_only,
        }
    }

    fn pick_physical_device(
        instance: &ash::Instance,
        surface_loader: &khr::surface::Instance,
        surface: vk::SurfaceKHR,
        required_device_extensions: &[*const c_char],
        score_physical_device_fn: ScorePhysicalDeviceFn,
    ) -> (vk::PhysicalDevice, QueueFamilyIndices) {
        let devices: Vec<vk::PhysicalDevice> =
            unsafe { instance.enumerate_physical_devices().unwrap() };

        if devices.len() <= 0 {
            panic!("no devices found");
        }

        let mut max_score: u32 = 0;
        let mut best = (vk::PhysicalDevice::null(), QueueFamilyIndices::default());

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
                best = (device, queue_family_indices);
            }
        }

        if max_score == 0 {
            panic!("failed to find suitable device");
        }

        debug!("chosen device with max score: {}", max_score);
        debug!("chosen device has queue families: {:?}", best.1);

        best
    }

    fn create_logical_device(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        queue_family_indices: &QueueFamilyIndices,
        required_device_extensions: &[*const c_char],
        allocator: &Rc<RefCell<VulkanAllocator>>,
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
                .create_device(
                    physical_device,
                    &create_info,
                    Some(&allocator.borrow_mut().get_allocation_callbacks()),
                )
                .unwrap()
        }
    }

    pub fn find_memory_type(
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

    pub fn unmap(&self, memory: vk::DeviceMemory) {
        unsafe {
            self.device.unmap_memory(memory);
        }
    }

    pub fn free_memory(&self, memory: vk::DeviceMemory) {
        unsafe {
            self.device.free_memory(memory, None);
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

    pub fn wait_idle(&self) {
        unsafe {
            self.device.device_wait_idle().unwrap();
        }
    }

    pub fn destroy_device(&self) {
        unsafe {
            self.device.destroy_device(Some(
                &self.allocator.borrow_mut().get_allocation_callbacks(),
            ));
        }
    }

    pub fn destroy_instance(&self) {
        unsafe {
            self.instance.destroy_instance(Some(
                &self.allocator.borrow_mut().get_allocation_callbacks(),
            ));
        }
    }
}
