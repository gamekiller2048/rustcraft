use ash::vk;
use std::{ffi::c_void, ptr};

use crate::vulkan_context::VulkanContext;

pub struct Buffer {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub mapped_memory: *mut c_void,
}

impl Buffer {
    pub fn create_vertex_buffer<T>(
        context: &VulkanContext,
        vertices: &[T],
        transfer_command_buffer: vk::CommandBuffer,
        sharing_mode: vk::SharingMode,
        queue_family_indices: &[u32],
    ) -> (Self, Self) {
        let buffer_size: vk::DeviceSize = (size_of::<T>() * vertices.len()) as u64;
        let (staging_buffer_memory, staging_buffer): (vk::DeviceMemory, vk::Buffer) =
            Self::create_buffer(
                context,
                buffer_size,
                vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::VERTEX_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                vk::SharingMode::EXCLUSIVE,
                &[],
            );

        let data_ptr: *mut T = context.map_memory(staging_buffer_memory, 0, buffer_size);
        unsafe {
            data_ptr.copy_from_nonoverlapping(vertices.as_ptr(), vertices.len());
        }
        context.unmap(staging_buffer_memory);

        let (vertex_buffer_memory, vertex_buffer): (vk::DeviceMemory, vk::Buffer) =
            Self::create_buffer(
                context,
                buffer_size,
                vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
                sharing_mode,
                queue_family_indices,
            );

        Self::copy_buffer(
            context,
            staging_buffer,
            vertex_buffer,
            buffer_size,
            transfer_command_buffer,
        );

        (
            Self {
                buffer: vertex_buffer,
                memory: vertex_buffer_memory,
                mapped_memory: ptr::null_mut(),
            },
            Self {
                buffer: staging_buffer,
                memory: staging_buffer_memory,
                mapped_memory: ptr::null_mut(),
            },
        )
    }

    pub fn create_index_buffer<T>(
        context: &VulkanContext,
        indices: &[T],
        transfer_command_buffer: vk::CommandBuffer,
        sharing_mode: vk::SharingMode,
        queue_family_indices: &[u32],
    ) -> (Self, Self) {
        let buffer_size: vk::DeviceSize = (size_of::<T>() * indices.len()) as u64;
        let (staging_buffer_memory, staging_buffer): (vk::DeviceMemory, vk::Buffer) =
            Self::create_buffer(
                context,
                buffer_size,
                vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::INDEX_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                vk::SharingMode::EXCLUSIVE,
                &[],
            );

        let data_ptr: *mut T = context.map_memory(staging_buffer_memory, 0, buffer_size);
        unsafe {
            data_ptr.copy_from_nonoverlapping(indices.as_ptr(), indices.len());
        }
        context.unmap(staging_buffer_memory);

        let (index_buffer_memory, index_buffer): (vk::DeviceMemory, vk::Buffer) =
            Self::create_buffer(
                context,
                buffer_size,
                vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
                sharing_mode,
                queue_family_indices,
            );

        Self::copy_buffer(
            context,
            staging_buffer,
            index_buffer,
            buffer_size,
            transfer_command_buffer,
        );

        (
            Self {
                buffer: index_buffer,
                memory: index_buffer_memory,
                mapped_memory: ptr::null_mut(),
            },
            Self {
                buffer: staging_buffer,
                memory: staging_buffer_memory,
                mapped_memory: ptr::null_mut(),
            },
        )
    }

    pub fn new(
        context: &VulkanContext,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        properties: vk::MemoryPropertyFlags,
        sharing_mode: vk::SharingMode,
        queue_family_indices: &[u32],
    ) -> Self {
        let (memory, buffer) = Self::create_buffer(
            context,
            size,
            usage,
            properties,
            sharing_mode,
            queue_family_indices,
        );

        Self {
            buffer,
            memory,
            mapped_memory: ptr::null_mut(),
        }
    }

    pub fn destroy(&self, context: &VulkanContext) {
        Self::destroy_buffer(context, self.buffer);
        unsafe {
            context.device.free_memory(self.memory, None);
        }
    }

    pub fn create_buffer(
        context: &VulkanContext,
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

        let buffer: vk::Buffer =
            unsafe { context.device.create_buffer(&create_info, None).unwrap() };

        let mem_requirements: vk::MemoryRequirements =
            unsafe { context.device.get_buffer_memory_requirements(buffer) };

        let allocate_info = vk::MemoryAllocateInfo {
            s_type: vk::StructureType::MEMORY_ALLOCATE_INFO,
            p_next: ptr::null(),
            allocation_size: mem_requirements.size,
            memory_type_index: context
                .find_memory_type(mem_requirements.memory_type_bits, properties),
            ..Default::default()
        };

        let buffer_memory: vk::DeviceMemory = unsafe {
            context
                .device
                .allocate_memory(&allocate_info, None)
                .unwrap()
        };

        unsafe {
            context
                .device
                .bind_buffer_memory(buffer, buffer_memory, 0)
                .unwrap();
        };

        (buffer_memory, buffer)
    }

    pub fn copy_buffer(
        context: &VulkanContext,
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
            context.device.cmd_copy_buffer(
                transfer_command_buffer,
                src_buffer,
                dst_buffer,
                std::slice::from_ref(&region),
            );
        }
    }

    pub fn destroy_buffer(context: &VulkanContext, buffer: vk::Buffer) {
        unsafe {
            context.device.destroy_buffer(buffer, None);
        }
    }
}
