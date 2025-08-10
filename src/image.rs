use ash::vk;
use std::cell::RefCell;
use std::ptr;
use std::rc::Rc;

use crate::buffer::Buffer;
use crate::vulkan_allocator::VulkanAllocator;
use crate::vulkan_context::VulkanContext;

pub struct Image {
    pub image: vk::Image,
    pub memory: vk::DeviceMemory,
}

impl Image {
    pub fn create_image(
        context: &VulkanContext,
        width: u32,
        height: u32,
        depth: u32,
        mip_levels: u32,
        array_layers: u32,
        format: vk::Format,
        tiling: vk::ImageTiling,
        usage: vk::ImageUsageFlags,
        properties: vk::MemoryPropertyFlags,
        sharing_mode: vk::SharingMode,
        queue_family_indices: &[u32],
        allocator: &Rc<RefCell<VulkanAllocator>>,
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
                depth: depth,
            },
            mip_levels: mip_levels,
            array_layers: array_layers,
            samples: vk::SampleCountFlags::TYPE_1,
            tiling: tiling,
            usage: usage,
            sharing_mode: sharing_mode,
            queue_family_index_count: queue_family_indices.len() as u32,
            p_queue_family_indices: queue_family_indices.as_ptr(),
            initial_layout: vk::ImageLayout::UNDEFINED,
            ..Default::default()
        };

        let image: vk::Image = unsafe {
            context
                .device
                .create_image(
                    &create_info,
                    Some(&allocator.borrow_mut().get_allocation_callbacks()),
                )
                .unwrap()
        };

        let mem_requirements: vk::MemoryRequirements =
            unsafe { context.device.get_image_memory_requirements(image) };

        let allocate_info = vk::MemoryAllocateInfo {
            s_type: vk::StructureType::MEMORY_ALLOCATE_INFO,
            p_next: ptr::null(),
            allocation_size: mem_requirements.size,
            memory_type_index: context
                .find_memory_type(mem_requirements.memory_type_bits, properties),
            ..Default::default()
        };

        let image_memory: vk::DeviceMemory = unsafe {
            context
                .device
                .allocate_memory(
                    &allocate_info,
                    Some(&allocator.borrow_mut().get_allocation_callbacks()),
                )
                .unwrap()
        };

        unsafe {
            context
                .device
                .bind_image_memory(image, image_memory, 0)
                .unwrap();
        }

        (image, image_memory)
    }

    pub fn destroy_image(
        context: &VulkanContext,
        image: vk::Image,
        allocator: &Rc<RefCell<VulkanAllocator>>,
    ) {
        unsafe {
            context.device.destroy_image(
                image,
                Some(&allocator.borrow_mut().get_allocation_callbacks()),
            );
        }
    }

    pub fn create_texture_image(
        context: &VulkanContext,
        pixels: &[u8],
        width: u32,
        height: u32,
        format: vk::Format,
        tiling: vk::ImageTiling,
        transfer_command_buffer: vk::CommandBuffer,
        sharing_mode: vk::SharingMode,
        queue_family_indices: &[u32],
        allocator: &Rc<RefCell<VulkanAllocator>>,
    ) -> (Self, Buffer) {
        let buffer_size: vk::DeviceSize = (size_of::<u8>() * pixels.len()) as u64;

        let (staging_buffer_memory, staging_buffer): (vk::DeviceMemory, vk::Buffer) =
            Buffer::create_buffer(
                context,
                buffer_size,
                vk::BufferUsageFlags::TRANSFER_SRC,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                vk::SharingMode::EXCLUSIVE,
                &[],
                allocator,
            );

        let data_ptr: *mut u8 = context.map_memory(staging_buffer_memory, 0, buffer_size);
        unsafe {
            data_ptr.copy_from_nonoverlapping(pixels.as_ptr(), pixels.len());
        }
        context.unmap(staging_buffer_memory);

        let (texture_image, texture_image_memory) = Self::create_image(
            context,
            width,
            height,
            1,
            1,
            1,
            format,
            tiling,
            vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            sharing_mode,
            queue_family_indices,
            allocator,
        );

        Self::transition_image_layout(
            context,
            texture_image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            transfer_command_buffer,
        );
        Self::copy_buffer_to_image(
            context,
            staging_buffer,
            texture_image,
            width,
            height,
            transfer_command_buffer,
        );
        Self::transition_image_layout(
            context,
            texture_image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            transfer_command_buffer,
        );

        (
            Self {
                image: texture_image,
                memory: texture_image_memory,
            },
            Buffer {
                buffer: staging_buffer,
                memory: staging_buffer_memory,
                mapped_memory: ptr::null_mut(),
            },
        )
    }

    pub fn transition_image_layout(
        context: &VulkanContext,
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
            context.device.cmd_pipeline_barrier(
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
        context: &VulkanContext,
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
            context.device.cmd_copy_buffer_to_image(
                transfer_command_buffer,
                buffer,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                std::slice::from_ref(&region),
            );
        }
    }

    pub fn destory_image(
        context: &VulkanContext,
        image: vk::Image,
        allocator: &Rc<RefCell<VulkanAllocator>>,
    ) {
        unsafe {
            context.device.destroy_image(
                image,
                Some(&allocator.borrow_mut().get_allocation_callbacks()),
            );
        }
    }

    pub fn destroy(&self, context: &VulkanContext, allocator: &Rc<RefCell<VulkanAllocator>>) {
        Self::destory_image(context, self.image, allocator);
        context.free_memory(self.memory);
    }
}
