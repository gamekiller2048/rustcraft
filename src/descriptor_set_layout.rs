use ash::vk;
use std::{collections::HashMap, marker::PhantomData, ptr, sync::Arc};

use crate::vulkan_allocator::VulkanAllocator;
use crate::vulkan_context::VulkanContext;
pub struct DescriptorSetLayout;

impl DescriptorSetLayout {
    pub fn create_descriptor_set_layout(
        context: &VulkanContext,
        bindings: &[vk::DescriptorSetLayoutBinding],
        allocator: &Arc<VulkanAllocator>,
    ) -> vk::DescriptorSetLayout {
        let create_info = vk::DescriptorSetLayoutCreateInfo {
            s_type: vk::StructureType::DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::DescriptorSetLayoutCreateFlags::empty(),
            binding_count: bindings.len() as u32,
            p_bindings: bindings.as_ptr(),
            _marker: PhantomData,
        };

        unsafe {
            context
                .device
                .create_descriptor_set_layout(&create_info, Some(&allocator.callbacks))
                .unwrap()
        }
    }

    pub fn destroy_descriptor_set_layout(
        context: &VulkanContext,
        descriptor_set_layout: vk::DescriptorSetLayout,
        allocator: &Arc<VulkanAllocator>,
    ) {
        unsafe {
            context
                .device
                .destroy_descriptor_set_layout(descriptor_set_layout, Some(&allocator.callbacks))
        };
    }
}

pub struct DescriptorSetLayoutBuilder<'a> {
    pub bindings: Vec<vk::DescriptorSetLayoutBinding<'a>>,
}

impl<'a> DescriptorSetLayoutBuilder<'a> {
    pub fn new() -> Self {
        DescriptorSetLayoutBuilder { bindings: vec![] }
    }

    pub fn add_uniform_buffer(
        mut self,
        binding: u32,
        num_descriptors: u32,
        stage_flags: vk::ShaderStageFlags,
    ) -> Self {
        let layout_binding = vk::DescriptorSetLayoutBinding {
            binding,
            descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: num_descriptors,
            stage_flags,
            p_immutable_samplers: ptr::null(),
            _marker: PhantomData,
        };

        self.bindings.push(layout_binding);
        self
    }

    pub fn add_image_sampler(
        mut self,
        binding: u32,
        num_descriptors: u32,
        stage_flags: vk::ShaderStageFlags,
    ) -> Self {
        let layout_binding = vk::DescriptorSetLayoutBinding {
            binding,
            descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: num_descriptors,
            stage_flags,
            p_immutable_samplers: ptr::null(),
            _marker: PhantomData,
        };

        self.bindings.push(layout_binding);
        self
    }

    pub fn add_storage_buffer(
        mut self,
        binding: u32,
        num_descriptors: u32,
        stage_flags: vk::ShaderStageFlags,
    ) -> Self {
        let layout_binding = vk::DescriptorSetLayoutBinding {
            binding,
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: num_descriptors,
            stage_flags,
            p_immutable_samplers: ptr::null(),
            _marker: PhantomData,
        };

        self.bindings.push(layout_binding);
        self
    }

    // `max_sets` is the max number of descriptor sets for this layout.
    // NOTE: if a pool uses multiple layouts, pool_sizes must be summed accross all layouts
    pub fn calculate_descriptor_counts(&self, max_sets: u32) -> HashMap<vk::DescriptorType, u32> {
        let mut descriptor_counts: HashMap<vk::DescriptorType, u32> = HashMap::new();

        for binding in self.bindings.iter() {
            *descriptor_counts
                .entry(binding.descriptor_type)
                .or_insert(0) += binding.descriptor_count * max_sets;
        }

        descriptor_counts
    }
}

pub fn create_descriptor_uniform_buffer_write<'a>(
    set: vk::DescriptorSet,
    buffer_info: &vk::DescriptorBufferInfo,
    binding: u32,
    index_offset: u32,
    num_descriptors: u32,
) -> vk::WriteDescriptorSet<'a> {
    vk::WriteDescriptorSet {
        s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
        p_next: ptr::null(),
        dst_set: set,
        dst_binding: binding,
        dst_array_element: index_offset,
        descriptor_count: num_descriptors,
        descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
        p_image_info: ptr::null(),
        p_buffer_info: buffer_info,
        p_texel_buffer_view: ptr::null(),
        _marker: PhantomData,
    }
}

pub fn create_descriptor_image_sampler_write<'a>(
    set: vk::DescriptorSet,
    image_info: &vk::DescriptorImageInfo,
    binding: u32,
    index_offset: u32,
    num_descriptors: u32,
) -> vk::WriteDescriptorSet<'a> {
    vk::WriteDescriptorSet {
        s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
        p_next: ptr::null(),
        dst_set: set,
        dst_binding: binding,
        dst_array_element: index_offset,
        descriptor_count: num_descriptors,
        descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        p_image_info: image_info,
        p_buffer_info: ptr::null(),
        p_texel_buffer_view: ptr::null(),
        _marker: PhantomData,
    }
}

pub fn create_descriptor_storage_buffer_write<'a>(
    set: vk::DescriptorSet,
    buffer_info: &vk::DescriptorBufferInfo,
    binding: u32,
    index_offset: u32,
    num_descriptors: u32,
) -> vk::WriteDescriptorSet<'a> {
    vk::WriteDescriptorSet {
        s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
        p_next: ptr::null(),
        dst_set: set,
        dst_binding: binding,
        dst_array_element: index_offset,
        descriptor_count: num_descriptors,
        descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
        p_image_info: ptr::null(),
        p_buffer_info: buffer_info,
        p_texel_buffer_view: ptr::null(),
        _marker: PhantomData,
    }
}
