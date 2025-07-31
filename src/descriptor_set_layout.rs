use ash::vk;
use std::{collections::HashMap, ptr};

use super::vulkan_context::VulkanContext;

pub struct DescriptorSetLayout;

impl DescriptorSetLayout {
    pub fn create_descriptor_set_layout(
        context: &VulkanContext,
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
            context
                .device
                .create_descriptor_set_layout(&create_info, None)
                .unwrap()
        }
    }

    pub fn destroy_descriptor_set_layout(
        context: &VulkanContext,
        descriptor_set_layout: vk::DescriptorSetLayout,
    ) {
        unsafe {
            context
                .device
                .destroy_descriptor_set_layout(descriptor_set_layout, None)
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
            ..Default::default()
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
            ..Default::default()
        };

        self.bindings.push(layout_binding);
        self
    }

    pub fn calculate_pool_sizes(&self, max_sets: u32) -> Vec<vk::DescriptorPoolSize> {
        let mut descriptor_counts: HashMap<vk::DescriptorType, u32> = HashMap::new();

        for binding in self.bindings.iter() {
            *descriptor_counts
                .entry(binding.descriptor_type)
                .or_insert(0) += binding.descriptor_count * max_sets;
        }

        let mut pool_sizes: Vec<vk::DescriptorPoolSize> = Vec::with_capacity(self.bindings.len());
        for (descriptor_type, count) in descriptor_counts.into_iter() {
            pool_sizes.push(vk::DescriptorPoolSize {
                ty: descriptor_type,
                descriptor_count: count,
            });
        }

        pool_sizes
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
        ..Default::default()
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
        ..Default::default()
    }
}
