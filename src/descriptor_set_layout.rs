use std::{collections::HashMap, ptr};
use ash::vk;

pub struct DescriptorSetLayoutBuilder<'a> {
    pub bindings: Vec<vk::DescriptorSetLayoutBinding<'a>>
}

impl<'a> DescriptorSetLayoutBuilder<'a> {
    pub fn new() -> Self {
        DescriptorSetLayoutBuilder {bindings: vec![]}
    }

    pub fn add_uniform_buffer(mut self, binding: u32, num_descriptors: u32, stage_flags: vk::ShaderStageFlags) -> Self {
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

    pub fn add_image_sampler(mut self, binding: u32, num_descriptors: u32, stage_flags: vk::ShaderStageFlags) -> Self {
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
            *descriptor_counts.entry(binding.descriptor_type).or_insert(0) += binding.descriptor_count * max_sets;
        }

        let mut pool_sizes: Vec<vk::DescriptorPoolSize> = Vec::with_capacity(self.bindings.len());
        for (descriptor_type, count) in descriptor_counts.into_iter() {
            pool_sizes.push(vk::DescriptorPoolSize {ty: descriptor_type, descriptor_count: count});
        }

        pool_sizes
    }
}