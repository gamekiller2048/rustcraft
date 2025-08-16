use ash::vk;
use core::slice;
use std::{marker::PhantomData, ptr, sync::Arc};

use crate::shader_module::ShaderModule;
use crate::vulkan_allocator::VulkanAllocator;
use crate::vulkan_context::VulkanContext;

pub struct ComputePipeline {
    pub pipeline: vk::Pipeline,
    pub compute_shader: vk::ShaderModule,
}

impl ComputePipeline {
    pub fn new(
        context: &VulkanContext,
        pipeline_layout: vk::PipelineLayout,
        compute_shader: vk::ShaderModule,
        allocator: &Arc<VulkanAllocator>,
    ) -> Self {
        let pipeline = Self::create_pipeline(context, pipeline_layout, compute_shader, allocator);

        Self {
            pipeline,
            compute_shader,
        }
    }

    pub fn destroy(&self, context: &VulkanContext, allocator: &Arc<VulkanAllocator>) {
        ComputePipeline::destroy_pipeline(context, self.pipeline, allocator);
        ShaderModule::destroy_shader_module(context, self.compute_shader, allocator);
    }

    pub fn create_pipeline(
        context: &VulkanContext,
        pipeline_layout: vk::PipelineLayout,
        compute_shader: vk::ShaderModule,
        allocator: &Arc<VulkanAllocator>,
    ) -> vk::Pipeline {
        let comp_shader_stage_create_info = vk::PipelineShaderStageCreateInfo {
            s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
            p_name: c"main".as_ptr(),
            p_next: ptr::null(),
            flags: vk::PipelineShaderStageCreateFlags::empty(),
            stage: vk::ShaderStageFlags::COMPUTE,
            module: compute_shader,
            p_specialization_info: ptr::null(), // for constexpr's in shader
            _marker: PhantomData,
        };

        let shader_stage_create_infos: [vk::PipelineShaderStageCreateInfo; 1] =
            [comp_shader_stage_create_info];

        let create_info = vk::ComputePipelineCreateInfo {
            s_type: vk::StructureType::COMPUTE_PIPELINE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::PipelineCreateFlags::empty(),
            stage: comp_shader_stage_create_info,
            layout: pipeline_layout,
            base_pipeline_handle: vk::Pipeline::null(),
            base_pipeline_index: 0,
            _marker: PhantomData,
        };

        unsafe {
            context
                .device
                .create_compute_pipelines(
                    vk::PipelineCache::null(),
                    slice::from_ref(&create_info),
                    Some(&allocator.callbacks),
                )
                .unwrap()[0]
        }
    }

    pub fn destroy_pipeline(
        context: &VulkanContext,
        pipeline: vk::Pipeline,
        allocator: &Arc<VulkanAllocator>,
    ) {
        unsafe {
            context
                .device
                .destroy_pipeline(pipeline, Some(&allocator.callbacks));
        }
    }
}
