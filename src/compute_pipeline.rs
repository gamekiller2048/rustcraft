use ash::vk;
use std::{cell::RefCell, ptr, rc::Rc};

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
        allocator: &Rc<RefCell<VulkanAllocator>>,
    ) -> Self {
        let pipeline = Self::create_pipeline(context, pipeline_layout, compute_shader);

        Self {
            pipeline,
            compute_shader,
        }
    }

    pub fn destroy(&self, context: &VulkanContext, allocator: &Rc<RefCell<VulkanAllocator>>) {
        ComputePipeline::destroy_pipeline(context, self.pipeline, allocator);
        ShaderModule::destroy_shader_module(context, self.compute_shader, allocator);
    }

    pub fn create_pipeline(
        context: &VulkanContext,
        pipeline_layout: vk::PipelineLayout,
        compute_shader: vk::ShaderModule,
    ) -> vk::Pipeline {
        let comp_shader_stage_create_info = vk::PipelineShaderStageCreateInfo {
            s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
            p_name: c"main".as_ptr(),
            p_next: ptr::null(),
            flags: vk::PipelineShaderStageCreateFlags::empty(),
            stage: vk::ShaderStageFlags::COMPUTE,
            module: compute_shader,
            p_specialization_info: ptr::null(), // for constexpr's in shader
            ..Default::default()
        };

        let shader_stage_create_infos: [vk::PipelineShaderStageCreateInfo; 1] =
            [comp_shader_stage_create_info];

        // pipelines[0]
        vk::Pipeline::null()
    }

    pub fn destroy_pipeline(
        context: &VulkanContext,
        pipeline: vk::Pipeline,
        allocator: &Rc<RefCell<VulkanAllocator>>,
    ) {
        unsafe {
            context.device.destroy_pipeline(
                pipeline,
                Some(&allocator.borrow_mut().get_allocation_callbacks()),
            );
        }
    }
}
