use ash::vk;
use nalgebra_glm as glm;

use crate::vertex::Vertex;

#[repr(C)]
pub struct MyVertex {
    pos: glm::Vec3,
    tex_uv: glm::Vec2,
}

impl Vertex for MyVertex {
    fn get_binding_description() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription {
            binding: 0,
            stride: size_of::<MyVertex>() as u32,
            input_rate: vk::VertexInputRate::VERTEX,
        }
    }

    fn get_attribute_description() -> Vec<vk::VertexInputAttributeDescription> {
        let attribute_description_pos = vk::VertexInputAttributeDescription {
            location: 0,
            binding: 0,
            format: vk::Format::R32G32B32_SFLOAT,
            offset: 0,
        };

        let attribute_description_color = vk::VertexInputAttributeDescription {
            location: 1,
            binding: 0,
            format: vk::Format::R32G32_SFLOAT,
            offset: 3 * size_of::<f32>() as u32,
        };

        vec![attribute_description_pos, attribute_description_color]
    }
}

pub const VERTICES: [MyVertex; 24] = [
    // Front face
    MyVertex {
        pos: glm::Vec3::new(-1.0, -1.0, 1.0),
        tex_uv: glm::Vec2::new(0.0, 0.0),
    },
    MyVertex {
        pos: glm::Vec3::new(1.0, -1.0, 1.0),
        tex_uv: glm::Vec2::new(1.0, 0.0),
    },
    MyVertex {
        pos: glm::Vec3::new(1.0, 1.0, 1.0),
        tex_uv: glm::Vec2::new(1.0, 1.0),
    },
    MyVertex {
        pos: glm::Vec3::new(-1.0, 1.0, 1.0),
        tex_uv: glm::Vec2::new(0.0, 1.0),
    },
    // Back face
    MyVertex {
        pos: glm::Vec3::new(-1.0, -1.0, -1.0),
        tex_uv: glm::Vec2::new(0.0, 0.0),
    },
    MyVertex {
        pos: glm::Vec3::new(1.0, -1.0, -1.0),
        tex_uv: glm::Vec2::new(1.0, 0.0),
    },
    MyVertex {
        pos: glm::Vec3::new(1.0, 1.0, -1.0),
        tex_uv: glm::Vec2::new(1.0, 1.0),
    },
    MyVertex {
        pos: glm::Vec3::new(-1.0, 1.0, -1.0),
        tex_uv: glm::Vec2::new(0.0, 1.0),
    },
    // Left face
    MyVertex {
        pos: glm::Vec3::new(-1.0, -1.0, -1.0),
        tex_uv: glm::Vec2::new(0.0, 0.0),
    },
    MyVertex {
        pos: glm::Vec3::new(-1.0, -1.0, 1.0),
        tex_uv: glm::Vec2::new(1.0, 0.0),
    },
    MyVertex {
        pos: glm::Vec3::new(-1.0, 1.0, 1.0),
        tex_uv: glm::Vec2::new(1.0, 1.0),
    },
    MyVertex {
        pos: glm::Vec3::new(-1.0, 1.0, -1.0),
        tex_uv: glm::Vec2::new(0.0, 1.0),
    },
    // Right face
    MyVertex {
        pos: glm::Vec3::new(1.0, -1.0, -1.0),
        tex_uv: glm::Vec2::new(0.0, 0.0),
    },
    MyVertex {
        pos: glm::Vec3::new(1.0, -1.0, 1.0),
        tex_uv: glm::Vec2::new(1.0, 0.0),
    },
    MyVertex {
        pos: glm::Vec3::new(1.0, 1.0, 1.0),
        tex_uv: glm::Vec2::new(1.0, 1.0),
    },
    MyVertex {
        pos: glm::Vec3::new(1.0, 1.0, -1.0),
        tex_uv: glm::Vec2::new(0.0, 1.0),
    },
    // Top face
    MyVertex {
        pos: glm::Vec3::new(-1.0, 1.0, -1.0),
        tex_uv: glm::Vec2::new(0.0, 0.0),
    },
    MyVertex {
        pos: glm::Vec3::new(-1.0, 1.0, 1.0),
        tex_uv: glm::Vec2::new(1.0, 0.0),
    },
    MyVertex {
        pos: glm::Vec3::new(1.0, 1.0, 1.0),
        tex_uv: glm::Vec2::new(1.0, 1.0),
    },
    MyVertex {
        pos: glm::Vec3::new(1.0, 1.0, -1.0),
        tex_uv: glm::Vec2::new(0.0, 1.0),
    },
    // Bottom face
    MyVertex {
        pos: glm::Vec3::new(-1.0, -1.0, -1.0),
        tex_uv: glm::Vec2::new(0.0, 0.0),
    },
    MyVertex {
        pos: glm::Vec3::new(-1.0, -1.0, 1.0),
        tex_uv: glm::Vec2::new(1.0, 0.0),
    },
    MyVertex {
        pos: glm::Vec3::new(1.0, -1.0, 1.0),
        tex_uv: glm::Vec2::new(1.0, 1.0),
    },
    MyVertex {
        pos: glm::Vec3::new(1.0, -1.0, -1.0),
        tex_uv: glm::Vec2::new(0.0, 1.0),
    },
];

pub const INDICES: [u16; 36] = [
    // Front face
    0, 1, 2, 2, 3, 0, // Back face
    4, 5, 6, 6, 7, 4, // Left face
    8, 9, 10, 10, 11, 8, // Right face
    12, 13, 14, 14, 15, 12, // Top face
    16, 17, 18, 18, 19, 16, // Bottom face
    20, 21, 22, 22, 23, 20,
];

// if graphics_queue.family_index != transfer_queue.family_index {
//     let vertex_buffer_memory_barrier = vk::BufferMemoryBarrier {
//         s_type: vk::StructureType::BUFFER_MEMORY_BARRIER,
//         p_next: ptr::null(),
//         src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
//         dst_access_mask: vk::AccessFlags::VERTEX_ATTRIBUTE_READ,
//         src_queue_family_index: transfer_queue.family_index,
//         dst_queue_family_index: graphics_queue.family_index,
//         buffer: vertex_buffer.buffer,
//         offset: 0,
//         size: vk::WHOLE_SIZE,
//         ..Default::default()
//     };

//     let index_buffer_memory_barrier = vk::BufferMemoryBarrier {
//         s_type: vk::StructureType::BUFFER_MEMORY_BARRIER,
//         p_next: ptr::null(),
//         src_access_mask: vk::AccessFlags::Transf,
//         dst_access_mask: vk::AccessFlags::VERTEX_ATTRIBUTE_READ,
//         src_queue_family_index: transfer_queue.family_index,
//         dst_queue_family_index: graphics_queue.family_index,
//         buffer: index_buffer.buffer,
//         offset: 0,
//         size: vk::WHOLE_SIZE,
//         ..Default::default()
//     };

//     let buffer_memory_barriers: [vk::BufferMemoryBarrier; 2] =
//         [vertex_buffer_memory_barrier, index_buffer_memory_barrier];

//     let texture_image_memory_barrier = vk::ImageMemoryBarrier {
//         s_type: vk::StructureType::BUFFER_MEMORY_BARRIER,
//         p_next: ptr::null(),
//         src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
//         dst_access_mask: vk::AccessFlags::VERTEX_ATTRIBUTE_READ,
//         src_queue_family_index: transfer_queue.family_index,
//         dst_queue_family_index: graphics_queue.family_index,
//         old_layout: vk::
//         ..Default::default()
//     };

//     transfer_command_buffer.pipeline_barrier(
//         &context,
//         vk::PipelineStageFlags::TRANSFER,
//         vk::PipelineStageFlags::BOTTOM_OF_PIPE,
//         &[],
//         &buffer_memory_barriers,
//         &[],
//     );
// }
