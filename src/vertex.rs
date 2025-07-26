use ash::vk;

pub trait Vertex {
    fn get_binding_description() -> vk::VertexInputBindingDescription;
    fn get_attribute_description() -> Vec<vk::VertexInputAttributeDescription>;
}