use ash::vk;

pub struct RenderPass {
    pub render_pass: vk::RenderPass,
    pub color_attachment_index: u32,
    pub depth_attachment_index: u32,
}
