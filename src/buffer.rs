use ash::vk;
use std::ffi::c_void;

pub struct Buffer {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub mapped_memory: *mut c_void,
}


