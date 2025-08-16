use std::{
    alloc::{AllocError, Allocator, Global, Layout},
    ptr::NonNull,
    sync::Mutex,
};

use log::error;

pub struct BumpAllocator<T: Allocator = Global> {
    buffer: Mutex<Vec<u8, T>>,
}

impl<T: Allocator> BumpAllocator<T> {
    pub fn new(capacity: usize, allocator: T) -> Self {
        Self {
            buffer: Mutex::new(Vec::with_capacity_in(capacity, allocator)),
        }
    }
}

impl BumpAllocator<Global> {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            buffer: Mutex::new(Vec::with_capacity(capacity)),
        }
    }
}

unsafe impl Allocator for BumpAllocator {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        let mut buffer = self.buffer.lock().unwrap();

        let start = ((buffer.as_ptr() as usize + buffer.len() + layout.align() - 1)
            & !(layout.align() - 1))
            - buffer.as_ptr() as usize;
        let end = start + layout.size();
        
        if end > buffer.capacity() {
            error!(
                "bump allocator not enough space, requested: {} bytes with {} align = {} when {} bytes free",
                layout.size(),
                layout.align(),
                start,
                buffer.capacity() - buffer.len(),
            );
            Err(AllocError)
        } else {
            buffer.resize(end, 0);

            let slice = &mut buffer[start..end];
            Ok(NonNull::from(slice))
        }
    }

    unsafe fn deallocate(&self, _ptr: NonNull<u8>, _layout: Layout) {
        // no deallocation, must deallocate entire arena at once
    }
}
