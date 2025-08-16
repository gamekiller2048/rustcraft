use ash::vk;
use log::{error, trace};
use std::{
    alloc::{Allocator, Layout},
    cmp::min,
    collections::HashMap,
    ffi::c_void,
    marker::PhantomData,
    ptr::{self, NonNull},
    sync::{Arc, Mutex},
};

pub struct VulkanAllocator {
    pub memory_layouts: Mutex<HashMap<*mut c_void, Layout>>,
    pub allocator: Arc<dyn Allocator + Send + Sync>,
    pub callbacks: vk::AllocationCallbacks<'static>,
    pub name: String,
}

impl VulkanAllocator {
    pub fn new(allocator: Arc<dyn Allocator + Send + Sync>, name: &str) -> Arc<Self> {
        Arc::new_cyclic(|weak_self| {
            let callbacks = vk::AllocationCallbacks {
                p_user_data: weak_self.as_ptr() as *mut c_void,
                pfn_allocation: Some(Self::allocate),
                pfn_reallocation: Some(Self::reallocate),
                pfn_free: Some(Self::free),
                pfn_internal_allocation: Some(Self::vulkan_internal_allocation_nofication),
                pfn_internal_free: Some(Self::vulkan_internal_free_nofication),
                _marker: PhantomData,
            };

            VulkanAllocator {
                memory_layouts: Mutex::new(HashMap::new()),
                allocator: allocator.clone(),
                callbacks,
                name: name.to_owned(),
            }
        })
    }

    extern "system" fn allocate(
        p_user_data: *mut c_void,
        size: usize,
        alignment: usize,
        allocation_scope: vk::SystemAllocationScope,
    ) -> *mut c_void {
        if size == 0 {
            return ptr::null_mut();
        }

        let allocator = unsafe { &mut (*(p_user_data as *mut VulkanAllocator)) };

        let layout = Layout::from_size_align(size, alignment).unwrap();
        let result = allocator.allocator.allocate(layout);

        if result.is_err() {
            error!("failed to allocate");
            return ptr::null_mut();
        }

        let p_memory = result.unwrap().as_ptr() as *mut c_void;
        allocator
            .memory_layouts
            .lock()
            .unwrap()
            .insert(p_memory, layout);

        trace!("alloc {:?} size={}, align={}", p_memory, size, alignment);
        p_memory
    }

    extern "system" fn reallocate(
        p_user_data: *mut c_void,
        p_original: *mut c_void,
        size: usize,
        alignment: usize,
        allocation_scope: vk::SystemAllocationScope,
    ) -> *mut c_void {
        if p_original.is_null() {
            return Self::allocate(p_user_data, size, alignment, allocation_scope);
        }

        if size == 0 {
            Self::free(p_user_data, p_original);
            return ptr::null_mut();
        }

        let allocator = unsafe { &mut (*(p_user_data as *mut VulkanAllocator)) };
        let result = allocator
            .memory_layouts
            .lock()
            .unwrap()
            .get(&p_original)
            .cloned();

        if result.is_none() {
            error!("failed to find memory block {:?} for realloc", p_original);
            return ptr::null_mut();
        }

        let original_layout = result.unwrap();

        if alignment != original_layout.align() {
            error!(
                "attempted realloc {:?} using different alignment. requested={} original={}",
                p_original,
                alignment,
                original_layout.align()
            );
            return ptr::null_mut();
        }

        trace!("doing alloc for realloc original={:?}", p_original);
        let p_memory = Self::allocate(p_user_data, size, alignment, allocation_scope);

        if p_memory.is_null() {
            error!("failed to realloc {:?}", p_original);
            return ptr::null_mut();
        }

        unsafe {
            ptr::copy_nonoverlapping(p_original, p_memory, min(original_layout.size(), size));
            Self::free(p_user_data, p_original);
        }

        p_memory
    }

    extern "system" fn free(p_user_data: *mut c_void, p_memory: *mut c_void) {
        if p_memory.is_null() {
            return;
        }

        let allocator = unsafe { &mut *(p_user_data as *mut VulkanAllocator) };
        let result = allocator
            .memory_layouts
            .lock()
            .unwrap()
            .get(&p_memory)
            .cloned();

        if result.is_none() {
            error!("failed to find memory block {:?} for free", p_memory);
            return;
        }

        let layout = result.unwrap().clone();

        unsafe {
            allocator
                .allocator
                .deallocate(NonNull::new(p_memory as *mut u8).unwrap(), layout);
        }

        allocator.memory_layouts.lock().unwrap().remove(&p_memory);

        trace!(
            "free {:?} size={}, align={}",
            p_memory,
            layout.size(),
            layout.align()
        );
    }

    extern "system" fn vulkan_internal_allocation_nofication(
        p_user_data: *mut c_void,
        size: usize,
        allocation_type: vk::InternalAllocationType,
        allocation_scope: vk::SystemAllocationScope,
    ) {
        trace!("internal vulkan alloc");
    }

    extern "system" fn vulkan_internal_free_nofication(
        p_user_data: *mut c_void,
        size: usize,
        allocation_type: vk::InternalAllocationType,
        allocation_scope: vk::SystemAllocationScope,
    ) {
        trace!("internal vulkan free");
    }
}
