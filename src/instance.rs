use std::ffi::{c_char, CString};
use std::ptr;
use ash::{khr, vk};

#[cfg(feature="validation")]
use std::ffi::{c_void, CStr};
#[cfg(feature="validation")]
use ash::ext;
#[cfg(feature="validation")]
use log::warn;


#[cfg(feature="validation")]
pub const REQUIRED_VALIDATION_LAYER_NAMES: [&'static str; 1] = ["VK_LAYER_KHRONOS_validation"];

#[cfg(feature="validation")]
unsafe extern "system" fn vulkan_debug_utils_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _p_user_data: *mut c_void,
) -> vk::Bool32 {
    let severity = match message_severity {
        vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => "[Verbose]",
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => "[Warning]",
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => "[Error]",
        vk::DebugUtilsMessageSeverityFlagsEXT::INFO => "[Info]",
        _ => "[Unknown]",
    };

    let types: &'static str = match message_type {
        vk::DebugUtilsMessageTypeFlagsEXT::GENERAL => "[General]",
        vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE => "[Performance]",
        vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION => "[Validation]",
        _ => "[Unknown]",
    };

    let message: &CStr = unsafe {
        CStr::from_ptr((*p_callback_data).p_message)
    };

    println!("[Debug]{}{}{:?}", severity, types, message);

    vk::FALSE
}

#[cfg(feature="validation")]
fn populate_debug_messenger_create_info(create_info: &mut vk::DebugUtilsMessengerCreateInfoEXT<'_>) {
    create_info.s_type = vk::StructureType::DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    create_info.p_next = ptr::null();

    create_info.flags = vk::DebugUtilsMessengerCreateFlagsEXT::empty();
    create_info.message_severity = 
            vk::DebugUtilsMessageSeverityFlagsEXT::WARNING |
            // vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE |
            // vk::DebugUtilsMessageSeverityFlagsEXT::INFO |
            vk::DebugUtilsMessageSeverityFlagsEXT::ERROR;

    create_info.message_type =
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL |
            vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE |
            vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION;

    create_info.pfn_user_callback = Some(vulkan_debug_utils_callback);
    create_info.p_user_data = ptr::null_mut();
}

#[cfg(feature="validation")]
pub fn create_debug_messenger(debug_utils_loader: &ext::debug_utils::Instance) -> vk::DebugUtilsMessengerEXT {
    if !cfg!(feature="validation") {
        return vk::DebugUtilsMessengerEXT::null();
    }

    let mut debug_create_info = vk::DebugUtilsMessengerCreateInfoEXT{..Default::default()};
    populate_debug_messenger_create_info(&mut debug_create_info);

    unsafe {
        debug_utils_loader
            .create_debug_utils_messenger(&debug_create_info, None)
            .unwrap()
    }
}

#[cfg(feature="validation")]
pub fn check_validation_layer_support(entry: &ash::Entry) -> bool {
    let supported_layers: Vec<vk::LayerProperties> = unsafe {
        entry.enumerate_instance_layer_properties().unwrap()
    };

    if supported_layers.len() <= 0 {
        warn!("no supported layers found");
        return false;
    }
    
    for required_layer_name in REQUIRED_VALIDATION_LAYER_NAMES.iter() {
        let mut is_layer_found = false;

        for layer in supported_layers.iter() {
            let layer_name: &str = unsafe {
                CStr::from_ptr(layer.layer_name.as_ptr()).to_str().unwrap()
            };
            
            if layer_name == *required_layer_name {
                is_layer_found = true;
                break;
            }
        }

        if is_layer_found == false {
            return false;
        }
    }

    true
}

pub fn create_instance(entry: &ash::Entry) -> ash::Instance {
    let application_name: CString = CString::new("app name").unwrap();
    let engine_name: CString = CString::new("engine name").unwrap();

    let app_info = vk::ApplicationInfo {
        s_type: vk::StructureType::APPLICATION_INFO,
        p_next: ptr::null(),
        p_application_name: application_name.as_ptr(),
        application_version: vk::make_api_version(0, 1, 0, 0),
        p_engine_name: engine_name.as_ptr(),
        engine_version: 0,
        api_version: vk::API_VERSION_1_0,
        ..Default::default()
    };

    #[cfg(target_os = "windows")]
    #[allow(unused_mut)]
    let mut extension_names: Vec<*const c_char> = vec![
        khr::surface::NAME.as_ptr(),
        khr::win32_surface::NAME.as_ptr(),
    ];

    #[cfg(feature="validation")]
    extension_names.push(ext::debug_utils::NAME.as_ptr());

    #[allow(unused_mut)]
    let mut create_info = vk::InstanceCreateInfo {
        s_type: vk::StructureType::INSTANCE_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::InstanceCreateFlags::empty(),
        p_application_info: &app_info,
        pp_enabled_extension_names: extension_names.as_ptr(),
        pp_enabled_layer_names: ptr::null(),
        enabled_extension_count: extension_names.len() as u32,
        enabled_layer_count: 0,
        ..Default::default()
    };

    let cstr_required_validation_layer_names: Vec<CString>;
    let p_required_validation_layer_names: Vec<*const i8>;
    
    #[allow(unused_mut)]
    let mut debug_create_info = vk::DebugUtilsMessengerCreateInfoEXT{..Default::default()};

    #[cfg(feature="validation")]
    {
        cstr_required_validation_layer_names = REQUIRED_VALIDATION_LAYER_NAMES
            .iter()
            .map(|layer_name: &&str| CString::new(*layer_name).unwrap())
            .collect();

        p_required_validation_layer_names= cstr_required_validation_layer_names
            .iter()
            .map(|layer_name: &CString| layer_name.as_ptr())
            .collect();

        populate_debug_messenger_create_info(&mut debug_create_info);
        create_info.p_next = &debug_create_info as *const vk::DebugUtilsMessengerCreateInfoEXT as *const c_void;
        create_info.pp_enabled_layer_names = p_required_validation_layer_names.as_ptr();
        create_info.enabled_layer_count = p_required_validation_layer_names.len() as u32;
    }

    unsafe {
        entry.create_instance(&create_info, None).unwrap()
    }
}