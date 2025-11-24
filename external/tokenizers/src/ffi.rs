use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use tokenizers::Tokenizer;

#[no_mangle]
pub extern "C" fn tokenizer_load(path: *const c_char) -> *mut Tokenizer {
    let c_str = unsafe { CStr::from_ptr(path) };
    let path_str = c_str.to_str().unwrap();

    let tok = Tokenizer::from_file(path_str).unwrap();
    Box::into_raw(Box::new(tok))
}

#[no_mangle]
pub extern "C" fn tokenizer_encode(handle: *mut Tokenizer, text: *const c_char) -> *mut c_char {
    let tok = unsafe { &mut *handle };
    let c_str = unsafe { CStr::from_ptr(text) };
    let text_str = c_str.to_str().unwrap();

    let enc = tok.encode(text_str, true).unwrap();
    let ids: Vec<String> = enc.get_ids().iter().map(|id| id.to_string()).collect();
    let result = ids.join(",");

    CString::new(result).unwrap().into_raw()
}

#[no_mangle]
pub extern "C" fn tokenizer_free_string(s: *mut c_char) {
    if s.is_null() { return; }
    unsafe { drop(CString::from_raw(s)) };
}

#[no_mangle]
pub extern "C" fn tokenizer_destroy(handle: *mut Tokenizer) {
    if handle.is_null() { return; }
    unsafe { drop(Box::from_raw(handle)) };
}