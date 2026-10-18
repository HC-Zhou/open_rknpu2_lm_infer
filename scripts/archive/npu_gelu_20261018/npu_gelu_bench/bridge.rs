use anyhow::{Result, ensure};
use std::ffi::{CStr, CString, c_char, c_void};

type Create = unsafe extern "C" fn(usize, u32) -> *mut c_void;
type Free = unsafe extern "C" fn(*mut c_void);
type Run = unsafe extern "C" fn(*mut c_void, *const u16, *mut u16, *mut f64) -> i32;
type Guard = unsafe extern "C" fn(*mut c_void) -> i32;
type Error = unsafe extern "C" fn() -> *const c_char;
pub struct Bridge {
    library: *mut c_void,
    count: usize,
    handle: *mut c_void,
    free: Free,
    run: Run,
    guard: Guard,
    error: Error,
}
unsafe fn symbol<T: Copy>(library: *mut c_void, name: &CStr) -> Result<T> {
    let pointer = unsafe { libc::dlsym(library, name.as_ptr()) };
    ensure!(!pointer.is_null(), "missing bridge symbol {name:?}");
    ensure!(
        std::mem::size_of::<T>() == std::mem::size_of_val(&pointer),
        "function pointer size"
    );
    Ok(unsafe { std::mem::transmute_copy(&pointer) })
}
impl Bridge {
    pub fn guard_ok(&self) -> bool {
        unsafe { (self.guard)(self.handle) == 1 }
    }
    pub fn new(path: &std::path::Path, n: usize, mask: u32) -> Result<Self> {
        let path = CString::new(path.canonicalize()?.to_string_lossy().as_bytes())?;
        let library = unsafe { libc::dlopen(path.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) };
        ensure!(!library.is_null(), "load experimental bridge");
        unsafe {
            let create: Create = symbol(library, c"npu_gelu_probe_create")?;
            let free = symbol(library, c"npu_gelu_probe_free")?;
            let run = symbol(library, c"npu_gelu_probe_run")?;
            let guard = symbol(library, c"npu_gelu_probe_guard")?;
            let error: Error = symbol(library, c"npu_gelu_probe_error")?;
            let handle = create(n, mask);
            ensure!(
                !handle.is_null(),
                "{}",
                CStr::from_ptr(error()).to_string_lossy()
            );
            Ok(Self {
                library,
                count: n,
                handle,
                free,
                run,
                guard,
                error,
            })
        }
    }
    pub fn apply(&self, values: &mut [u16], stages: &mut [f64; 6]) -> Result<()> {
        ensure!(values.len() == self.count, "GELU bridge shape mismatch");
        let pointer = values.as_mut_ptr();
        let rc = unsafe { (self.run)(self.handle, pointer, pointer, stages.as_mut_ptr()) };
        ensure!(
            rc == 0,
            "{}",
            unsafe { CStr::from_ptr((self.error)()) }.to_string_lossy()
        );
        Ok(())
    }
}
impl Drop for Bridge {
    fn drop(&mut self) {
        unsafe {
            (self.free)(self.handle);
            libc::dlclose(self.library);
        }
    }
}
