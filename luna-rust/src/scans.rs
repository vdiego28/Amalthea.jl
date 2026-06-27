use super::io::{Hdf5Writer, get_hdf5_api};

#[cfg(unix)]
use std::fs::File;
#[cfg(unix)]
use std::os::unix::io::AsRawFd;

pub struct FlockLock {
    #[cfg(unix)]
    file: File,
}

impl FlockLock {
    pub fn new(lock_path: &str) -> Result<Self, String> {
        #[cfg(unix)]
        {
            let file = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .open(lock_path)
                .map_err(|e| format!("Failed to open lock file: {}", e))?;
            Ok(Self { file })
        }
        #[cfg(not(unix))]
        {
            let _ = lock_path;
            Ok(Self {})
        }
    }

    pub fn lock(&self) -> Result<(), String> {
        #[cfg(unix)]
        unsafe {
            let fd = self.file.as_raw_fd();
            let res = libc::flock(fd, libc::LOCK_EX);
            if res != 0 {
                return Err("Failed to acquire flock".to_string());
            }
        }
        Ok(())
    }

    pub fn unlock(&self) -> Result<(), String> {
        #[cfg(unix)]
        unsafe {
            let fd = self.file.as_raw_fd();
            let res = libc::flock(fd, libc::LOCK_UN);
            if res != 0 {
                return Err("Failed to release flock".to_string());
            }
        }
        Ok(())
    }
}

pub struct ScanQueue {
    qfile: String,
    lock_path: String,
    total_points: usize,
}

impl ScanQueue {
    pub fn new(qfile: &str, total_points: usize) -> Self {
        let lock_path = format!("{}_lock", qfile);
        Self {
            qfile: qfile.to_string(),
            lock_path,
            total_points,
        }
    }

    pub fn checkout_next_index(&self) -> Result<Option<usize>, String> {
        let lock = FlockLock::new(&self.lock_path)?;
        lock.lock()?;
        
        let result = (|| -> Result<Option<usize>, String> {
            let writer = Hdf5Writer::open_existing(&self.qfile)?;
            let mut qdata = vec![0; self.total_points];

            let found_idx = writer.with_existing_int_dataset_2d("qdata", &mut qdata, |writer, dset_id, qdata| {
                writer.read_dataset_int(dset_id, qdata)?;
                Ok(qdata.iter().position(|&x| x == 0))
            })?;

            if let Some(idx) = found_idx {
                qdata[idx] = 1;
                writer.with_existing_int_dataset_2d("qdata", &mut qdata, |writer, dset_id, qdata| {
                    writer.write_dataset_int(dset_id, qdata)
                })?;
                Ok(Some(idx))
            } else {
                Ok(None)
            }
        })();
        
        let _ = lock.unlock();
        result
    }

    pub fn mark_completed(&self, idx: usize, success: bool) -> Result<(), String> {
        let lock = FlockLock::new(&self.lock_path)?;
        lock.lock()?;
        
        let result = (|| -> Result<(), String> {
            if !std::path::Path::new(&self.qfile).exists() {
                return Ok(());
            }
            let writer = Hdf5Writer::open_existing(&self.qfile)?;
            let mut qdata = vec![0; self.total_points];

            writer.with_existing_int_dataset_2d("qdata", &mut qdata, |writer, dset_id, qdata| {
                writer.read_dataset_int(dset_id, qdata)?;

                if idx < qdata.len() {
                    qdata[idx] = if success { 2 } else { 3 };
                    writer.write_dataset_int(dset_id, qdata)?;
                }

                Ok(())
            })?;

            Ok(())
        })();
        
        let _ = lock.unlock();
        result
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn init_scan_queue(qfile_ptr: *const libc::c_char, total_points: usize) -> *mut ScanQueue {
    if qfile_ptr.is_null() {
        return std::ptr::null_mut();
    }
    let qfile_cstr = unsafe { std::ffi::CStr::from_ptr(qfile_ptr) };
    let qfile = qfile_cstr.to_string_lossy().into_owned();
    if let Ok(writer) = Hdf5Writer::open_or_create(&qfile) {
        if let Ok(api) = get_hdf5_api() {
            let dims = [total_points as u64];
            let maxdims = [total_points as u64];
            if let Ok(dset_id) = writer.create_dataset_2d(writer.file_id, "qdata", api.h5t_native_int, &dims, &maxdims) {
                let qdata = vec![0; total_points];
                let _ = writer.write_dataset_int(dset_id, &qdata);
                writer.close_dataset(dset_id);
            }
        }
    }
    let queue = Box::new(ScanQueue::new(&qfile, total_points));
    Box::into_raw(queue)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn free_scan_queue(queue: *mut ScanQueue) {
    if !queue.is_null() {
        unsafe {
            let _ = Box::from_raw(queue);
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn checkout_next_index(queue: *mut ScanQueue) -> isize {
    if queue.is_null() {
        return -1;
    }
    let queue_ref = unsafe { &*queue };
    match queue_ref.checkout_next_index() {
        Ok(Some(idx)) => idx as isize,
        Ok(None) => -1,
        Err(err) => {
            eprintln!("Error in checkout_next_index: {}", err);
            -2
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mark_completed(queue: *mut ScanQueue, idx: usize, success: i32) -> i32 {
    if queue.is_null() {
        return -1;
    }
    let queue_ref = unsafe { &*queue };
    match queue_ref.mark_completed(idx, success != 0) {
        Ok(()) => 0,
        Err(err) => {
            eprintln!("Error in mark_completed: {}", err);
            -2
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flock_lock_new() {
        let mut temp_path = std::env::temp_dir();
        temp_path.push(format!("luna_test_flock_lock_new_{}", std::process::id()));
        let lock_path = temp_path.to_str().unwrap();

        // Ensure the file does not exist before we start
        if temp_path.exists() {
            let _ = std::fs::remove_file(&temp_path);
        }

        // Test creating the lock file
        let lock = FlockLock::new(lock_path);
        assert!(lock.is_ok(), "FlockLock::new should succeed");

        #[cfg(unix)]
        {
            assert!(temp_path.exists(), "FlockLock::new should create the file on Unix");
        }

        // Test opening an existing lock file
        let lock2 = FlockLock::new(lock_path);
        assert!(lock2.is_ok(), "FlockLock::new should succeed even if file already exists");

        // Clean up
        if temp_path.exists() {
            let _ = std::fs::remove_file(&temp_path);
        }
    }

    #[test]
    #[cfg(unix)]
    fn test_flock_lock_new_error() {
        let mut temp_path = std::env::temp_dir();
        temp_path.push(format!("luna_test_nonexistent_dir_{}", std::process::id()));
        temp_path.push("lock_file");
        let lock_path = temp_path.to_str().unwrap();

        let lock = FlockLock::new(lock_path);
        assert!(lock.is_err(), "FlockLock::new should fail when directory does not exist");
        if let Err(msg) = lock {
            assert!(msg.starts_with("Failed to open lock file:"), "Unexpected error message: {}", msg);
        }
    }
}
