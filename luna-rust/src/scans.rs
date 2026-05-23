use std::fs::File;
use std::os::unix::io::AsRawFd;
use std::path::Path;
use std::ffi::CStr;
use super::io::{Hdf5Writer, get_hdf5_api};

pub struct FlockLock {
    file: File,
}

impl FlockLock {
    pub fn new(lock_path: &str) -> Result<Self, String> {
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(lock_path)
            .map_err(|e| format!("Failed to open lock file: {}", e))?;
        Ok(Self { file })
    }

    pub fn lock(&self) -> Result<(), String> {
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
            let writer = Hdf5Writer::open_or_create(&self.qfile)?;
            let api = get_hdf5_api()?;
            
            let dims = [self.total_points as u64];
            let maxdims = [self.total_points as u64];
            let dset_id = writer.create_dataset_2d(writer.file_id, "qdata", api.h5t_native_int, &dims, &maxdims)?;
            
            let mut qdata = vec![0; self.total_points];
            writer.read_dataset_int(dset_id, &mut qdata)?;
            
            let found_idx = qdata.iter().position(|&x| x == 0);
            
            if let Some(idx) = found_idx {
                qdata[idx] = 1;
                writer.write_dataset_int(dset_id, &qdata)?;
                writer.close_dataset(dset_id);
                Ok(Some(idx))
            } else {
                writer.close_dataset(dset_id);
                
                let all_finished = qdata.iter().all(|&x| x > 1);
                if all_finished {
                    let _ = std::fs::remove_file(&self.qfile);
                    let _ = std::fs::remove_file(&self.lock_path);
                }
                
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
            if !Path::new(&self.qfile).exists() {
                return Ok(());
            }
            let writer = Hdf5Writer::open_or_create(&self.qfile)?;
            let api = get_hdf5_api()?;
            
            let dims = [self.total_points as u64];
            let maxdims = [self.total_points as u64];
            let dset_id = writer.create_dataset_2d(writer.file_id, "qdata", api.h5t_native_int, &dims, &maxdims)?;
            
            let mut qdata = vec![0; self.total_points];
            writer.read_dataset_int(dset_id, &mut qdata)?;
            
            if idx < qdata.len() {
                qdata[idx] = if success { 2 } else { 3 };
                writer.write_dataset_int(dset_id, &qdata)?;
            }
            
            writer.close_dataset(dset_id);
            
            let all_finished = qdata.iter().all(|&x| x > 1);
            if all_finished {
                let _ = std::fs::remove_file(&self.qfile);
                let _ = std::fs::remove_file(&self.lock_path);
            }
            
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
    let qfile_cstr = unsafe { CStr::from_ptr(qfile_ptr) };
    let qfile = qfile_cstr.to_string_lossy().into_owned();
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
