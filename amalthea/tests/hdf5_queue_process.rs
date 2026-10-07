//! Exercise native queue initialization and claims in independent HDF5 processes.
use amalthea::io::{Hdf5Writer, get_hdf5_api};
use amalthea::scans::{self, ScanQueue};
use std::ffi::CString;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const WORKERS: usize = 4;
const POINTS: usize = 64;
const TIMEOUT: Duration = Duration::from_secs(60);
const WORKER_DIRECTORY: &str = "AMALTHEA_HDF5_QUEUE_TEST_DIRECTORY";
const WORKER_INDEX: &str = "AMALTHEA_HDF5_QUEUE_TEST_WORKER";

struct Directory(PathBuf);

impl Directory {
    fn new() -> Self {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "amalthea-queue-processes-{}-{stamp}",
            std::process::id()
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
}

impl Drop for Directory {
    fn drop(&mut self) {
        if std::thread::panicking() {
            eprintln!(
                "Queue process test diagnostics retained at {}",
                self.0.display()
            );
        } else {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
}

struct Queue(*mut ScanQueue);

impl Queue {
    fn open(path: &Path) -> Self {
        let path = CString::new(path.to_str().unwrap()).unwrap();
        let pointer = unsafe { scans::init_scan_queue(path.as_ptr(), POINTS) };
        assert!(!pointer.is_null(), "Native queue initialization failed");
        Self(pointer)
    }

    fn claim(&self) -> Option<usize> {
        let index = unsafe { scans::checkout_next_index(self.0) };
        assert!(index >= -1, "Native queue checkout failed: {index}");
        (index >= 0).then_some(index as usize)
    }

    fn complete(&self, index: usize) {
        let status = unsafe { scans::mark_completed(self.0, index, i32::from(index % 2 == 0)) };
        assert_eq!(status, 0, "Native queue completion failed");
    }
}

impl Drop for Queue {
    fn drop(&mut self) {
        unsafe { scans::free_scan_queue(self.0) };
    }
}

struct Workers(Vec<(Child, PathBuf)>);

impl Workers {
    fn spawn(directory: &Path, names: impl IntoIterator<Item = String>) -> Self {
        let mut workers = Self(Vec::new());
        for worker in names {
            let log_path = directory.join(format!("worker-{worker}.log"));
            let log = std::fs::File::create(&log_path).unwrap();
            let child = Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "queue_worker", "--ignored", "--nocapture"])
                .env(WORKER_DIRECTORY, directory)
                .env(WORKER_INDEX, worker.to_string())
                .stdout(Stdio::from(log.try_clone().unwrap()))
                .stderr(Stdio::from(log))
                .spawn()
                .unwrap();
            workers.0.push((child, log_path));
        }
        workers
    }

    fn wait_for_markers(&mut self, directory: &Path, prefix: &str, deadline: Instant) {
        while !(0..WORKERS).all(|worker| directory.join(format!("{prefix}-{worker}")).is_file()) {
            for (child, log) in &mut self.0 {
                if let Some(status) = child.try_wait().unwrap() {
                    panic!(
                        "Queue worker exited before {prefix}: {status}\n{}",
                        std::fs::read_to_string(log).unwrap()
                    );
                }
            }
            assert!(Instant::now() < deadline, "Timed out waiting for {prefix}");
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    fn finish(&mut self, deadline: Instant) {
        loop {
            let mut finished = 0;
            for (child, log) in &mut self.0 {
                if let Some(status) = child.try_wait().unwrap() {
                    assert!(
                        status.success(),
                        "Queue worker failed: {status}\n{}",
                        std::fs::read_to_string(log).unwrap()
                    );
                    finished += 1;
                }
            }
            if finished == self.0.len() {
                return;
            }
            assert!(Instant::now() < deadline, "Queue workers did not terminate");
            std::thread::sleep(Duration::from_millis(10));
        }
    }
}

impl Drop for Workers {
    fn drop(&mut self) {
        // A failed assertion or lock regression must not leave hung test children.
        for (child, _) in &mut self.0 {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

fn signal(directory: &Path, name: &str, contents: &str) {
    let temporary = directory.join(format!("{name}.tmp"));
    std::fs::write(&temporary, contents).unwrap();
    std::fs::rename(temporary, directory.join(name)).unwrap();
}

fn read_queue(path: &Path) -> Vec<i32> {
    let reader = Hdf5Writer::open_existing(path.to_str().unwrap()).unwrap();
    let mut states = vec![0; POINTS];
    reader
        .with_existing_int_dataset_2d("qdata", &mut states, |reader, dataset, states| {
            reader.read_dataset_int(dataset, states)
        })
        .unwrap();
    states
}

#[test]
fn test_hdf5_queue_processes() {
    if let Err(error) = get_hdf5_api() {
        assert!(
            !std::env::var("AMALTHEA_REQUIRE_HDF5_TESTS").is_ok_and(|value| value == "1"),
            "Native queue process test requires HDF5: {error}"
        );
        println!("Skipping native queue process test: HDF5 unavailable: {error}");
        return;
    }

    for resume in [false, true] {
        let directory = Directory::new();
        let queue_path = directory.0.join("queue.h5");
        let mut expected = vec![0; POINTS];
        if resume {
            // Preserve successful, failed, and already claimed work on reopen.
            expected[..3].copy_from_slice(&[2, 3, 1]);
            let api = get_hdf5_api().unwrap();
            let writer = Hdf5Writer::open_or_create(queue_path.to_str().unwrap()).unwrap();
            let dataset = writer
                .create_dataset_2d(
                    writer.file_id,
                    "qdata",
                    api.h5t_native_int,
                    &[POINTS as u64],
                    &[POINTS as u64],
                )
                .unwrap();
            writer.write_dataset_int(dataset, &expected).unwrap();
            writer.close_dataset(dataset);
        }

        // Release every process into initialization from the same barrier.
        // All queue calls stay in supervised children so a lock regression
        // cannot block this process before it enforces the deadline.
        let deadline = Instant::now() + TIMEOUT;
        let mut workers =
            Workers::spawn(&directory.0, (0..WORKERS).map(|worker| worker.to_string()));
        workers.wait_for_markers(&directory.0, "ready", deadline);
        signal(&directory.0, "start", "");
        workers.wait_for_markers(&directory.0, "claimed", deadline);

        let available: Vec<_> = (0..POINTS).filter(|&index| expected[index] == 0).collect();
        for worker in 0..WORKERS {
            let index: usize =
                std::fs::read_to_string(directory.0.join(format!("claimed-{worker}")))
                    .unwrap()
                    .parse()
                    .unwrap();
            assert!(index < POINTS, "Worker returned an invalid index");
            assert_eq!(
                expected[index], 0,
                "Work was claimed twice or reclaimed on resume"
            );
            expected[index] = 1;
        }
        assert_eq!(read_queue(&queue_path), expected);
        // A separate process reinitializes while every worker has a claim.
        Workers::spawn(&directory.0, ["reopen".to_string()]).finish(deadline);
        assert_eq!(
            read_queue(&queue_path),
            expected,
            "Reinitialization lost progress"
        );

        signal(&directory.0, "continue", "");
        workers.finish(deadline);
        let mut claims = Vec::new();
        for worker in 0..WORKERS {
            let report =
                std::fs::read_to_string(directory.0.join(format!("result-{worker}"))).unwrap();
            claims.extend(report.lines().map(|line| line.parse::<usize>().unwrap()));
        }
        claims.sort_unstable();
        assert_eq!(
            claims, available,
            "Every available point must be claimed exactly once"
        );
        for index in available {
            expected[index] = if index % 2 == 0 { 2 } else { 3 };
        }
        assert_eq!(
            read_queue(&queue_path),
            expected,
            "Completion states must persist"
        );
        Workers::spawn(&directory.0, ["drained".to_string()]).finish(deadline);
        assert_eq!(read_queue(&queue_path), expected);
    }
}

#[test]
#[ignore = "subprocess helper, invoked only by test_hdf5_queue_processes"]
fn queue_worker() {
    let directory =
        PathBuf::from(std::env::var_os(WORKER_DIRECTORY).expect("Missing worker directory"));
    let worker = std::env::var(WORKER_INDEX).expect("Missing worker index");
    let deadline = Instant::now() + TIMEOUT;
    if worker == "reopen" || worker == "drained" {
        let queue = Queue::open(&directory.join("queue.h5"));
        if worker == "drained" {
            assert_eq!(
                queue.claim(),
                None,
                "Finished or in-progress points were reclaimed"
            );
        }
        return;
    }
    signal(&directory, &format!("ready-{worker}"), "");
    while !directory.join("start").is_file() {
        assert!(
            Instant::now() < deadline,
            "Parent did not start queue workers"
        );
        std::thread::sleep(Duration::from_millis(10));
    }
    let queue = Queue::open(&directory.join("queue.h5"));
    let first = queue
        .claim()
        .expect("Each worker must claim at least one point");
    signal(&directory, &format!("claimed-{worker}"), &first.to_string());
    while !directory.join("continue").is_file() {
        assert!(
            Instant::now() < deadline,
            "Parent did not release queue workers"
        );
        std::thread::sleep(Duration::from_millis(10));
    }
    let mut claims = vec![first];
    queue.complete(first);
    while let Some(index) = queue.claim() {
        claims.push(index);
        // Give other processes a chance to claim while this point is in progress.
        std::thread::yield_now();
        queue.complete(index);
    }
    let report: String = claims.iter().map(|index| format!("{index}\n")).collect();
    signal(&directory, &format!("result-{worker}"), &report);
}
