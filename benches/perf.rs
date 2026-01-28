use gpusorted_map::{Capacity, GpuSortedMap, Key, KvEntry, Value};
use rand::rngs::StdRng;
use rand::{RngCore, SeedableRng};
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::Write;
use std::path::PathBuf;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

fn main() {
    let sizes = [1_usize, 10, 100, 1_000, 10_000, 100_000, 1_000_000];
    let mut rng = StdRng::seed_from_u64(0x5eed);

    let mut rows = Vec::new();
    for size in sizes {
        let keys_raw: Vec<u32> = (0..size).map(|_| rng.next_u32()).collect();
        let keys: Vec<Key> = keys_raw.iter().copied().map(Key::new).collect();
        let entries: Vec<KvEntry> = keys_raw
            .iter()
            .map(|&key| KvEntry {
                key: Key::new(key),
                value: Value::new(key.wrapping_mul(31)),
            })
            .collect();

        let (btree_put, btree_get) = bench_btree(&keys_raw);
        let (gpu_put, gpu_get) = bench_gpu(&entries, &keys);

        rows.push((size, btree_put, btree_get, gpu_put, gpu_get));
    }

    let commit = git_head().unwrap_or_else(|| "unknown".to_string());
    let stamp = unix_timestamp();
    let dir = PathBuf::from("perf");
    if let Err(err) = fs::create_dir_all(&dir) {
        eprintln!("failed to create perf dir: {err}");
        std::process::exit(1);
    }

    let filename = format!("{commit}_{stamp}.csv");
    let path = dir.join(filename);
    let mut file = File::create(&path).expect("failed to create perf output file");
    writeln!(file, "size,btree_put_ms,btree_get_ms,gpu_put_ms,gpu_get_ms")
        .expect("failed to write header");

    for (size, btree_put, btree_get, gpu_put, gpu_get) in rows {
        writeln!(
            file,
            "{},{:.3},{:.3},{:.3},{:.3}",
            size,
            duration_ms(btree_put),
            duration_ms(btree_get),
            duration_ms(gpu_put),
            duration_ms(gpu_get)
        )
        .expect("failed to write row");
    }

    println!("wrote perf results to {}", path.display());
}

fn bench_btree(keys: &[u32]) -> (Duration, Duration) {
    let mut map = BTreeMap::new();
    let start = Instant::now();
    for &key in keys {
        map.insert(key, key.wrapping_mul(31));
    }
    let put = start.elapsed();

    let start = Instant::now();
    let mut sink = 0_u64;
    for &key in keys {
        if let Some(value) = map.get(&key) {
            sink = sink.wrapping_add(*value as u64);
        }
    }
    std::hint::black_box(sink);
    (put, start.elapsed())
}

fn bench_gpu(entries: &[KvEntry], keys: &[Key]) -> (Duration, Duration) {
    let mut map = pollster::block_on(GpuSortedMap::new(Capacity::new((entries.len() * 2) as u32)))
        .expect("failed to init GPU map");

    let start = Instant::now();
    map.bulk_put(entries).expect("bulk_put failed");
    let put = start.elapsed();

    let start = Instant::now();
    let values = map.bulk_get(keys);
    std::hint::black_box(values);
    (put, start.elapsed())
}

fn git_head() -> Option<String> {
    let output = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let mut head = String::from_utf8_lossy(&output.stdout).to_string();
    head.truncate(head.trim_end().len());
    Some(head.trim().to_string())
}

fn unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_else(|_| Duration::from_secs(0))
        .as_secs()
}

fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}
