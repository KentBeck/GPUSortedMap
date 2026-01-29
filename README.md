# GPUSortedMap

[![CI](https://github.com/KentBeck/GPUSortedMap/actions/workflows/ci.yml/badge.svg)](https://github.com/KentBeck/GPUSortedMap/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/gpusorted_map.svg)](https://crates.io/crates/gpusorted_map)
[![Documentation](https://docs.rs/gpusorted_map/badge.svg)](https://docs.rs/gpusorted_map)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

GPU-accelerated sorted key/value store built in Rust on top of wgpu. The core
idea is a slab-allocated sorted array optimized for batched GPU operations.

## Features

- 🚀 **GPU-accelerated**: Leverages compute shaders for high-throughput batch operations
- 📦 **Slab-allocated**: Efficient memory management with sorted array structure
- 🔄 **Batch operations**: Optimized for bulk inserts, lookups, and deletes
- 🎯 **Range queries**: Fast half-open interval scans `[from, to)`
- 🔧 **Zero-copy**: Direct GPU memory access for maximum performance
- 🌐 **Cross-platform**: Supports Metal (macOS), Vulkan (Linux), and D3D12 (Windows)
- 🛡️ **Type-safe**: Strong newtype wrappers for keys, values, and capacities

## Quick start

Add the crate:

```bash
cargo add gpusorted_map
```

Minimal example:

```rust
use gpusorted_map::{Capacity, GpuSortedMap, Key, KvEntry, Value};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut map = pollster::block_on(GpuSortedMap::new(Capacity::new(1024)))?;

    map.bulk_put(&[
        KvEntry { key: Key::new(1), value: Value::new(10) },
        KvEntry { key: Key::new(2), value: Value::new(20) },
    ])?;

    assert_eq!(map.get(Key::new(1)), Some(Value::new(10)));
    map.delete(Key::new(2));
    assert_eq!(map.get(Key::new(2)), None);

    Ok(())
}
```

## Requirements

- A working GPU backend supported by wgpu (Metal on macOS, Vulkan on Linux,
  D3D12 on Windows).
- Tests attempt to use a fallback adapter if no GPU is present. If no
  software adapter is available, tests may fail.

## API overview

- `bulk_put(&[KvEntry]) -> Result<(), GpuMapError>` - Batch insert/update
- `bulk_get(&[Key]) -> Vec<Option<Value>>` - Batch lookup
- `bulk_delete(&[Key])` - Batch delete
- `range(from_key, to_key) -> Vec<KvEntry>` - Half-open range query `[from, to)`
- Convenience helpers: `put`, `get`, `delete`

### Advanced examples

#### Working with ranges

```rust
use gpusorted_map::{Capacity, GpuSortedMap, Key, KvEntry, Value};

let mut map = pollster::block_on(GpuSortedMap::new(Capacity::new(1024)))?;

// Insert multiple entries
map.bulk_put(&[
    KvEntry { key: Key::new(10), value: Value::new(100) },
    KvEntry { key: Key::new(20), value: Value::new(200) },
    KvEntry { key: Key::new(30), value: Value::new(300) },
])?;

// Query a range [10, 30) - includes 10 and 20, but not 30
let entries = map.range(Key::new(10), Key::new(30));
assert_eq!(entries.len(), 2);

// Iterate over range
for entry in map.range_iter(Key::new(10), Key::new(40)) {
    println!("Key: {}, Value: {}", entry.key.0, entry.value.0);
}
# Ok::<(), Box<dyn std::error::Error>>(())
```

#### Batch operations for performance

```rust
use gpusorted_map::{Capacity, GpuSortedMap, Key, KvEntry, Value};

let mut map = pollster::block_on(GpuSortedMap::new(Capacity::new(10000)))?;

// Prepare a large batch
let entries: Vec<KvEntry> = (0..5000)
    .map(|i| KvEntry { 
        key: Key::new(i), 
        value: Value::new(i * 10) 
    })
    .collect();

// Single GPU operation handles all 5000 entries
map.bulk_put(&entries)?;

// Batch lookup is also efficient
let keys: Vec<Key> = (0..100).map(Key::new).collect();
let values = map.bulk_get(&keys);
# Ok::<(), Box<dyn std::error::Error>>(())
```

#### Error handling

```rust
use gpusorted_map::{Capacity, GpuSortedMap, Key, KvEntry, Value, GpuMapError};

let mut map = pollster::block_on(GpuSortedMap::new(Capacity::new(10)))?;

// Handle capacity errors
let large_batch: Vec<KvEntry> = (0..20)
    .map(|i| KvEntry { key: Key::new(i), value: Value::new(i) })
    .collect();

match map.bulk_put(&large_batch) {
    Err(GpuMapError::CapacityExceeded { capacity, requested }) => {
        println!("Capacity {} exceeded, requested {}", capacity.0, requested.0);
    }
    Err(GpuMapError::DuplicateKeys { key }) => {
        println!("Duplicate key found: {}", key.0);
    }
    Err(GpuMapError::TombstoneValueReserved { value }) => {
        println!("Reserved value used: 0x{:08X}", value.0);
    }
    Ok(_) => println!("Batch inserted successfully"),
}
# Ok::<(), Box<dyn std::error::Error>>(())
```

### Newtype helpers

`Key`, `Value`, `Capacity`, and `Length` are `#[repr(transparent)]` newtypes over `u32`.
You can use `Key::new(42)` or `Key::from(42)`, and convert back with `u32::from(key)`.

Notes:
- `0xFFFF_FFFF` is reserved as the tombstone value.
- `bulk_put` returns `GpuMapError::CapacityExceeded` when the requested size
  exceeds the slab capacity.
- `bulk_put` returns `GpuMapError::DuplicateKeys` if the batch contains the same key twice.
- `len()` reports live entries (tombstones excluded).

## Benchmarks

The benchmark writes CSV output into `perf/`:

```bash
cargo bench --bench perf
```

## Project status

This crate is experimental and the public API may change.

## Documentation

- **API Documentation**: [docs.rs/gpusorted_map](https://docs.rs/gpusorted_map)
- **Development Guide**: See [DEVELOPMENT.md](DEVELOPMENT.md) for contribution setup
- **Examples**: Check the [examples/](examples/) directory
- **Changelog**: See [CHANGELOG.md](CHANGELOG.md) for version history

## Roadmap

See `ROADMAP.md`.

## Contributing

See `CONTRIBUTING.md`.

## License

MIT. See `LICENSE`.
