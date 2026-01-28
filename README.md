# GPUSortedMap

GPU-accelerated sorted key/value store built in Rust on top of wgpu. The core
idea is a slab-allocated sorted array optimized for batched GPU operations.

## Quick start

Add the crate:

```bash
cargo add gpusorted_map
```

Minimal example:

```rust
use gpusorted_map::{GpuSortedMap, KvEntry};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut map = pollster::block_on(GpuSortedMap::new(1024))?;

    map.bulk_put(&[
        KvEntry { key: 1, value: 10 },
        KvEntry { key: 2, value: 20 },
    ])?;

    assert_eq!(map.get(1), Some(10));
    map.delete(2);
    assert_eq!(map.get(2), None);

    Ok(())
}
```

## Requirements

- A working GPU backend supported by wgpu (Metal on macOS, Vulkan on Linux,
  D3D12 on Windows).
- Tests attempt to use a fallback adapter if no GPU is present. If no
  software adapter is available, tests may fail.

## API overview

- `bulk_put(&[KvEntry]) -> Result<(), GpuMapError>`
- `bulk_get(&[u32]) -> Vec<Option<u32>>`
- `bulk_delete(&[u32])`
- `range(from_key, to_key) -> Vec<KvEntry>` (half-open range)
- Convenience helpers: `put`, `get`, `delete`

Notes:
- `0xFFFF_FFFF` is reserved as the tombstone value.
- `bulk_put` returns `GpuMapError::CapacityExceeded` when the requested size
  exceeds the slab capacity.

## Benchmarks

The benchmark writes CSV output into `perf/`:

```bash
cargo bench --bench perf
```

## Project status

This crate is experimental and the public API may change.

## Roadmap

See `ROADMAP.md`.

## Contributing

See `CONTRIBUTING.md`.

## License

MIT. See `LICENSE`.
