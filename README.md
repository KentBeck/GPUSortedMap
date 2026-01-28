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

- `bulk_put(&[KvEntry]) -> Result<(), GpuMapError>`
- `bulk_get(&[Key]) -> Vec<Option<Value>>`
- `bulk_delete(&[Key])`
- `range(from_key, to_key) -> Vec<KvEntry>` (half-open range)
- Convenience helpers: `put`, `get`, `delete`

### Newtype helpers

`Key`, `Value`, `Capacity`, and `Length` are `#[repr(transparent)]` newtypes over `u32`.
You can use `Key::new(42)` or `Key::from(42)`, and convert back with `u32::from(key)`.

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
