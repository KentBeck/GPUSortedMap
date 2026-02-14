# AGENTS.md

This file helps coding agents work safely and quickly in this repository.

## Project purpose

`gpusorted_map` is a GPU-backed sorted key/value map optimized for batch operations:

- `bulk_put`
- `bulk_get`
- `bulk_delete`
- `range`

Core implementation lives in:

- `src/lib.rs`
- `src/pipelines/*.rs`
- `src/gpu_array.rs`

## Canonical commands

Run these before proposing changes:

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test
```

Benchmark:

```bash
cargo bench --bench perf
```

## Critical invariants

1. Slab keys are sorted.
2. Tombstones are encoded inline as reserved value `0xFFFF_FFFF`.
3. User inserts must reject tombstone value (`GpuMapError::TombstoneValueReserved`).
4. `len()` reports live entries (`live_len`), not raw slab length.
5. `range()` must exclude tombstones from returned entries.
6. `bulk_get` must return `None` for missing keys and tombstoned keys.

## Editing guidance

- Prefer keeping host-side API behavior stable in `src/lib.rs`.
- Pipeline WGSL changes should preserve buffer layouts and bind indices.
- If you touch merge/delete/get shaders, add or update tests in `src/lib.rs`.
- Keep benchmarks reproducible; benchmark key generation must avoid duplicate keys.

## Common change map

- Public API and semantics: `src/lib.rs`
- GPU buffer wrappers/meta: `src/gpu_array.rs`
- Bulk read path: `src/pipelines/bulk_get.rs`
- Bulk delete path: `src/pipelines/bulk_delete.rs`
- Put/sort/dedup/merge path: `src/pipelines/bulk_put.rs`
- Range bounds path: `src/pipelines/range_scan.rs`
- Performance harness: `benches/perf.rs`

## CI expectations

GitHub Actions checks format, clippy, docs, build, and tests across platforms/toolchains.
Typical failure sources:

- `cargo fmt --check` mismatch
- clippy warnings promoted to errors
- tests relying on behavior that changed in WGSL merge/get/delete logic

## Notes

- Tombstone structure and tradeoffs are documented in `notes/tombstone_structure.md`.
- Perf CSV files are ignored by default (`/perf/*.csv` in `.gitignore`), so force-add when needed.
