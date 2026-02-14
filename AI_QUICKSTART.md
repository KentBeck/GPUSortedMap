# AI Quickstart

Use this as the shortest path to productive changes in `gpusorted_map`.

## 1) Read first

- `README.md` for API expectations
- `AGENTS.md` for repo rules and invariants
- `src/lib.rs` for orchestration and externally visible behavior

## 2) Architecture in 60 seconds

- `GpuSortedMap` in `src/lib.rs` coordinates all operations.
- `GpuArray`/`GpuStorage` in `src/gpu_array.rs` manage storage buffers + metadata.
- Compute pipelines:
  - `src/pipelines/bulk_put.rs`: sort, dedup, merge
  - `src/pipelines/bulk_get.rs`: parallel binary-search lookups
  - `src/pipelines/bulk_delete.rs`: parallel binary-search + tombstone write
  - `src/pipelines/range_scan.rs`: key-bound discovery for `[from, to)`

## 3) High-value invariants to protect

- Slab keys remain sorted.
- Tombstone sentinel is `0xFFFF_FFFF` and is reserved from user values.
- `len()` means live entries, not slab slots.
- `range()` and `bulk_get()` hide tombstones from callers.

## 4) Task playbooks

### Add or modify an operation

1. Update host behavior in `src/lib.rs`.
2. Update/extend pipeline shader and binding layout in the relevant `src/pipelines/*.rs`.
3. Add tests in `src/lib.rs` that assert API-level behavior.
4. Run full local checks.

### Change put/merge behavior

1. Start in `src/pipelines/bulk_put.rs`.
2. Verify dedup + merge length semantics (`merge_meta.len`).
3. Re-check capacity and `live_len` semantics in `src/lib.rs`.
4. Add regression tests around duplicates, deletes, and tombstone compaction.

### Change tombstone semantics

1. Audit `bulk_delete`, `bulk_get`, and `range` paths together.
2. Confirm `TOMBSTONE_VALUE` checks are aligned across host and WGSL.
3. Revisit `notes/tombstone_structure.md`.

## 5) Local validation checklist

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test
```

Optional benchmark run:

```bash
cargo bench --bench perf
```

If recording benchmark output, force-add the generated CSV from `perf/`.

## 6) Fast triage for CI failures

- Fmt-only failure: run `cargo fmt --all`.
- Clippy failure: fix warnings, do not silence broadly.
- Test failure in map behavior: inspect `src/lib.rs` tests first, then the relevant pipeline.
- Cross-platform build failure: check buffer/layout assumptions and unsupported WGSL usage.

## 7) What “done” looks like

- Behavior covered by tests, not just implementation changes.
- All local checks pass.
- Changeset clearly states why invariants still hold.
