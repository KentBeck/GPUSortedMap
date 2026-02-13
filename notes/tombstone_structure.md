# Tombstone structure notes

## What the tombstone is

- Tombstones are encoded inline in `KvEntry.value` using a reserved sentinel: `0xFFFF_FFFF`.
- The sentinel is defined in host code as `TOMBSTONE_VALUE`.
- There is no separate tombstone bitmap or side index.

References:
- `src/lib.rs:249`
- `src/lib.rs:241`
- `README.md:149`

## Where tombstones live

- The sorted slab stores `KvEntry { key, value }` records on GPU.
- Deletion updates only the `value` field at the matched key slot.
- Key ordering remains unchanged; deleted keys still occupy their sorted position.

References:
- `src/lib.rs:254`
- `src/pipelines/bulk_delete.rs:183`
- `src/pipelines/bulk_delete.rs:184`

## Tombstone write path (delete)

- `GpuSortedMap::bulk_delete` de-duplicates keys and computes how many currently exist.
- It dispatches `BulkDeletePipeline`, which binary-searches each key in parallel on GPU (`workgroup_size(64)`).
- On match, shader writes `0xffffffffu` into `slab[lo].value`.
- `live_len` is reduced by the number of keys that were live before dispatch.

References:
- `src/lib.rs:403`
- `src/lib.rs:407`
- `src/lib.rs:408`
- `src/lib.rs:410`
- `src/pipelines/bulk_delete.rs:126`
- `src/pipelines/bulk_delete.rs:163`
- `src/pipelines/bulk_delete.rs:183`
- `src/pipelines/bulk_delete.rs:184`

## Tombstone read path (get)

- `bulk_get` shader returns `{ value, found }` for matched keys.
- Host-side post-processing converts results to `Option<Value>`.
- A result is `None` when `found == 0` or when `value == TOMBSTONE_VALUE`.

References:
- `src/pipelines/bulk_get.rs:171`
- `src/pipelines/bulk_get.rs:175`
- `src/pipelines/bulk_get.rs:176`

## Tombstone read path (range)

- Range shader computes `[start, end)` over sorted keys only.
- It does not filter tombstones in WGSL.
- Host-side `GpuSortedMap::range` filters out entries where `value == TOMBSTONE_VALUE`.

References:
- `src/pipelines/range_scan.rs:243`
- `src/lib.rs:430`
- `src/lib.rs:434`

## Interaction with put/merge

- `bulk_put` rejects input entries that use the reserved sentinel value.
- Merge logic overwrites existing keys with new input values, so re-putting a deleted key revives it.
- Slab logical length (`slab.meta.len`) can include tombstoned entries; live count is tracked separately in `live_len`.

References:
- `src/lib.rs:365`
- `src/lib.rs:370`
- `src/lib.rs:398`
- `src/lib.rs:448`
- `src/pipelines/bulk_put.rs:525`

## Behavioral guarantees covered by tests

- Tombstone value is rejected for user inserts.
- Range excludes tombstoned entries.
- Delete reduces the externally reported live length.

References:
- `src/lib.rs:819`
- `src/lib.rs:796`
- `src/lib.rs:910`

## Tradeoffs and implications

- Pros:
  - Delete is cheap (single value write; no compaction).
  - Sorted key layout remains stable for binary search.
- Costs:
  - Slab can accumulate dead slots, increasing search span and memory traffic over time.
  - `range` currently copies full key span then filters on CPU, so tombstone-heavy ranges pay extra readback cost.
  - Value domain is reduced by one reserved sentinel.

## Possible follow-up directions

- Add periodic GPU compaction to remove tombstoned slots and shrink `slab.meta.len`.
- Add an optional GPU-side range filter/write-compact path to avoid tombstone readback.
- If full `u32` value space is required later, migrate to explicit liveness metadata (bitmap/byte-mask).
