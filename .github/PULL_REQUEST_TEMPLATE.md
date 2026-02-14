## Summary

Describe the change and why it matters.

## Invariants touched

- [ ] Sorted slab key ordering
- [ ] Tombstone sentinel behavior (`0xFFFF_FFFF`)
- [ ] `live_len` / `len()` semantics
- [ ] Range behavior (`[from, to)` and tombstone filtering)
- [ ] None

## Risk and behavior notes

List regressions this could cause and why they are unlikely.

## Testing

- [ ] `cargo fmt --all -- --check`
- [ ] `cargo clippy --all-targets --all-features -- -D warnings`
- [ ] `cargo test`
- [ ] `PERF_SIZES=1,10,100 cargo bench --bench perf` (smoke)
- [ ] Full perf run + CSV captured (if perf-relevant)

## Evidence

Include key output snippets, failing test names fixed, and benchmark deltas if relevant.
