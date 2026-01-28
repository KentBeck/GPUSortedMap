# Contributing

Thanks for helping improve GPUSortedMap.

## Requirements

- Rust stable
- A working GPU backend supported by wgpu (Metal/Vulkan/D3D12)

Tests attempt to use a fallback adapter if no GPU is present. If no software
adapter is available, tests may fail.

## Common commands

Format:

```bash
cargo fmt
```

Lint:

```bash
cargo clippy --all-targets --all-features
```

Tests:

```bash
cargo test
```

Benchmarks (writes CSV to `perf/`):

```bash
cargo bench --bench perf
```

## Pull requests

- Keep changes focused and explain intent in the PR description.
- Include test/bench results when relevant.
- Update docs if you change public behavior.
