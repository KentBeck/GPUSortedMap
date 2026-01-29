# Development Guide

This guide helps contributors and maintainers work on GPUSortedMap effectively.

## Prerequisites

- **Rust**: Stable toolchain (MSRV: 1.73)
- **GPU Backend**: One of:
  - Metal (macOS)
  - Vulkan (Linux)
  - DirectX 12 (Windows)
- **Optional**: For benchmarking, ensure your GPU has adequate VRAM

## Development Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/KentBeck/GPUSortedMap.git
   cd GPUSortedMap
   ```

2. **Build the project**:
   ```bash
   cargo build
   ```

3. **Run tests**:
   ```bash
   cargo test
   ```
   
   Note: Tests require a GPU or software adapter. If tests fail with "no suitable GPU adapters found", your environment may lack GPU support.

## Development Workflow

### Code Formatting

Always format your code before committing:

```bash
cargo fmt --all
```

### Linting

Run clippy to catch common mistakes:

```bash
cargo clippy --all-targets --all-features -- -D warnings
```

### Testing

Run the full test suite:

```bash
cargo test --all-features
```

Run a specific test:

```bash
cargo test test_name
```

### Documentation

Generate and view documentation locally:

```bash
cargo doc --open --no-deps
```

Test documentation examples:

```bash
cargo test --doc
```

### Benchmarking

Run performance benchmarks:

```bash
cargo bench --bench perf
```

Results are written to the `perf/` directory as CSV files.

### Running Examples

```bash
cargo run --example basic
```

## Project Structure

```
GPUSortedMap/
├── src/
│   ├── lib.rs              # Public API and core logic
│   ├── gpu_array.rs        # GPU buffer management
│   ├── pipelines.rs        # Pipeline orchestration
│   └── pipelines/          # Individual compute pipelines
│       ├── bulk_get.rs
│       ├── bulk_put.rs
│       ├── bulk_delete.rs
│       ├── range_scan.rs
│       └── utils.rs
├── benches/                # Performance benchmarks
├── examples/               # Usage examples
├── tests/                  # Integration tests (if any)
└── .github/workflows/      # CI/CD configuration
```

## Adding New Features

1. **Write tests first**: Add tests that demonstrate the desired behavior
2. **Implement minimally**: Make the smallest change that passes tests
3. **Document**: Add doc comments and update README if needed
4. **Benchmark**: If performance-critical, add benchmarks
5. **Update CHANGELOG**: Document the change in CHANGELOG.md

## GPU Compute Shaders

Shaders are written in WGSL (WebGPU Shading Language) and embedded in the Rust code.

Key files containing shaders:
- `src/pipelines/bulk_get.rs` - Binary search shader
- `src/pipelines/bulk_put.rs` - Merge and sort shader
- `src/pipelines/bulk_delete.rs` - Tombstone marking shader
- `src/pipelines/range_scan.rs` - Range query shader

## Performance Considerations

1. **Batch Operations**: GPU operations have overhead. Batch operations are most efficient with >1000 items
2. **Memory Layout**: Data uses 8-byte aligned structs (key: u32, value: u32)
3. **PCIe Transfer**: Data transfer between CPU and GPU has latency; design for bulk operations
4. **Workgroup Size**: Shaders use 64-thread workgroups for optimal occupancy

## Debugging GPU Code

1. **Use validation layers**: wgpu enables them by default in debug builds
2. **Check shader compilation**: Errors appear at pipeline creation time
3. **CPU fallback**: Tests attempt CPU adapter fallback for debugging without GPU

## Release Process

1. Update version in `Cargo.toml`
2. Update `CHANGELOG.md` with release notes
3. Commit changes: `git commit -m "Release vX.Y.Z"`
4. Tag the release: `git tag vX.Y.Z`
5. Push tags: `git push origin vX.Y.Z`
6. CI will automatically publish to crates.io and create a GitHub release

## Troubleshooting

### Tests Fail with "no suitable GPU adapters found"

This happens in environments without GPU access (e.g., some CI runners). The code attempts to use a software adapter fallback, but if unavailable, tests will fail. This is expected behavior.

### Performance Lower than Expected

- Ensure you're running in release mode: `cargo build --release`
- Check batch sizes are large enough (>1000 items recommended)
- Verify GPU is being used (check adapter info during initialization)

## Getting Help

- Open an issue on GitHub for bugs or feature requests
- See `CONTRIBUTING.md` for contribution guidelines
- Review `ROADMAP.md` for planned features
