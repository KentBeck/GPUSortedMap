# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Project metadata and documentation improvements
- CHANGELOG for tracking version history
- Minimum Supported Rust Version (MSRV) specification

### Fixed
- Clippy warnings for cleaner, more idiomatic code

## [0.1.0] - 2026-01-28

### Added
- Initial release
- GPU-accelerated sorted key/value store using wgpu
- Core operations: `bulk_put`, `bulk_get`, `bulk_delete`, `range`
- Convenience single-key operations: `put`, `get`, `delete`
- Slab-allocated sorted array architecture
- Support for Metal, Vulkan, and D3D12 backends
- Comprehensive test suite
- Benchmarking infrastructure
- Documentation and examples

[Unreleased]: https://github.com/KentBeck/GPUSortedMap/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/KentBeck/GPUSortedMap/releases/tag/v0.1.0
