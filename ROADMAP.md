# GPUSortedMap Roadmap

## Project Blueprint: "Slab-GPU KV"

### Phase 1: Core Engine & Memory Architecture
Objective: Establish the Rust/wgpu foundation and the memory layout.
Technology Stack: Rust, wgpu (Metal/Vulkan/D3D12), bytemuck (Pod/Zeroable data safety).
Memory Layout: Implement a slab-based layout. Avoid linked pointers. Use a u32 key/value entry system (8 bytes per entry) to fit within standard cache lines.
Buffer Strategy:
- Main slab: a large storage buffer containing sorted entries.
- Input buffer: for batch updates.
- Uniforms: for metadata (current slab size, search keys).

### Phase 2: GPU Kernels (WGSL)
Objective: Write the logic that executes on the GPU cores.
- `get_kernel`: implement a warp-centric binary search. A group of 64 threads should handle a batch of 64 lookups. Optimize for branch divergence.
- `range_scan_kernel`: implement a "lower bound" kernel that returns the starting index for `startKey`. The actual range retrieval is a simple memory copy of the subsequent N elements.
- `delete_kernel` (tombstones): implement deletion by writing a `0xFFFFFFFF` marker to the value slot.

### Phase 3: Multi-Platform Bridge (C-ABI)
Objective: Expose the Rust engine to Go, Python, and C++.
Task: Use `#[no_mangle]` and `extern "C"` to export the following functions:
- `gpu_kv_init()`: Initializes the wgpu instance and device.
- `gpu_kv_bulk_put(ptr, len)`: Ingests data, sorts it via CPU (or GPU radix sort if complexity allows), and uploads.
- `gpu_kv_get_batch(keys_ptr, keys_len, results_ptr)`: Dispatches the compute shader.
Binding Generation: Use cbindgen to generate a .h header file.

### Phase 4: Benchmarking Suite
Objective: Prove performance against existing B+ tree projects.
- CPU baseline: use the current B+ tree implementation as the control.
- The "PCIe tax" test: measure the latency of a single get() vs. a bulk_get(100,000).
- Throughput test: chart ops/second as the batch size increases from 1 to 1,000,000.
- Memory efficiency: compare VRAM usage vs. RAM usage for the same 10,000,000 keys.

## LLM Prompt (legacy)

"You are an expert systems engineer. Implement a high-performance Key/Value store in Rust using wgpu for hardware acceleration. Requirements: Data Structure: Use a slab-allocated sorted array. Operations: bulk_put, bulk_get, range_scan, and delete. Concurrency: Use wgpu compute pipelines with WGSL shaders. Portability: Ensure the instance creation selects the best available backend (Metal for Mac, Vulkan for NVIDIA). Interface: Provide a C-compatible API (extern \"C\") so it can be called from Python or Go. Testing: Write a test module that validates key persistence across bulk operations and handles empty/missing key lookups. Benchmarking: Include a benchmark that measures 'Throughput per Batch Size' and outputs results in a CSV format."

## Success Metrics

A successful implementation should reach a break-even point where the GPU's
throughput surpasses the CPU's B+ tree once the batch size exceeds
~5,000–10,000 keys.
