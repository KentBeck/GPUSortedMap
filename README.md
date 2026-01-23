# GPUSortedMap

High-performance GPU-accelerated sorted key/value store, built in Rust on
top of wgpu. The core idea is a slab-allocated sorted array optimized for
batched operations on the GPU.

## Project Blueprint: "Slab-GPU KV"

### Phase 1: Core Engine & Memory Architecture
Objective: Establish the Rust/wgpu foundation and the memory layout.
Technology Stack: Rust, wgpu (for Metal/Vulkan/D3D12), bytemuck (Pod/Zeroable data safety).
Memory Layout: Implement a Slab-based layout. Avoid linked pointers. Use a u32 key/value entry system (8 bytes per entry) to fit within standard cache lines.
Buffer Strategy:
* Main Slab: A large storage buffer containing sorted entries.
Input Buffer: For batch updates.
Uniforms: For metadata (current slab size, search keys).

### Phase 2: The GPU Kernels (WGSL)
Objective: Write the logic that executes on the GPU cores.
get_kernel: Implement a Warp-Centric Binary Search. A group of 64 threads should handle a batch of 64 lookups. Use the while loop logic provided previously, but optimize for branch divergence.
range_scan_kernel: Instead of a complex iterator, implement a "Lower Bound" kernel that returns the starting index for startKey. The actual range retrieval is a simple memory copy of the subsequent $N$ elements.
delete_kernel (Tombstones): Implement deletion by writing a 0xFFFFFFFF marker (Tombstone) to the value slot.

### Phase 3: The Multi-Platform Bridge (C-ABI)
Objective: Expose the Rust engine to Go, Python, and C++.
Task: Use #[no_mangle] and extern "C" to export the following functions:
gpu_kv_init(): Initializes the wgpu instance and device.
gpu_kv_bulk_put(ptr, len): Ingests data, sorts it via CPU (or GPU Radix Sort if complexity allows), and uploads.
gpu_kv_get_batch(keys_ptr, keys_len, results_ptr): Dispatches the compute shader.
Binding Generation: Use cbindgen to generate a .h header file.

### Phase 4: Rigorous Benchmarking Suite
Objective: Prove performance against your existing B+ tree projects.
CPU Baseline: Use your current B+ tree implementation as the control.
The "PCIe Tax" Test: Measure the latency of a single get() vs. a bulk_get(100,000).
Throughput Test: Chart "Ops/Second" as the batch size increases from $1$ to $10^6$.
Memory Efficiency: Compare VRAM usage vs. RAM usage for the same $10^7$ keys.

## 🤖 Instructions for the Implementing LLM

Prompt to give the LLM:
"You are an expert systems engineer. Implement a high-performance Key/Value store in Rust using wgpu for hardware acceleration.Requirements:Data Structure: Use a Slab-allocated Sorted Array.Operations: bulk_put, bulk_get, range_scan, and delete.Concurrency: Use wgpu Compute Pipelines with WGSL shaders.Portability: Ensure the Instance creation selects the best available backend (Metal for Mac, Vulkan for NVIDIA).Interface: Provide a C-compatible API (extern "C") so it can be called from Python or Go.Testing: Write a test module that validates key persistence across bulk operations and handles empty/missing key lookups.Benchmarking: Include a benchmark that measures 'Throughput per Batch Size' and outputs results in a CSV format."

## 📈 Success Metrics

A successful implementation should reach a "Break-even Point" where the GPU's
throughput surpasses the CPU's B+ tree once the batch size exceeds
~5,000–10,000 keys.
