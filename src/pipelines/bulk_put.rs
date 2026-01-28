use std::sync::Arc;

use crate::gpu_array::{GpuArray, GpuStorage};
use crate::pipelines::core::ComputeStep;
use crate::pipelines::data::{DedupParams, InputMeta, MergeMeta, SortParams};
use crate::pipelines::utils::{create_buffer_with_data, readback_single};
use crate::{Key, KvEntry, Length, Value};

const BULK_SORT_BIND_INPUT: u32 = 0;
const BULK_SORT_BIND_PARAMS: u32 = 1;

const BULK_DEDUP_BIND_INPUT: u32 = 0;
const BULK_DEDUP_BIND_PARAMS: u32 = 1;
const BULK_DEDUP_BIND_META: u32 = 2;

const BULK_MERGE_BIND_SLAB: u32 = 0;
const BULK_MERGE_BIND_INPUT: u32 = 1;
const BULK_MERGE_BIND_OUTPUT: u32 = 2;
const BULK_MERGE_BIND_SLAB_META: u32 = 3;
const BULK_MERGE_BIND_INPUT_META: u32 = 4;
const BULK_MERGE_BIND_MERGE_META: u32 = 5;

pub struct BulkPutPipeline {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    sort_step: ComputeStep,
    dedup_step: ComputeStep,
    merge_step: ComputeStep,
}

impl BulkPutPipeline {
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        let sort_step = ComputeStep::new(
            Arc::clone(&device),
            BULK_SORT_WGSL,
            "main",
            &[
                wgpu::BindGroupLayoutEntry {
                    binding: BULK_SORT_BIND_INPUT,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: BULK_SORT_BIND_PARAMS,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        );

        let dedup_step = ComputeStep::new(
            Arc::clone(&device),
            BULK_DEDUP_WGSL,
            "main",
            &[
                wgpu::BindGroupLayoutEntry {
                    binding: BULK_DEDUP_BIND_INPUT,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: BULK_DEDUP_BIND_PARAMS,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: BULK_DEDUP_BIND_META,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        );

        let merge_step = ComputeStep::new(
            Arc::clone(&device),
            BULK_MERGE_WGSL,
            "main",
            &[
                wgpu::BindGroupLayoutEntry {
                    binding: BULK_MERGE_BIND_SLAB,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: BULK_MERGE_BIND_INPUT,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: BULK_MERGE_BIND_OUTPUT,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: BULK_MERGE_BIND_SLAB_META,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: BULK_MERGE_BIND_INPUT_META,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: BULK_MERGE_BIND_MERGE_META,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        );

        Self {
            device,
            queue,
            sort_step,
            dedup_step,
            merge_step,
        }
    }

    pub fn execute(
        &self,
        slab: &GpuArray<KvEntry>,
        input: &GpuArray<KvEntry>,
        merge: &GpuArray<KvEntry>,
        merge_meta: &GpuStorage<MergeMeta>,
        entries_len: u32,
    ) -> Result<u32, crate::GpuMapError> {
        let len = entries_len;
        let padded_len = len.next_power_of_two();
        if padded_len > slab.capacity().0 {
            return Err(crate::GpuMapError::CapacityExceeded {
                capacity: slab.capacity(),
                requested: Length::new(slab.len().0 + padded_len),
            });
        }

        if padded_len > len {
            let pad_count = (padded_len - len) as usize;
            let padding = vec![
                KvEntry {
                    key: Key::new(u32::MAX),
                    value: Value::new(0),
                };
                pad_count
            ];
            let offset = (len as u64) * std::mem::size_of::<KvEntry>() as u64;
            self.queue
                .write_buffer(input.buffer(), offset, bytemuck::cast_slice(&padding));
        }

        if len > 1 {
            self.run_sort_step(input, padded_len);
        }

        let dedup_len = self.run_dedup_step(input, len, merge_meta);

        let merge_len = self.run_merge_step(slab, input, merge, merge_meta, dedup_len);
        Ok(merge_len)
    }

    fn run_sort_step(&self, input: &GpuArray<KvEntry>, padded_len: u32) {
        let sort_params_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bulk-sort-params"),
            size: std::mem::size_of::<SortParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sort_bind_group = self.sort_step.create_bind_group(
            "bulk-sort-bind-group",
            &[
                wgpu::BindGroupEntry {
                    binding: BULK_SORT_BIND_INPUT,
                    resource: input.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: BULK_SORT_BIND_PARAMS,
                    resource: sort_params_buffer.as_entire_binding(),
                },
            ],
        );

        let workgroups = ((padded_len + 63) / 64) as u32;
        let mut k = 2u32;
        let max_k = padded_len;
        while k <= max_k {
            let mut j = k / 2;
            while j > 0 {
                let params = SortParams {
                    k,
                    j,
                    len: padded_len,
                    _pad: 0,
                };
                self.queue
                    .write_buffer(&sort_params_buffer, 0, bytemuck::bytes_of(&params));

                let mut sort_encoder =
                    self.device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("bulk-sort-encoder"),
                        });
                self.sort_step.dispatch(
                    &mut sort_encoder,
                    "bulk-sort-pass",
                    &sort_bind_group,
                    (workgroups, 1, 1),
                );
                self.queue.submit(Some(sort_encoder.finish()));
                j /= 2;
            }
            k *= 2;
        }
    }

    fn run_dedup_step(
        &self,
        input: &GpuArray<KvEntry>,
        len: u32,
        merge_meta: &GpuStorage<MergeMeta>,
    ) -> u32 {
        let dedup_params = DedupParams { len, _pad: [0; 3] };
        let dedup_params_buffer = create_buffer_with_data(
            &self.device,
            "bulk-dedup-params",
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            &[dedup_params],
        );
        let dedup_bind_group = self.dedup_step.create_bind_group(
            "bulk-dedup-bind-group",
            &[
                wgpu::BindGroupEntry {
                    binding: BULK_DEDUP_BIND_INPUT,
                    resource: input.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: BULK_DEDUP_BIND_PARAMS,
                    resource: dedup_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: BULK_DEDUP_BIND_META,
                    resource: merge_meta.buffer().as_entire_binding(),
                },
            ],
        );

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("bulk-put-encoder"),
            });
        self.dedup_step.dispatch(
            &mut encoder,
            "bulk-dedup-pass",
            &dedup_bind_group,
            (1, 1, 1),
        );

        let dedup_readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bulk-dedup-readback"),
            size: std::mem::size_of::<MergeMeta>() as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(
            merge_meta.buffer(),
            0,
            &dedup_readback,
            0,
            std::mem::size_of::<MergeMeta>() as u64,
        );

        self.queue.submit(Some(encoder.finish()));
        let dedup_meta = readback_single::<MergeMeta>(&self.device, &dedup_readback);
        dedup_meta.len
    }

    fn run_merge_step(
        &self,
        slab: &GpuArray<KvEntry>,
        input: &GpuArray<KvEntry>,
        merge: &GpuArray<KvEntry>,
        merge_meta: &GpuStorage<MergeMeta>,
        dedup_len: u32,
    ) -> u32 {
        let input_meta = InputMeta {
            len: dedup_len,
            _pad: [0; 3],
        };
        let input_meta_buffer = create_buffer_with_data(
            &self.device,
            "input-meta-buffer",
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            &[input_meta],
        );

        let merge_bind_group = self.merge_step.create_bind_group(
            "bulk-merge-bind-group",
            &[
                wgpu::BindGroupEntry {
                    binding: BULK_MERGE_BIND_SLAB,
                    resource: slab.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: BULK_MERGE_BIND_INPUT,
                    resource: input.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: BULK_MERGE_BIND_OUTPUT,
                    resource: merge.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: BULK_MERGE_BIND_SLAB_META,
                    resource: slab.meta_buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: BULK_MERGE_BIND_INPUT_META,
                    resource: input_meta_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: BULK_MERGE_BIND_MERGE_META,
                    resource: merge_meta.buffer().as_entire_binding(),
                },
            ],
        );

        let merge_readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("merge-meta-readback"),
            size: std::mem::size_of::<MergeMeta>() as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("bulk-merge-encoder"),
            });
        self.merge_step.dispatch(
            &mut encoder,
            "bulk-merge-pass",
            &merge_bind_group,
            (1, 1, 1),
        );

        let slab_bytes = (slab.capacity().0 as u64) * std::mem::size_of::<KvEntry>() as u64;
        encoder.copy_buffer_to_buffer(merge.buffer(), 0, slab.buffer(), 0, slab_bytes);
        encoder.copy_buffer_to_buffer(
            merge_meta.buffer(),
            0,
            &merge_readback,
            0,
            std::mem::size_of::<MergeMeta>() as u64,
        );
        self.queue.submit(Some(encoder.finish()));

        let merge_meta_val = readback_single::<MergeMeta>(&self.device, &merge_readback);
        merge_meta_val.len
    }
}

const BULK_SORT_WGSL: &str = r#"
struct KvEntry {
    key: u32,
    value: u32,
};

struct SortParams {
    k: u32,
    j: u32,
    len: u32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read_write> data: array<KvEntry>;
@group(0) @binding(1) var<uniform> params: SortParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.len) {
        return;
    }

    let ixj = i ^ params.j;
    if (ixj > i && ixj < params.len) {
        let a = data[i];
        let b = data[ixj];
        let ascending = (i & params.k) == 0u;
        if (ascending) {
            if (a.key > b.key) {
                data[i] = b;
                data[ixj] = a;
            }
        } else {
            if (a.key < b.key) {
                data[i] = b;
                data[ixj] = a;
            }
        }
    }
}
"#;

const BULK_DEDUP_WGSL: &str = r#"
struct KvEntry {
    key: u32,
    value: u32,
};

struct DedupParams {
    len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

struct MergeMeta {
    len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read_write> data: array<KvEntry>;
@group(0) @binding(1) var<uniform> params: DedupParams;
@group(0) @binding(2) var<storage, read_write> dedup_meta: MergeMeta;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x > 0u) {
        return;
    }

    let len = params.len;
    if (len == 0u) {
        dedup_meta.len = 0u;
        return;
    }

    var write_idx: u32 = 0u;
    var prev_key: u32 = 0u;
    var i: u32 = 0u;
    while (i < len) {
        let entry = data[i];
        if (write_idx == 0u) {
            data[write_idx] = entry;
            prev_key = entry.key;
            write_idx = 1u;
        } else {
            if (entry.key == prev_key) {
                data[write_idx - 1u] = entry;
            } else {
                data[write_idx] = entry;
                prev_key = entry.key;
                write_idx = write_idx + 1u;
            }
        }
        i = i + 1u;
    }

    dedup_meta.len = write_idx;
}
"#;

const BULK_MERGE_WGSL: &str = r#"
struct KvEntry {
    key: u32,
    value: u32,
};

struct SlabMeta {
    len: u32,
    capacity: u32,
    _pad0: u32,
    _pad1: u32,
};

struct InputMeta {
    len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

struct MergeMeta {
    len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read> slab: array<KvEntry>;
@group(0) @binding(1) var<storage, read> input: array<KvEntry>;
@group(0) @binding(2) var<storage, read_write> output: array<KvEntry>;
@group(0) @binding(3) var<uniform> slab_meta: SlabMeta;
@group(0) @binding(4) var<uniform> input_meta: InputMeta;
@group(0) @binding(5) var<storage, read_write> merge_meta: MergeMeta;

fn merge_partition(k: u32, slab_len: u32, input_len: u32) -> vec2<u32> {
    var i_low: u32 = 0u;
    if (k > input_len) {
        i_low = k - input_len;
    }
    var i_high: u32 = k;
    if (i_high > slab_len) {
        i_high = slab_len;
    }

    var i: u32 = i_high;
    var j: u32 = k - i;
    loop {
        let move_left = i > 0u && j < input_len && slab[i - 1u].key >= input[j].key;
        let move_right = j > 0u && i < slab_len && input[j - 1u].key > slab[i].key;
        if (move_left) {
            i_high = i - 1u;
            i = (i_low + i_high) / 2u;
            j = k - i;
            continue;
        }
        if (move_right) {
            i_low = i + 1u;
            i = (i_low + i_high + 1u) / 2u;
            j = k - i;
            continue;
        }
        break;
    }
    return vec2<u32>(i, j);
}

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x > 0u) {
        return;
    }

    let slab_len = slab_meta.len;
    let input_len = input_meta.len;
    if (slab_len == 0u && input_len == 0u) {
        merge_meta.len = 0u;
        return;
    }

    var i: u32 = 0u;
    var j: u32 = 0u;
    var k: u32 = 0u;

    while (i < slab_len && j < input_len) {
        let a = slab[i];
        let b = input[j];
        if (a.key < b.key) {
            output[k] = a;
            i = i + 1u;
        } else if (b.key < a.key) {
            output[k] = b;
            j = j + 1u;
        } else {
            output[k] = b;
            i = i + 1u;
            j = j + 1u;
        }
        k = k + 1u;
    }

    while (i < slab_len) {
        output[k] = slab[i];
        i = i + 1u;
        k = k + 1u;
    }

    while (j < input_len) {
        output[k] = input[j];
        j = j + 1u;
        k = k + 1u;
    }

    merge_meta.len = k;
}
"#;
