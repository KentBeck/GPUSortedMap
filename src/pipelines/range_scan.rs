use std::sync::Arc;

use bytemuck::{Pod, Zeroable};

use crate::gpu_array::GpuArray;
use crate::pipelines::core::ComputeStep;
use crate::pipelines::utils::{create_buffer_with_data, readback_single, readback_vec};
use crate::KvEntry;

const RANGE_BIND_SLAB: u32 = 0;
const RANGE_BIND_SLAB_META: u32 = 1;
const RANGE_BIND_PARAMS: u32 = 2;
const RANGE_BIND_OUTPUT_META: u32 = 3;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug, Default)]
struct RangeParams {
    from_key: u32,
    to_key: u32,
    _pad: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug, Default)]
struct RangeMeta {
    start: u32,
    end: u32,
    _pad: [u32; 2],
}

pub struct RangeScanPipeline {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    step: ComputeStep,
}

impl RangeScanPipeline {
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        let step = ComputeStep::new(
            Arc::clone(&device),
            RANGE_WGSL,
            "main",
            &[
                wgpu::BindGroupLayoutEntry {
                    binding: RANGE_BIND_SLAB,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: RANGE_BIND_SLAB_META,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: RANGE_BIND_PARAMS,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: RANGE_BIND_OUTPUT_META,
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
            step,
        }
    }

    pub fn execute(&self, slab: &GpuArray<KvEntry>, from_key: u32, to_key: u32) -> Vec<KvEntry> {
        if from_key >= to_key || slab.len() == 0 {
            return Vec::new();
        }

        let params = RangeParams {
            from_key,
            to_key,
            _pad: [0; 2],
        };
        let params_buffer = create_buffer_with_data(
            &self.device,
            "range-params",
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            &[params],
        );

        let output_meta = RangeMeta::default();
        let output_meta_buffer = create_buffer_with_data(
            &self.device,
            "range-meta",
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            &[output_meta],
        );

        let bind_group = self.step.create_bind_group(
            "range-bind-group",
            &[
                wgpu::BindGroupEntry {
                    binding: RANGE_BIND_SLAB,
                    resource: slab.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: RANGE_BIND_SLAB_META,
                    resource: slab.meta_buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: RANGE_BIND_PARAMS,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: RANGE_BIND_OUTPUT_META,
                    resource: output_meta_buffer.as_entire_binding(),
                },
            ],
        );

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("range-encoder"),
            });
        self.step
            .dispatch(&mut encoder, "range-pass", &bind_group, (1, 1, 1));

        let output_readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("range-meta-readback"),
            size: std::mem::size_of::<RangeMeta>() as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(
            &output_meta_buffer,
            0,
            &output_readback,
            0,
            std::mem::size_of::<RangeMeta>() as u64,
        );
        self.queue.submit(Some(encoder.finish()));

        let meta = readback_single::<RangeMeta>(&self.device, &output_readback);
        if meta.end <= meta.start {
            return Vec::new();
        }

        let count = meta.end - meta.start;
        let byte_len = (count as u64) * std::mem::size_of::<KvEntry>() as u64;
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("range-readback"),
            size: byte_len,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("range-copy-encoder"),
            });
        let offset = (meta.start as u64) * std::mem::size_of::<KvEntry>() as u64;
        encoder.copy_buffer_to_buffer(slab.buffer(), offset, &readback, 0, byte_len);
        self.queue.submit(Some(encoder.finish()));

        readback_vec::<KvEntry>(&self.device, &readback)
    }
}

const RANGE_WGSL: &str = r#"
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

struct RangeParams {
    from_key: u32,
    to_key: u32,
    _pad0: u32,
    _pad1: u32,
};

struct RangeMeta {
    start: u32,
    end: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read> slab: array<KvEntry>;
@group(0) @binding(1) var<uniform> slab_meta: SlabMeta;
@group(0) @binding(2) var<uniform> params: RangeParams;
@group(0) @binding(3) var<storage, read_write> out_meta: RangeMeta;

fn lower_bound(tgt: u32, len: u32) -> u32 {
    var lo: u32 = 0u;
    var hi: u32 = len;
    while (lo < hi) {
        let mid = (lo + hi) / 2u;
        let mid_key = slab[mid].key;
        if (mid_key < tgt) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    return lo;
}

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x > 0u) {
        return;
    }

    let len = slab_meta.len;
    if (len == 0u) {
        out_meta.start = 0u;
        out_meta.end = 0u;
        return;
    }

    let start = lower_bound(params.from_key, len);
    let end = lower_bound(params.to_key, len);

    out_meta.start = start;
    out_meta.end = end;
}
"#;
