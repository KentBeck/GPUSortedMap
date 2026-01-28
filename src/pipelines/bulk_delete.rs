use std::sync::Arc;

use crate::gpu_array::GpuArray;
use crate::pipelines::core::ComputeStep;
use crate::pipelines::data::KeysMeta;
use crate::pipelines::utils::create_buffer_with_data;
use crate::KvEntry;

const BULK_DELETE_BIND_SLAB: u32 = 0;
const BULK_DELETE_BIND_SLAB_META: u32 = 1;
const BULK_DELETE_BIND_KEYS: u32 = 2;
const BULK_DELETE_BIND_KEYS_META: u32 = 3;

pub struct BulkDeletePipeline {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    step: ComputeStep,
}

impl BulkDeletePipeline {
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        let step = ComputeStep::new(
            Arc::clone(&device),
            BULK_DELETE_WGSL,
            "main",
            &[
                wgpu::BindGroupLayoutEntry {
                    binding: BULK_DELETE_BIND_SLAB,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: BULK_DELETE_BIND_SLAB_META,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: BULK_DELETE_BIND_KEYS,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: BULK_DELETE_BIND_KEYS_META,
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
        Self {
            device,
            queue,
            step,
        }
    }

    pub fn execute(&self, slab: &GpuArray<KvEntry>, keys: &[u32]) {
        if keys.is_empty() {
            return;
        }

        let keys_buffer = create_buffer_with_data(
            &self.device,
            "delete-keys-buffer",
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            keys,
        );
        let keys_meta = KeysMeta {
            len: keys.len() as u32,
            _pad: [0; 3],
        };
        let keys_meta_buffer = create_buffer_with_data(
            &self.device,
            "delete-keys-meta-buffer",
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            &[keys_meta],
        );

        let bind_group = self.step.create_bind_group(
            "bulk-delete-bind-group",
            &[
                wgpu::BindGroupEntry {
                    binding: BULK_DELETE_BIND_SLAB,
                    resource: slab.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: BULK_DELETE_BIND_SLAB_META,
                    resource: slab.meta_buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: BULK_DELETE_BIND_KEYS,
                    resource: keys_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: BULK_DELETE_BIND_KEYS_META,
                    resource: keys_meta_buffer.as_entire_binding(),
                },
            ],
        );

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("bulk-delete-encoder"),
            });

        let workgroups = ((keys.len() as u32) + 63) / 64;
        self.step.dispatch(
            &mut encoder,
            "bulk-delete-pass",
            &bind_group,
            (workgroups, 1, 1),
        );

        self.queue.submit(Some(encoder.finish()));
    }
}

const BULK_DELETE_WGSL: &str = r#"
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

struct KeysMeta {
    len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read_write> slab: array<KvEntry>;
@group(0) @binding(1) var<uniform> slab_meta: SlabMeta;
@group(0) @binding(2) var<storage, read> keys: array<u32>;
@group(0) @binding(3) var<uniform> keys_meta: KeysMeta;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= keys_meta.len) {
        return;
    }

    let key = keys[idx];
    var lo: u32 = 0u;
    var hi: u32 = slab_meta.len;
    while (lo < hi) {
        let mid = (lo + hi) / 2u;
        let mid_key = slab[mid].key;
        if (mid_key < key) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }

    if (lo < slab_meta.len && slab[lo].key == key) {
        slab[lo].value = 0xffffffffu;
    }
}
"#;
