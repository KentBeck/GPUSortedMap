use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug, Default)]
pub struct KvEntry {
    pub key: u32,
    pub value: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug, Default)]
pub struct ResultEntry {
    pub value: u32,
    pub found: u32,
    pub _pad: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug, Default)]
pub struct SlabMeta {
    pub len: u32,
    pub capacity: u32,
    pub _pad: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug, Default)]
pub struct KeysMeta {
    pub len: u32,
    pub _pad: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug, Default)]
pub struct InputMeta {
    pub len: u32,
    pub _pad: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug, Default)]
pub struct MergeMeta {
    pub len: u32,
    pub _pad: [u32; 3],
}

pub struct GpuSortedMap {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub slab_buffer: wgpu::Buffer,
    pub input_buffer: wgpu::Buffer,
    pub meta_buffer: wgpu::Buffer,
    merge_buffer: wgpu::Buffer,
    merge_meta_buffer: wgpu::Buffer,
    bulk_get_pipeline: wgpu::ComputePipeline,
    bulk_get_bind_group_layout: wgpu::BindGroupLayout,
    bulk_merge_pipeline: wgpu::ComputePipeline,
    bulk_merge_bind_group_layout: wgpu::BindGroupLayout,
    capacity: u32,
    len: u32,
}

const TOMBSTONE_VALUE: u32 = u32::MAX;

struct BulkPutOp {
    incoming: Vec<KvEntry>,
}

impl BulkPutOp {
    fn new(map: &GpuSortedMap, entries: &[KvEntry]) -> Result<Self, GpuMapError> {
        // TODO: Decide whether bulk_put should accept duplicate keys (e.g., last-write-wins).
        let mut incoming_map = std::collections::BTreeMap::new();
        for entry in entries {
            incoming_map.insert(entry.key, entry.value);
        }
        let incoming: Vec<KvEntry> = incoming_map
            .into_iter()
            .map(|(key, value)| KvEntry { key, value })
            .collect();

        let requested = map.len + incoming.len() as u32;
        if requested > map.capacity {
            return Err(GpuMapError::CapacityExceeded {
                capacity: map.capacity,
                requested,
            });
        }

        Ok(Self { incoming })
    }

    fn execute(self, map: &mut GpuSortedMap) -> Result<(), GpuMapError> {
        if !self.incoming.is_empty() {
            map.queue.write_buffer(
                &map.input_buffer,
                0,
                bytemuck::cast_slice(&self.incoming),
            );
        }
        let input_meta = InputMeta {
            len: self.incoming.len() as u32,
            _pad: [0; 3],
        };
        let input_meta_buffer =
            create_uniform_buffer(&map.device, "input-meta-buffer", &input_meta);

        let merge_bind_group = map.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bulk-merge-bind-group"),
            layout: &map.bulk_merge_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: map.slab_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: map.input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: map.merge_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: map.meta_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: input_meta_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: map.merge_meta_buffer.as_entire_binding(),
                },
            ],
        });

        let merge_readback = map.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("merge-meta-readback"),
            size: std::mem::size_of::<MergeMeta>() as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = map
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("bulk-merge-encoder"),
            });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("bulk-merge-pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&map.bulk_merge_pipeline);
            cpass.set_bind_group(0, &merge_bind_group, &[]);
            cpass.dispatch_workgroups(1, 1, 1);
        }
        let slab_bytes = (map.capacity as u64) * std::mem::size_of::<KvEntry>() as u64;
        encoder.copy_buffer_to_buffer(&map.merge_buffer, 0, &map.slab_buffer, 0, slab_bytes);
        encoder.copy_buffer_to_buffer(
            &map.merge_meta_buffer,
            0,
            &merge_readback,
            0,
            std::mem::size_of::<MergeMeta>() as u64,
        );
        map.queue.submit(Some(encoder.finish()));

        let merge_len = readback_merge_len(&map.device, &merge_readback);
        map.update_len(merge_len);
        Ok(())
    }
}

struct BulkGetOp {
    keys_len: usize,
    _keys_buffer: wgpu::Buffer,
    _keys_meta_buffer: wgpu::Buffer,
    results_buffer: wgpu::Buffer,
    readback_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    workgroups: u32,
}

impl BulkGetOp {
    fn new(map: &GpuSortedMap, keys: &[u32]) -> Self {
        let keys_len = keys.len();
        let keys_buffer = create_buffer_with_data(
            &map.device,
            "keys-buffer",
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            keys,
        );
        let results_buffer = map.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("results-buffer"),
            size: (keys_len * std::mem::size_of::<ResultEntry>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback_buffer = map.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("results-readback-buffer"),
            size: (keys_len * std::mem::size_of::<ResultEntry>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let keys_meta = KeysMeta {
            len: keys_len as u32,
            _pad: [0; 3],
        };
        let keys_meta_buffer = create_uniform_buffer(&map.device, "keys-meta-buffer", &keys_meta);

        let bind_group = map.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bulk-get-bind-group"),
            layout: &map.bulk_get_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: map.slab_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: map.meta_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: keys_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: keys_meta_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: results_buffer.as_entire_binding(),
                },
            ],
        });

        let workgroups = ((keys_len as u32) + 63) / 64;

        Self {
            keys_len,
            _keys_buffer: keys_buffer,
            _keys_meta_buffer: keys_meta_buffer,
            results_buffer,
            readback_buffer,
            bind_group,
            workgroups,
        }
    }

    fn execute(self, map: &GpuSortedMap) -> Vec<Option<u32>> {
        let mut encoder = map
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("bulk-get-encoder"),
            });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("bulk-get-pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&map.bulk_get_pipeline);
            cpass.set_bind_group(0, &self.bind_group, &[]);
            cpass.dispatch_workgroups(self.workgroups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(
            &self.results_buffer,
            0,
            &self.readback_buffer,
            0,
            (self.keys_len * std::mem::size_of::<ResultEntry>()) as u64,
        );
        map.queue.submit(Some(encoder.finish()));

        let result_entries = readback_results(&map.device, &self.readback_buffer);
        result_entries
            .iter()
            .map(|entry| if entry.found == 0 { None } else { Some(entry.value) })
            .collect()
    }
}

struct BulkDeleteOp {
    entries: Vec<KvEntry>,
}

impl BulkDeleteOp {
    fn new(keys: &[u32]) -> Self {
        let entries = keys
            .iter()
            .map(|&key| KvEntry {
                key,
                value: TOMBSTONE_VALUE,
            })
            .collect();
        Self { entries }
    }

    fn execute(self, map: &mut GpuSortedMap) -> Result<(), GpuMapError> {
        BulkPutOp::new(map, &self.entries)?.execute(map)
    }
}

impl GpuSortedMap {
    pub async fn new(capacity: u32) -> Result<Self, wgpu::RequestDeviceError> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .expect("no suitable GPU adapters found");

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("gpu-sorted-map-device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                },
                None,
            )
            .await?;

        let slab_size = (capacity as u64) * std::mem::size_of::<KvEntry>() as u64;
        let slab_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("slab-buffer"),
            size: slab_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let input_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("input-buffer"),
            size: slab_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let meta = SlabMeta {
            len: 0,
            capacity,
            _pad: [0; 2],
        };
        let meta_size = std::mem::size_of::<SlabMeta>() as u64;
        let meta_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("meta-buffer"),
            size: meta_size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        {
            let mut view = meta_buffer.slice(..).get_mapped_range_mut();
            view.copy_from_slice(bytemuck::bytes_of(&meta));
        }
        meta_buffer.unmap();

        let merge_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("merge-buffer"),
            size: slab_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let merge_meta_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("merge-meta-buffer"),
            size: std::mem::size_of::<MergeMeta>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bulk_get_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bulk-get-shader"),
            source: wgpu::ShaderSource::Wgsl(BULK_GET_WGSL.into()),
        });
        let bulk_get_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("bulk-get-bind-group-layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let bulk_get_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("bulk-get-pipeline-layout"),
                bind_group_layouts: &[&bulk_get_bind_group_layout],
                push_constant_ranges: &[],
            });
        let bulk_get_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("bulk-get-pipeline"),
            layout: Some(&bulk_get_pipeline_layout),
            module: &bulk_get_shader,
            entry_point: "main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        let bulk_merge_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bulk-merge-shader"),
            source: wgpu::ShaderSource::Wgsl(BULK_MERGE_WGSL.into()),
        });
        let bulk_merge_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("bulk-merge-bind-group-layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let bulk_merge_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("bulk-merge-pipeline-layout"),
                bind_group_layouts: &[&bulk_merge_bind_group_layout],
                push_constant_ranges: &[],
            });
        let bulk_merge_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("bulk-merge-pipeline"),
                layout: Some(&bulk_merge_pipeline_layout),
                module: &bulk_merge_shader,
                entry_point: "main",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            });

        Ok(Self {
            device,
            queue,
            slab_buffer,
            input_buffer,
            meta_buffer,
            merge_buffer,
            merge_meta_buffer,
            bulk_get_pipeline,
            bulk_get_bind_group_layout,
            bulk_merge_pipeline,
            bulk_merge_bind_group_layout,
            capacity,
            len: 0,
        })
    }

    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    pub fn len(&self) -> u32 {
        self.len
    }

    pub fn update_len(&mut self, new_len: u32) {
        self.len = new_len.min(self.capacity);
        let meta = SlabMeta {
            len: self.len,
            capacity: self.capacity,
            _pad: [0; 2],
        };
        self.queue
            .write_buffer(&self.meta_buffer, 0, bytemuck::bytes_of(&meta));
    }

    pub fn bulk_put(&mut self, entries: &[KvEntry]) -> Result<(), GpuMapError> {
        if entries.is_empty() {
            return Ok(());
        }

        BulkPutOp::new(self, entries)?.execute(self)
    }

    pub fn bulk_get(&self, keys: &[u32]) -> Vec<Option<u32>> {
        if keys.is_empty() {
            return Vec::new();
        }
        BulkGetOp::new(self, keys).execute(self)
    }

    pub fn put(&mut self, key: u32, value: u32) -> Result<(), GpuMapError> {
        let entry = KvEntry { key, value };
        self.bulk_put(std::slice::from_ref(&entry))
    }

    pub fn get(&self, key: u32) -> Option<u32> {
        self.bulk_get(&[key]).into_iter().next().unwrap_or(None)
    }

    pub fn bulk_delete(&mut self, keys: &[u32]) -> Result<(), GpuMapError> {
        if keys.is_empty() {
            return Ok(());
        }
        BulkDeleteOp::new(keys).execute(self)
    }

    pub fn delete(&mut self, key: u32) -> Result<(), GpuMapError> {
        self.bulk_delete(&[key])
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuMapError {
    CapacityExceeded { capacity: u32, requested: u32 },
}

const BULK_GET_WGSL: &str = r#"
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

struct ResultEntry {
    value: u32,
    found: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read> slab: array<KvEntry>;
@group(0) @binding(1) var<uniform> slab_meta: SlabMeta;
@group(0) @binding(2) var<storage, read> keys: array<u32>;
@group(0) @binding(3) var<uniform> keys_meta: KeysMeta;
@group(0) @binding(4) var<storage, read_write> results: array<ResultEntry>;

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
        let value = slab[lo].value;
        if (value == 0xffffffffu) {
            results[idx].value = 0u;
            results[idx].found = 0u;
        } else {
            results[idx].value = value;
            results[idx].found = 1u;
        }
    } else {
        results[idx].value = 0u;
        results[idx].found = 0u;
    }
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

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let slab_len = slab_meta.len;
    let input_len = input_meta.len;
    let total_len = slab_len + input_len;
    if (total_len == 0u) {
        if (gid.x == 0u) {
            merge_meta.len = 0u;
        }
        return;
    }

    let chunk_size: u32 = 256u;
    let chunk_count = (total_len + chunk_size - 1u) / chunk_size;
    let chunk_index = gid.x;
    if (chunk_index >= chunk_count) {
        return;
    }

    let k0 = chunk_index * chunk_size;
    var k1 = k0 + chunk_size;
    if (k1 > total_len) {
        k1 = total_len;
    }

    let start = merge_partition(k0, slab_len, input_len);
    let end = merge_partition(k1, slab_len, input_len);

    var i = start.x;
    var j = start.y;
    var k = k0;

    while (k < k1) {
        if (i < end.x && j < end.y) {
            let a = slab[i];
            let b = input[j];
            if (a.key < b.key) {
                output[k] = a;
                i = i + 1u;
            } else {
                output[k] = b;
                j = j + 1u;
            }
        } else if (i < end.x) {
            output[k] = slab[i];
            i = i + 1u;
        } else {
            output[k] = input[j];
            j = j + 1u;
        }
        k = k + 1u;
    }

    if (chunk_index == 0u) {
        merge_meta.len = total_len;
    }
}
"#;

fn create_buffer_with_data<T: Pod>(
    device: &wgpu::Device,
    label: &str,
    usage: wgpu::BufferUsages,
    data: &[T],
) -> wgpu::Buffer {
    let size = (data.len() * std::mem::size_of::<T>()) as u64;
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage,
        mapped_at_creation: true,
    });
    if !data.is_empty() {
        let mut view = buffer.slice(..).get_mapped_range_mut();
        view.copy_from_slice(bytemuck::cast_slice(data));
    }
    buffer.unmap();
    buffer
}

fn create_uniform_buffer<T: Pod>(device: &wgpu::Device, label: &str, value: &T) -> wgpu::Buffer {
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: std::mem::size_of::<T>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: true,
    });
    {
        let mut view = buffer.slice(..).get_mapped_range_mut();
        view.copy_from_slice(bytemuck::bytes_of(value));
    }
    buffer.unmap();
    buffer
}

fn readback_results(device: &wgpu::Device, buffer: &wgpu::Buffer) -> Vec<ResultEntry> {
    let slice = buffer.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = sender.send(res);
    });
    device.poll(wgpu::Maintain::Wait);
    receiver.recv().expect("readback channel closed").unwrap();
    let data = slice.get_mapped_range();
    let results = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    buffer.unmap();
    results
}

fn readback_merge_len(device: &wgpu::Device, buffer: &wgpu::Buffer) -> u32 {
    let slice = buffer.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = sender.send(res);
    });
    device.poll(wgpu::Maintain::Wait);
    receiver.recv().expect("readback channel closed").unwrap();
    let data = slice.get_mapped_range();
    let meta: &[MergeMeta] = bytemuck::cast_slice(&data);
    let len = meta.first().map(|m| m.len).unwrap_or(0);
    drop(data);
    buffer.unmap();
    len
}

#[cfg(test)]
mod tests {
    use super::{GpuSortedMap, KvEntry};

    #[test]
    fn creates_gpu_sorted_map() {
        let map = pollster::block_on(GpuSortedMap::new(1024));
        assert!(map.is_ok(), "GpuSortedMap::new should succeed");
    }

    #[test]
    fn put_then_get() {
        let mut map = pollster::block_on(GpuSortedMap::new(8)).unwrap();
        let entries = [
            KvEntry { key: 42, value: 7 },
            KvEntry { key: 7, value: 9 },
            KvEntry { key: 13, value: 1 },
        ];
        map.bulk_put(&entries).unwrap();

        let results = map.bulk_get(&[7, 13, 42, 99]);
        assert_eq!(results, vec![Some(9), Some(1), Some(7), None]);
    }

    #[test]
    fn single_put_then_get() {
        let mut map = pollster::block_on(GpuSortedMap::new(4)).unwrap();
        map.put(5, 11).unwrap();
        assert_eq!(map.get(5), Some(11));
    }

    #[test]
    fn single_get_missing_key() {
        let map = pollster::block_on(GpuSortedMap::new(4)).unwrap();
        assert_eq!(map.get(9), None);
    }

    #[test]
    fn delete_existing_key() {
        let mut map = pollster::block_on(GpuSortedMap::new(8)).unwrap();
        let entries = [
            KvEntry { key: 1, value: 10 },
            KvEntry { key: 2, value: 20 },
        ];
        map.bulk_put(&entries).unwrap();

        map.delete(1).unwrap();
        assert_eq!(map.get(1), None);
        assert_eq!(map.get(2), Some(20));
    }

    #[test]
    fn bulk_delete_mixed_keys() {
        let mut map = pollster::block_on(GpuSortedMap::new(8)).unwrap();
        let entries = [
            KvEntry { key: 5, value: 50 },
            KvEntry { key: 7, value: 70 },
            KvEntry { key: 9, value: 90 },
        ];
        map.bulk_put(&entries).unwrap();

        map.bulk_delete(&[7, 7, 99]).unwrap();
        let results = map.bulk_get(&[5, 7, 9, 99]);
        assert_eq!(results, vec![Some(50), None, Some(90), None]);
    }

    #[test]
    fn delete_then_put_restores_value() {
        let mut map = pollster::block_on(GpuSortedMap::new(8)).unwrap();
        map.put(3, 30).unwrap();
        map.delete(3).unwrap();
        assert_eq!(map.get(3), None);

        map.put(3, 31).unwrap();
        assert_eq!(map.get(3), Some(31));
    }

    #[test]
    fn bulk_delete_empty_is_noop() {
        let mut map = pollster::block_on(GpuSortedMap::new(4)).unwrap();
        map.bulk_delete(&[]).unwrap();
        assert_eq!(map.get(1), None);
    }
}
