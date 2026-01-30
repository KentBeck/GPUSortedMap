use bytemuck::{Pod, Zeroable};

use crate::{
    GpuMapError, GpuSortedMap, InputMeta, KvEntry, MergeMeta, ResultEntry, TOMBSTONE_VALUE,
};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug, Default)]
struct ScanParams {
    len: u32,
    _pad: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug, Default)]
struct RadixParams {
    len: u32,
    bit: u32,
    _pad: [u32; 2],
}

const SCAN_BLOCK_SIZE: u32 = 256;

pub(crate) const BULK_GET_WGSL: &str = r#"
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
@group(0) @binding(1) var<storage, read> slab_meta: SlabMeta;
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

pub(crate) const BULK_MERGE_WGSL: &str = r#"
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
@group(0) @binding(3) var<storage, read> slab_meta: SlabMeta;
@group(0) @binding(4) var<storage, read> input_meta: InputMeta;
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

pub(crate) const RADIX_SORT_WGSL: &str = r#"
struct KvEntry {
    key: u32,
    value: u32,
};

struct RadixParams {
    len: u32,
    bit: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read> input_entries: array<KvEntry>;
@group(0) @binding(1) var<storage, read_write> output_entries: array<KvEntry>;
@group(0) @binding(2) var<storage, read_write> flags: array<u32>;
@group(0) @binding(3) var<storage, read> prefix: array<u32>;
@group(0) @binding(4) var<storage, read> zero_count: array<u32>;
@group(0) @binding(5) var<uniform> params: RadixParams;

@compute @workgroup_size(256)
fn radix_flags(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.len) {
        return;
    }
    let key = input_entries[idx].key;
    let bit = (key >> params.bit) & 1u;
    flags[idx] = select(1u, 0u, bit == 1u);
}

@compute @workgroup_size(256)
fn radix_scatter(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.len) {
        return;
    }
    let zero_total = zero_count[0];
    let zero_flag = flags[idx];
    let pos = select(zero_total + (idx - prefix[idx]), prefix[idx], zero_flag == 1u);
    output_entries[pos] = input_entries[idx];
}
"#;

pub(crate) const SCAN_WGSL: &str = r#"
struct ScanParams {
    len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read> input_values: array<u32>;
@group(0) @binding(1) var<storage, read_write> output_values: array<u32>;
@group(0) @binding(2) var<storage, read_write> block_sums: array<u32>;
@group(0) @binding(3) var<storage, read> scanned_block_sums: array<u32>;
@group(0) @binding(4) var<uniform> params: ScanParams;

var<workgroup> temp: array<u32, 256>;

@compute @workgroup_size(256)
fn scan_block(@builtin(local_invocation_id) lid: vec3<u32>,
              @builtin(global_invocation_id) gid: vec3<u32>,
              @builtin(workgroup_id) wid: vec3<u32>) {
    let idx = gid.x;
    let local_idx = lid.x;
    var value: u32 = 0u;
    if (idx < params.len) {
        value = input_values[idx];
    }
    temp[local_idx] = value;
    workgroupBarrier();

    var offset: u32 = 1u;
    for (var d: u32 = 128u; d > 0u; d = d >> 1u) {
        workgroupBarrier();
        if (local_idx < d) {
            let ai = offset * (2u * local_idx + 1u) - 1u;
            let bi = offset * (2u * local_idx + 2u) - 1u;
            temp[bi] = temp[bi] + temp[ai];
        }
        offset = offset << 1u;
    }

    if (local_idx == 0u) {
        block_sums[wid.x] = temp[255];
        temp[255] = 0u;
    }
    workgroupBarrier();

    for (var d: u32 = 1u; d <= 128u; d = d << 1u) {
        offset = offset >> 1u;
        workgroupBarrier();
        if (local_idx < d) {
            let ai = offset * (2u * local_idx + 1u) - 1u;
            let bi = offset * (2u * local_idx + 2u) - 1u;
            let t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] = temp[bi] + t;
        }
    }
    workgroupBarrier();

    if (idx < params.len) {
        output_values[idx] = temp[local_idx];
    }
}

@compute @workgroup_size(256)
fn scan_add(@builtin(global_invocation_id) gid: vec3<u32>,
            @builtin(workgroup_id) wid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.len) {
        return;
    }
    if (wid.x == 0u) {
        return;
    }
    output_values[idx] = output_values[idx] + scanned_block_sums[wid.x];
}
"#;

pub(crate) const DEDUP_WGSL: &str = r#"
struct KvEntry {
    key: u32,
    value: u32,
};

struct ScanParams {
    len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read> input_entries: array<KvEntry>;
@group(0) @binding(1) var<storage, read_write> output_entries: array<KvEntry>;
@group(0) @binding(2) var<storage, read_write> flags: array<u32>;
@group(0) @binding(3) var<storage, read> prefix: array<u32>;
@group(0) @binding(4) var<uniform> params: ScanParams;

@compute @workgroup_size(256)
fn dedup_flags(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.len) {
        return;
    }
    let is_last = idx + 1u >= params.len;
    var keep: bool = is_last;
    if (!is_last) {
        keep = input_entries[idx].key != input_entries[idx + 1u].key;
    }
    flags[idx] = select(0u, 1u, keep);
}

@compute @workgroup_size(256)
fn dedup_compact(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.len) {
        return;
    }
    if (flags[idx] == 1u) {
        let out_idx = prefix[idx];
        output_entries[out_idx] = input_entries[idx];
    }
}
"#;

pub(crate) const COUNT_WGSL: &str = r#"
struct ScanParams {
    len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

struct InputMeta {
    len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read> flags: array<u32>;
@group(0) @binding(1) var<storage, read> prefix: array<u32>;
@group(0) @binding(2) var<storage, read_write> count: array<u32>;
@group(0) @binding(3) var<storage, read_write> input_meta: InputMeta;
@group(0) @binding(4) var<uniform> params: ScanParams;

@compute @workgroup_size(1)
fn total_count() {
    if (params.len == 0u) {
        count[0] = 0u;
        return;
    }
    let last = params.len - 1u;
    count[0] = prefix[last] + flags[last];
}

@compute @workgroup_size(1)
fn write_input_meta() {
    input_meta.len = count[0];
}
"#;

pub(crate) const META_WGSL: &str = r#"
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

@group(0) @binding(0) var<storage, read_write> slab_meta: SlabMeta;
@group(0) @binding(1) var<storage, read> input_meta: InputMeta;
@group(0) @binding(2) var<storage, read> merge_meta: MergeMeta;
@group(0) @binding(3) var<storage, read_write> error: array<u32>;

@compute @workgroup_size(1)
fn capacity_check() {
    let requested = slab_meta.len + input_meta.len;
    error[0] = select(0u, 1u, requested > slab_meta.capacity);
}

@compute @workgroup_size(1)
fn write_slab_meta() {
    slab_meta.len = merge_meta.len;
}
"#;

pub(crate) struct ScanLevel {
    pub(crate) block_sums: wgpu::Buffer,
    pub(crate) scanned_block_sums: wgpu::Buffer,
}

pub(crate) struct BulkPutOp {
    entries: Vec<KvEntry>,
}

impl BulkPutOp {
    pub(crate) fn new(entries: &[KvEntry]) -> Self {
        Self {
            entries: entries.to_vec(),
        }
    }

    pub(crate) fn execute(self, map: &mut GpuSortedMap) -> Result<(), GpuMapError> {
        if self.entries.is_empty() {
            return Ok(());
        }

        map.queue.write_buffer(
            &map.input_buffer,
            0,
            bytemuck::cast_slice(&self.entries),
        );
        let input_meta = InputMeta {
            len: self.entries.len() as u32,
            _pad: [0; 3],
        };
        map.queue
            .write_buffer(&map.input_meta_buffer, 0, bytemuck::bytes_of(&input_meta));

        let mut encoder = map
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("bulk-put-sort-encoder"),
            });

        let mut src = &map.input_buffer;
        let mut dst = &map.sort_buffer;
        let workgroups = ((input_meta.len + 255) / 256).max(1);

        for bit in 0..32u32 {
            let radix_params = RadixParams {
                len: input_meta.len,
                bit,
                _pad: [0; 2],
            };
            let radix_params_buffer =
                create_uniform_buffer(&map.device, "radix-params-buffer", &radix_params);
            let radix_bind_group = map.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("radix-bind-group"),
                layout: &map.bulk_put_pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: src.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: dst.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: map.flags_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: map.prefix_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: map.zero_count_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: radix_params_buffer.as_entire_binding(),
                    },
                ],
            });

            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("radix-flags-pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&map.bulk_put_pipeline);
                cpass.set_bind_group(0, &radix_bind_group, &[]);
                cpass.dispatch_workgroups(workgroups, 1, 1);
            }

            dispatch_scan(
                &map.device,
                &mut encoder,
                &map.scan_pipeline,
                &map.scan_add_pipeline,
                &map.flags_buffer,
                &map.prefix_buffer,
                &map.scan_levels,
                input_meta.len,
            );

            let scan_params = ScanParams {
                len: input_meta.len,
                _pad: [0; 3],
            };
            let scan_params_buffer =
                create_uniform_buffer(&map.device, "scan-params-buffer", &scan_params);
            let count_bind_group = map.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("radix-count-bind-group"),
                layout: &map.total_count_pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: map.flags_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: map.prefix_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: map.zero_count_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: map.input_meta_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: scan_params_buffer.as_entire_binding(),
                    },
                ],
            });
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("radix-count-pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&map.total_count_pipeline);
                cpass.set_bind_group(0, &count_bind_group, &[]);
                cpass.dispatch_workgroups(1, 1, 1);
            }

            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("radix-scatter-pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&map.scatter_pipeline);
                cpass.set_bind_group(0, &radix_bind_group, &[]);
                cpass.dispatch_workgroups(workgroups, 1, 1);
            }

            std::mem::swap(&mut src, &mut dst);
        }

        let scan_params = ScanParams {
            len: input_meta.len,
            _pad: [0; 3],
        };
        let scan_params_buffer =
            create_uniform_buffer(&map.device, "dedup-scan-params-buffer", &scan_params);
        let dedup_bind_group = map.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("dedup-bind-group"),
            layout: &map.dedup_flag_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: src.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: map.compact_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: map.flags_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: map.prefix_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: scan_params_buffer.as_entire_binding(),
                },
            ],
        });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("dedup-flags-pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&map.dedup_flag_pipeline);
            cpass.set_bind_group(0, &dedup_bind_group, &[]);
            cpass.dispatch_workgroups(workgroups, 1, 1);
        }

        dispatch_scan(
            &map.device,
            &mut encoder,
            &map.scan_pipeline,
            &map.scan_add_pipeline,
            &map.flags_buffer,
            &map.prefix_buffer,
            &map.scan_levels,
            input_meta.len,
        );

        let count_bind_group = map.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("dedup-count-bind-group"),
            layout: &map.total_count_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: map.flags_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: map.prefix_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: map.unique_count_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: map.input_meta_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: scan_params_buffer.as_entire_binding(),
                },
            ],
        });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("dedup-count-pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&map.total_count_pipeline);
            cpass.set_bind_group(0, &count_bind_group, &[]);
            cpass.dispatch_workgroups(1, 1, 1);
        }
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("write-input-meta-pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&map.write_input_meta_pipeline);
            cpass.set_bind_group(0, &count_bind_group, &[]);
            cpass.dispatch_workgroups(1, 1, 1);
        }
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("dedup-compact-pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&map.dedup_compact_pipeline);
            cpass.set_bind_group(0, &dedup_bind_group, &[]);
            cpass.dispatch_workgroups(workgroups, 1, 1);
        }

        let meta_bind_group = map.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bulk-put-meta-bind-group"),
            layout: &map.capacity_check_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: map.meta_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: map.input_meta_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: map.merge_meta_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: map.error_buffer.as_entire_binding(),
                },
            ],
        });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("capacity-check-pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&map.capacity_check_pipeline);
            cpass.set_bind_group(0, &meta_bind_group, &[]);
            cpass.dispatch_workgroups(1, 1, 1);
        }

        let error_readback = map.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bulk-put-error-readback"),
            size: std::mem::size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(
            &map.error_buffer,
            0,
            &error_readback,
            0,
            std::mem::size_of::<u32>() as u64,
        );
        map.queue.submit(Some(encoder.finish()));

        let error = readback_u32(&map.device, &error_readback);
        if error != 0 {
            let unique_readback = map.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("bulk-put-unique-readback"),
                size: std::mem::size_of::<u32>() as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut encoder = map
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("bulk-put-unique-readback-encoder"),
                });
            encoder.copy_buffer_to_buffer(
                &map.unique_count_buffer,
                0,
                &unique_readback,
                0,
                std::mem::size_of::<u32>() as u64,
            );
            map.queue.submit(Some(encoder.finish()));
            let unique = readback_u32(&map.device, &unique_readback);
            return Err(GpuMapError::CapacityExceeded {
                capacity: map.capacity,
                requested: map.len + unique,
            });
        }

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
                    resource: map.compact_buffer.as_entire_binding(),
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
                    resource: map.input_meta_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: map.merge_meta_buffer.as_entire_binding(),
                },
            ],
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
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("write-slab-meta-pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&map.write_slab_meta_pipeline);
            cpass.set_bind_group(0, &meta_bind_group, &[]);
            cpass.dispatch_workgroups(1, 1, 1);
        }
        map.queue.submit(Some(encoder.finish()));

        let merge_len = readback_merge_len(&map.device, &merge_readback);
        map.len = merge_len;
        Ok(())
    }
}

pub(crate) struct BulkGetOp {
    keys_len: usize,
    _keys_buffer: wgpu::Buffer,
    _keys_meta_buffer: wgpu::Buffer,
    results_buffer: wgpu::Buffer,
    readback_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    workgroups: u32,
}

impl BulkGetOp {
    pub(crate) fn new(map: &GpuSortedMap, keys: &[u32]) -> Self {
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
        let keys_meta = crate::KeysMeta {
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

    pub(crate) fn execute(self, map: &GpuSortedMap) -> Vec<Option<u32>> {
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

pub(crate) struct BulkDeleteOp {
    entries: Vec<KvEntry>,
}

impl BulkDeleteOp {
    pub(crate) fn new(keys: &[u32]) -> Self {
        let entries = keys
            .iter()
            .map(|&key| KvEntry {
                key,
                value: TOMBSTONE_VALUE,
            })
            .collect();
        Self { entries }
    }

    pub(crate) fn execute(self, map: &mut GpuSortedMap) -> Result<(), GpuMapError> {
        BulkPutOp::new(&self.entries).execute(map)
    }
}

pub(crate) fn create_scan_levels(device: &wgpu::Device, capacity: u32) -> Vec<ScanLevel> {
    let mut levels = Vec::new();
    let mut level_len = capacity;
    loop {
        let blocks = (level_len + SCAN_BLOCK_SIZE - 1) / SCAN_BLOCK_SIZE;
        if blocks == 0 {
            break;
        }
        let size = (blocks as u64) * std::mem::size_of::<u32>() as u64;
        let block_sums = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scan-block-sums"),
            size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let scanned_block_sums = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scan-scanned-block-sums"),
            size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        levels.push(ScanLevel {
            block_sums,
            scanned_block_sums,
        });
        if blocks <= 1 {
            break;
        }
        level_len = blocks;
    }
    levels
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

fn dispatch_scan(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    scan_pipeline: &wgpu::ComputePipeline,
    scan_add_pipeline: &wgpu::ComputePipeline,
    input: &wgpu::Buffer,
    output: &wgpu::Buffer,
    levels: &[ScanLevel],
    len: u32,
) {
    if len == 0 {
        return;
    }

    let original_input = input;
    let mut level_len = len;
    let mut input_buffer = input;
    let mut output_buffer = output;
    let mut level_info: Vec<(u32, u32, &wgpu::Buffer, &wgpu::Buffer, &wgpu::Buffer)> =
        Vec::new();

    for level in levels {
        let blocks = (level_len + SCAN_BLOCK_SIZE - 1) / SCAN_BLOCK_SIZE;
        if blocks == 0 {
            break;
        }
        let params = ScanParams {
            len: level_len,
            _pad: [0; 3],
        };
        let params_buffer = create_uniform_buffer(device, "scan-params-buffer", &params);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scan-bind-group"),
            layout: &scan_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: level.block_sums.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: level.scanned_block_sums.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("scan-block-pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(scan_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(blocks, 1, 1);
        }

        level_info.push((
            level_len,
            blocks,
            output_buffer,
            &level.block_sums,
            &level.scanned_block_sums,
        ));
        if blocks <= 1 {
            break;
        }
        level_len = blocks;
        input_buffer = &level.block_sums;
        output_buffer = &level.scanned_block_sums;
    }

    for (index, (level_len, blocks, output_buffer, block_sums, scanned_block_sums)) in
        level_info.iter().enumerate().rev()
    {
        if index == level_info.len() - 1 {
            continue;
        }
        let params = ScanParams {
            len: *level_len,
            _pad: [0; 3],
        };
        let params_buffer = create_uniform_buffer(device, "scan-add-params-buffer", &params);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scan-add-bind-group"),
            layout: &scan_add_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: original_input.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: block_sums.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: scanned_block_sums.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("scan-add-pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(scan_add_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(*blocks, 1, 1);
    }
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

fn readback_u32(device: &wgpu::Device, buffer: &wgpu::Buffer) -> u32 {
    let slice = buffer.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = sender.send(res);
    });
    device.poll(wgpu::Maintain::Wait);
    receiver.recv().expect("readback channel closed").unwrap();
    let data = slice.get_mapped_range();
    let values: &[u32] = bytemuck::cast_slice(&data);
    let value = values.first().copied().unwrap_or(0);
    drop(data);
    buffer.unmap();
    value
}
