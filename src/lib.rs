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

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug, Default)]
struct ScanParams {
    pub len: u32,
    pub _pad: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug, Default)]
struct RadixParams {
    pub len: u32,
    pub bit: u32,
    pub _pad: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug, Default)]
struct ScatterParams {
    pub len: u32,
    pub zero_count: u32,
    pub _pad: [u32; 2],
}

const SCAN_BLOCK_SIZE: u32 = 256;

struct ScanLevel {
    block_sums: wgpu::Buffer,
    scanned_block_sums: wgpu::Buffer,
}

pub struct GpuSortedMap {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub slab_buffer: wgpu::Buffer,
    pub input_buffer: wgpu::Buffer,
    pub meta_buffer: wgpu::Buffer,
    input_meta_buffer: wgpu::Buffer,
    sort_buffer: wgpu::Buffer,
    compact_buffer: wgpu::Buffer,
    flags_buffer: wgpu::Buffer,
    prefix_buffer: wgpu::Buffer,
    zero_count_buffer: wgpu::Buffer,
    unique_count_buffer: wgpu::Buffer,
    error_buffer: wgpu::Buffer,
    scan_levels: Vec<ScanLevel>,
    merge_buffer: wgpu::Buffer,
    merge_meta_buffer: wgpu::Buffer,
    bulk_get_pipeline: wgpu::ComputePipeline,
    bulk_get_bind_group_layout: wgpu::BindGroupLayout,
    bulk_merge_pipeline: wgpu::ComputePipeline,
    bulk_merge_bind_group_layout: wgpu::BindGroupLayout,
    bulk_put_pipeline: wgpu::ComputePipeline,
    scan_pipeline: wgpu::ComputePipeline,
    scan_add_pipeline: wgpu::ComputePipeline,
    scatter_pipeline: wgpu::ComputePipeline,
    dedup_flag_pipeline: wgpu::ComputePipeline,
    dedup_compact_pipeline: wgpu::ComputePipeline,
    total_count_pipeline: wgpu::ComputePipeline,
    write_input_meta_pipeline: wgpu::ComputePipeline,
    capacity_check_pipeline: wgpu::ComputePipeline,
    write_slab_meta_pipeline: wgpu::ComputePipeline,
    capacity: u32,
    len: u32,
}

const TOMBSTONE_VALUE: u32 = u32::MAX;

struct BulkPutOp {
    entries: Vec<KvEntry>,
}

impl BulkPutOp {
    fn new(entries: &[KvEntry]) -> Self {
        Self {
            entries: entries.to_vec(),
        }
    }

    fn execute(self, map: &mut GpuSortedMap) -> Result<(), GpuMapError> {
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
        BulkPutOp::new(&self.entries).execute(map)
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
        if adapter.get_info().device_type == wgpu::DeviceType::Cpu {
            panic!("fallback adapter detected; aborting without CPU fallback");
        }

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
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: true,
        });
        {
            let mut view = meta_buffer.slice(..).get_mapped_range_mut();
            view.copy_from_slice(bytemuck::bytes_of(&meta));
        }
        meta_buffer.unmap();

        let input_meta_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("input-meta-buffer"),
            size: std::mem::size_of::<InputMeta>() as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let sort_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sort-buffer"),
            size: slab_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let compact_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("compact-buffer"),
            size: slab_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let u32_buffer_size = (capacity as u64) * std::mem::size_of::<u32>() as u64;
        let flags_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flags-buffer"),
            size: u32_buffer_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let prefix_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("prefix-buffer"),
            size: u32_buffer_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let zero_count_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("zero-count-buffer"),
            size: std::mem::size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let unique_count_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("unique-count-buffer"),
            size: std::mem::size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let error_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bulk-put-error-buffer"),
            size: std::mem::size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let scan_levels = create_scan_levels(&device, capacity);

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
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
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

        let radix_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("radix-sort-shader"),
            source: wgpu::ShaderSource::Wgsl(RADIX_SORT_WGSL.into()),
        });
        let radix_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("radix-bind-group-layout"),
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
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
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
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let radix_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("radix-pipeline-layout"),
                bind_group_layouts: &[&radix_bind_group_layout],
                push_constant_ranges: &[],
            });
        let bulk_put_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("radix-flags-pipeline"),
            layout: Some(&radix_pipeline_layout),
            module: &radix_shader,
            entry_point: "radix_flags",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });
        let scatter_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("radix-scatter-pipeline"),
            layout: Some(&radix_pipeline_layout),
            module: &radix_shader,
            entry_point: "radix_scatter",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        let scan_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("scan-shader"),
            source: wgpu::ShaderSource::Wgsl(SCAN_WGSL.into()),
        });
        let scan_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("scan-bind-group-layout"),
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
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
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
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
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
                ],
            });
        let scan_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("scan-pipeline-layout"),
            bind_group_layouts: &[&scan_bind_group_layout],
            push_constant_ranges: &[],
        });
        let scan_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("scan-block-pipeline"),
            layout: Some(&scan_pipeline_layout),
            module: &scan_shader,
            entry_point: "scan_block",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });
        let scan_add_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("scan-add-pipeline"),
            layout: Some(&scan_pipeline_layout),
            module: &scan_shader,
            entry_point: "scan_add",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        let dedup_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("dedup-shader"),
            source: wgpu::ShaderSource::Wgsl(DEDUP_WGSL.into()),
        });
        let dedup_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("dedup-bind-group-layout"),
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
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
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
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
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
                ],
            });
        let dedup_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("dedup-pipeline-layout"),
                bind_group_layouts: &[&dedup_bind_group_layout],
                push_constant_ranges: &[],
            });
        let dedup_flag_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("dedup-flag-pipeline"),
            layout: Some(&dedup_pipeline_layout),
            module: &dedup_shader,
            entry_point: "dedup_flags",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });
        let dedup_compact_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("dedup-compact-pipeline"),
                layout: Some(&dedup_pipeline_layout),
                module: &dedup_shader,
                entry_point: "dedup_compact",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            });

        let count_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("count-shader"),
            source: wgpu::ShaderSource::Wgsl(COUNT_WGSL.into()),
        });
        let count_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("count-bind-group-layout"),
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
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
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
                ],
            });
        let count_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("count-pipeline-layout"),
                bind_group_layouts: &[&count_bind_group_layout],
                push_constant_ranges: &[],
            });
        let total_count_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("total-count-pipeline"),
            layout: Some(&count_pipeline_layout),
            module: &count_shader,
            entry_point: "total_count",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });
        let write_input_meta_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("write-input-meta-pipeline"),
                layout: Some(&count_pipeline_layout),
                module: &count_shader,
                entry_point: "write_input_meta",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            });

        let meta_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("meta-shader"),
            source: wgpu::ShaderSource::Wgsl(META_WGSL.into()),
        });
        let meta_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("meta-bind-group-layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
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
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let meta_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("meta-pipeline-layout"),
                bind_group_layouts: &[&meta_bind_group_layout],
                push_constant_ranges: &[],
            });
        let capacity_check_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("capacity-check-pipeline"),
                layout: Some(&meta_pipeline_layout),
                module: &meta_shader,
                entry_point: "capacity_check",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            });
        let write_slab_meta_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("write-slab-meta-pipeline"),
                layout: Some(&meta_pipeline_layout),
                module: &meta_shader,
                entry_point: "write_slab_meta",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            });

        Ok(Self {
            device,
            queue,
            slab_buffer,
            input_buffer,
            meta_buffer,
            input_meta_buffer,
            sort_buffer,
            compact_buffer,
            flags_buffer,
            prefix_buffer,
            zero_count_buffer,
            unique_count_buffer,
            error_buffer,
            scan_levels,
            merge_buffer,
            merge_meta_buffer,
            bulk_get_pipeline,
            bulk_get_bind_group_layout,
            bulk_merge_pipeline,
            bulk_merge_bind_group_layout,
            bulk_put_pipeline,
            scan_pipeline,
            scan_add_pipeline,
            scatter_pipeline,
            dedup_flag_pipeline,
            dedup_compact_pipeline,
            total_count_pipeline,
            write_input_meta_pipeline,
            capacity_check_pipeline,
            write_slab_meta_pipeline,
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

        BulkPutOp::new(entries).execute(self)
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

const RADIX_SORT_WGSL: &str = r#"
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

const SCAN_WGSL: &str = r#"
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

const DEDUP_WGSL: &str = r#"
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

const COUNT_WGSL: &str = r#"
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

const META_WGSL: &str = r#"
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

fn create_scan_levels(device: &wgpu::Device, capacity: u32) -> Vec<ScanLevel> {
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
