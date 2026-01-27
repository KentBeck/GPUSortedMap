use bytemuck::{Pod, Zeroable};
use std::marker::PhantomData;

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

pub struct GpuArray<T: Pod> {
    buffer: wgpu::Buffer,
    meta_buffer: wgpu::Buffer,
    capacity: u32,
    len: u32,
    _marker: PhantomData<T>,
}

pub struct GpuStorage<T: Pod> {
    buffer: wgpu::Buffer,
    _marker: PhantomData<T>,
}

impl<T: Pod> GpuStorage<T> {
    pub fn new(device: &wgpu::Device, usage: wgpu::BufferUsages, label: &str) -> Self {
        let size = std::mem::size_of::<T>() as u64;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage,
            mapped_at_creation: false,
        });
        Self {
            buffer,
            _marker: PhantomData,
        }
    }

    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }
}

impl<T: Pod> GpuArray<T> {
    pub fn new(
        device: &wgpu::Device,
        capacity: u32,
        buffer_usage: wgpu::BufferUsages,
        label: &str,
    ) -> Self {
        let size = (capacity as u64) * std::mem::size_of::<T>() as u64;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: buffer_usage,
            mapped_at_creation: false,
        });

        let meta = SlabMeta {
            len: 0,
            capacity,
            _pad: [0; 2],
        };
        let meta_size = std::mem::size_of::<SlabMeta>() as u64;
        let meta_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("slab-meta-buffer"),
            size: meta_size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        {
            let mut view = meta_buffer.slice(..).get_mapped_range_mut();
            view.copy_from_slice(bytemuck::bytes_of(&meta));
        }
        meta_buffer.unmap();

        Self {
            buffer,
            meta_buffer,
            capacity,
            len: 0,
            _marker: PhantomData,
        }
    }

    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    pub fn meta_buffer(&self) -> &wgpu::Buffer {
        &self.meta_buffer
    }

    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    pub fn len(&self) -> u32 {
        self.len
    }

    pub fn update_len(&mut self, queue: &wgpu::Queue, new_len: u32) {
        self.len = new_len.min(self.capacity);
        let meta = SlabMeta {
            len: self.len,
            capacity: self.capacity,
            _pad: [0; 2],
        };
        queue.write_buffer(&self.meta_buffer, 0, bytemuck::bytes_of(&meta));
    }

    pub fn write(&self, queue: &wgpu::Queue, data: &[T]) {
        if data.is_empty() {
            return;
        }
        queue.write_buffer(&self.buffer, 0, bytemuck::cast_slice(data));
    }
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
pub struct SortParams {
    pub k: u32,
    pub j: u32,
    pub len: u32,
    pub _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug, Default)]
pub struct DedupParams {
    pub len: u32,
    pub _pad: [u32; 3],
}

pub struct GpuSortedMap {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub slab: GpuArray<KvEntry>,
    pub input: GpuArray<KvEntry>,
    merge: GpuArray<KvEntry>,
    merge_meta: GpuStorage<MergeMeta>,
    bulk_get: BulkGetPipeline,
    bulk_delete: BulkDeletePipeline,
    bulk_put: BulkPutPipeline,
}

pub struct BulkGetPipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

pub struct BulkDeletePipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

pub struct BulkPutPipeline {
    sort_pipeline: wgpu::ComputePipeline,
    sort_bind_group_layout: wgpu::BindGroupLayout,
    dedup_pipeline: wgpu::ComputePipeline,
    dedup_bind_group_layout: wgpu::BindGroupLayout,
    merge_pipeline: wgpu::ComputePipeline,
    merge_bind_group_layout: wgpu::BindGroupLayout,
}
impl BulkGetPipeline {
    pub fn new(device: &wgpu::Device) -> Self {
        let bulk_get_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bulk-get-shader"),
            source: wgpu::ShaderSource::Wgsl(BULK_GET_WGSL.into()),
        });
        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("bulk-get-bind-group-layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: BULK_GET_BIND_SLAB,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: BULK_GET_BIND_SLAB_META,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: BULK_GET_BIND_KEYS,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: BULK_GET_BIND_KEYS_META,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: BULK_GET_BIND_RESULTS,
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
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("bulk-get-pipeline-layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("bulk-get-pipeline"),
            layout: Some(&pipeline_layout),
            module: &bulk_get_shader,
            entry_point: "main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        Self {
            pipeline,
            bind_group_layout,
        }
    }

    pub fn bind_group(
        &self,
        device: &wgpu::Device,
        slab: &wgpu::Buffer,
        slab_meta: &wgpu::Buffer,
        keys: &wgpu::Buffer,
        keys_meta: &wgpu::Buffer,
        results: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bulk-get-bind-group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: BULK_GET_BIND_SLAB,
                    resource: slab.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: BULK_GET_BIND_SLAB_META,
                    resource: slab_meta.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: BULK_GET_BIND_KEYS,
                    resource: keys.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: BULK_GET_BIND_KEYS_META,
                    resource: keys_meta.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: BULK_GET_BIND_RESULTS,
                    resource: results.as_entire_binding(),
                },
            ],
        })
    }

    pub fn execute(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        slab: &GpuArray<KvEntry>,
        keys: &[u32],
    ) -> Vec<Option<u32>> {
        if keys.is_empty() {
            return Vec::new();
        }

        let keys_buffer = create_buffer_with_data(
            device,
            "keys-buffer",
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            keys,
        );
        let results_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("results-buffer"),
            size: (keys.len() * std::mem::size_of::<ResultEntry>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("results-readback-buffer"),
            size: (keys.len() * std::mem::size_of::<ResultEntry>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let keys_meta = KeysMeta {
            len: keys.len() as u32,
            _pad: [0; 3],
        };
        let keys_meta_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("keys-meta-buffer"),
            size: std::mem::size_of::<KeysMeta>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        {
            let mut view = keys_meta_buffer.slice(..).get_mapped_range_mut();
            view.copy_from_slice(bytemuck::bytes_of(&keys_meta));
        }
        keys_meta_buffer.unmap();

        let bind_group = self.bind_group(
            device,
            slab.buffer(),
            slab.meta_buffer(),
            &keys_buffer,
            &keys_meta_buffer,
            &results_buffer,
        );

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("bulk-get-encoder"),
        });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("bulk-get-pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            let workgroups = ((keys.len() as u32) + 63) / 64;
            cpass.dispatch_workgroups(workgroups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(
            &results_buffer,
            0,
            &readback_buffer,
            0,
            (keys.len() * std::mem::size_of::<ResultEntry>()) as u64,
        );
        queue.submit(Some(encoder.finish()));

        let result_entries = readback_vec::<ResultEntry>(device, &readback_buffer);
        result_entries
            .iter()
            .map(|entry| {
                if entry.found == 0 || entry.value == TOMBSTONE_VALUE {
                    None
                } else {
                    Some(entry.value)
                }
            })
            .collect()
    }
}

impl BulkDeletePipeline {
    pub fn new(device: &wgpu::Device) -> Self {
        let bulk_delete_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bulk-delete-shader"),
            source: wgpu::ShaderSource::Wgsl(BULK_DELETE_WGSL.into()),
        });
        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("bulk-delete-bind-group-layout"),
                entries: &[
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
            });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("bulk-delete-pipeline-layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("bulk-delete-pipeline"),
            layout: Some(&pipeline_layout),
            module: &bulk_delete_shader,
            entry_point: "main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        Self {
            pipeline,
            bind_group_layout,
        }
    }

    pub fn execute(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        slab: &GpuArray<KvEntry>,
        keys: &[u32],
    ) {
        if keys.is_empty() {
            return;
        }

        let keys_buffer = create_buffer_with_data(
            device,
            "delete-keys-buffer",
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            keys,
        );
        let keys_meta = KeysMeta {
            len: keys.len() as u32,
            _pad: [0; 3],
        };
        let keys_meta_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("delete-keys-meta-buffer"),
            size: std::mem::size_of::<KeysMeta>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        {
            let mut view = keys_meta_buffer.slice(..).get_mapped_range_mut();
            view.copy_from_slice(bytemuck::bytes_of(&keys_meta));
        }
        keys_meta_buffer.unmap();

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bulk-delete-bind-group"),
            layout: &self.bind_group_layout,
            entries: &[
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
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("bulk-delete-encoder"),
        });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("bulk-delete-pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            let workgroups = ((keys.len() as u32) + 63) / 64;
            cpass.dispatch_workgroups(workgroups, 1, 1);
        }
        queue.submit(Some(encoder.finish()));
    }
}

impl BulkPutPipeline {
    pub fn new(device: &wgpu::Device) -> Self {
        let sort_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bulk-sort-shader"),
            source: wgpu::ShaderSource::Wgsl(BULK_SORT_WGSL.into()),
        });
        let sort_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("bulk-sort-bind-group-layout"),
                entries: &[
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
            });
        let sort_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("bulk-sort-pipeline-layout"),
            bind_group_layouts: &[&sort_bind_group_layout],
            push_constant_ranges: &[],
        });
        let sort_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("bulk-sort-pipeline"),
            layout: Some(&sort_pipeline_layout),
            module: &sort_shader,
            entry_point: "main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        let dedup_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bulk-dedup-shader"),
            source: wgpu::ShaderSource::Wgsl(BULK_DEDUP_WGSL.into()),
        });
        let dedup_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("bulk-dedup-bind-group-layout"),
                entries: &[
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
            });
        let dedup_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("bulk-dedup-pipeline-layout"),
                bind_group_layouts: &[&dedup_bind_group_layout],
                push_constant_ranges: &[],
            });
        let dedup_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("bulk-dedup-pipeline"),
            layout: Some(&dedup_pipeline_layout),
            module: &dedup_shader,
            entry_point: "main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        let merge_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bulk-merge-shader"),
            source: wgpu::ShaderSource::Wgsl(BULK_MERGE_WGSL.into()),
        });
        let merge_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("bulk-merge-bind-group-layout"),
                entries: &[
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
            });
        let merge_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("bulk-merge-pipeline-layout"),
                bind_group_layouts: &[&merge_bind_group_layout],
                push_constant_ranges: &[],
            });
        let merge_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("bulk-merge-pipeline"),
            layout: Some(&merge_pipeline_layout),
            module: &merge_shader,
            entry_point: "main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        Self {
            sort_pipeline,
            sort_bind_group_layout,
            dedup_pipeline,
            dedup_bind_group_layout,
            merge_pipeline,
            merge_bind_group_layout,
        }
    }

    pub fn execute(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        slab: &GpuArray<KvEntry>,
        input: &GpuArray<KvEntry>,
        merge: &GpuArray<KvEntry>,
        merge_meta: &GpuStorage<MergeMeta>,
        entries_len: u32,
    ) -> Result<u32, GpuMapError> {
        let len = entries_len;
        let padded_len = len.next_power_of_two();
        if padded_len > slab.capacity() {
            return Err(GpuMapError::CapacityExceeded {
                capacity: slab.capacity(),
                requested: slab.len() + padded_len,
            });
        }

        if padded_len > len {
            let pad_count = (padded_len - len) as usize;
            let padding = vec![
                KvEntry {
                    key: u32::MAX,
                    value: 0,
                };
                pad_count
            ];
            let offset = (len as u64) * std::mem::size_of::<KvEntry>() as u64;
            queue.write_buffer(input.buffer(), offset, bytemuck::cast_slice(&padding));
        }

        if len > 1 {
            let sort_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("bulk-sort-params"),
                size: std::mem::size_of::<SortParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let sort_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bulk-sort-bind-group"),
                layout: &self.sort_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: BULK_SORT_BIND_INPUT,
                        resource: input.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: BULK_SORT_BIND_PARAMS,
                        resource: sort_params_buffer.as_entire_binding(),
                    },
                ],
            });

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
                    queue.write_buffer(&sort_params_buffer, 0, bytemuck::bytes_of(&params));
                    let mut encoder =
                        device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("bulk-sort-encoder"),
                        });
                    {
                        let mut cpass = encoder.begin_compute_pass(
                            &wgpu::ComputePassDescriptor {
                                label: Some("bulk-sort-pass"),
                                timestamp_writes: None,
                            },
                        );
                        cpass.set_pipeline(&self.sort_pipeline);
                        cpass.set_bind_group(0, &sort_bind_group, &[]);
                        cpass.dispatch_workgroups(workgroups, 1, 1);
                    }
                    queue.submit(Some(encoder.finish()));
                    j /= 2;
                }
                k *= 2;
            }
        }

        let dedup_params = DedupParams { len, _pad: [0; 3] };
        let dedup_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bulk-dedup-params"),
            size: std::mem::size_of::<DedupParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        {
            let mut view = dedup_params_buffer.slice(..).get_mapped_range_mut();
            view.copy_from_slice(bytemuck::bytes_of(&dedup_params));
        }
        dedup_params_buffer.unmap();
        let dedup_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bulk-dedup-bind-group"),
            layout: &self.dedup_bind_group_layout,
            entries: &[
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
        });
        let dedup_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bulk-dedup-readback"),
            size: std::mem::size_of::<MergeMeta>() as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut dedup_encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("bulk-dedup-encoder"),
            });
        {
            let mut cpass = dedup_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("bulk-dedup-pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.dedup_pipeline);
            cpass.set_bind_group(0, &dedup_bind_group, &[]);
            cpass.dispatch_workgroups(1, 1, 1);
        }
        dedup_encoder.copy_buffer_to_buffer(
            merge_meta.buffer(),
            0,
            &dedup_readback,
            0,
            std::mem::size_of::<MergeMeta>() as u64,
        );
        queue.submit(Some(dedup_encoder.finish()));
        let dedup_meta = readback_vec::<MergeMeta>(device, &dedup_readback);
        let dedup_len = dedup_meta.first().map(|m| m.len).unwrap_or(0);

        let input_meta = InputMeta {
            len: dedup_len,
            _pad: [0; 3],
        };
        let input_meta_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("input-meta-buffer"),
            size: std::mem::size_of::<InputMeta>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        {
            let mut view = input_meta_buffer.slice(..).get_mapped_range_mut();
            view.copy_from_slice(bytemuck::bytes_of(&input_meta));
        }
        input_meta_buffer.unmap();

        let merge_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bulk-merge-bind-group"),
            layout: &self.merge_bind_group_layout,
            entries: &[
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
        });

        let merge_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("merge-meta-readback"),
            size: std::mem::size_of::<MergeMeta>() as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("bulk-merge-encoder"),
        });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("bulk-merge-pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.merge_pipeline);
            cpass.set_bind_group(0, &merge_bind_group, &[]);
            cpass.dispatch_workgroups(1, 1, 1);
        }

        let slab_bytes = (slab.capacity() as u64) * std::mem::size_of::<KvEntry>() as u64;
        encoder.copy_buffer_to_buffer(merge.buffer(), 0, slab.buffer(), 0, slab_bytes);
        encoder.copy_buffer_to_buffer(
            merge_meta.buffer(),
            0,
            &merge_readback,
            0,
            std::mem::size_of::<MergeMeta>() as u64,
        );
        queue.submit(Some(encoder.finish()));

        let merge_meta_vec = readback_vec::<MergeMeta>(device, &merge_readback);
        Ok(merge_meta_vec.first().map(|m| m.len).unwrap_or(0))
    }
}

const TOMBSTONE_VALUE: u32 = 0xFFFF_FFFF;

const BULK_GET_BIND_SLAB: u32 = 0;
const BULK_GET_BIND_SLAB_META: u32 = 1;
const BULK_GET_BIND_KEYS: u32 = 2;
const BULK_GET_BIND_KEYS_META: u32 = 3;
const BULK_GET_BIND_RESULTS: u32 = 4;

const BULK_DELETE_BIND_SLAB: u32 = 0;
const BULK_DELETE_BIND_SLAB_META: u32 = 1;
const BULK_DELETE_BIND_KEYS: u32 = 2;
const BULK_DELETE_BIND_KEYS_META: u32 = 3;

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

        let slab = GpuArray::new(
            &device,
            capacity,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            "slab-buffer",
        );

        let input = GpuArray::new(
            &device,
            capacity,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            "input-buffer",
        );

        let merge = GpuArray::new(
            &device,
            capacity,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            "merge-buffer",
        );

        let merge_meta = GpuStorage::new(
            &device,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            "merge-meta-buffer",
        );

        let bulk_get = BulkGetPipeline::new(&device);
        let bulk_delete = BulkDeletePipeline::new(&device);
        let bulk_put = BulkPutPipeline::new(&device);

        Ok(Self {
            device,
            queue,
            slab,
            input,
            merge,
            merge_meta,
            bulk_get,
            bulk_delete,
            bulk_put,
        })
    }

    pub fn capacity(&self) -> u32 {
        self.slab.capacity()
    }

    pub fn len(&self) -> u32 {
        self.slab.len()
    }

    pub fn update_len(&mut self, new_len: u32) {
        self.slab.update_len(&self.queue, new_len);
    }

    pub fn bulk_put(&mut self, entries: &[KvEntry]) -> Result<(), GpuMapError> {
        if entries.is_empty() {
            return Ok(());
        }

        if entries.iter().any(|entry| entry.value == TOMBSTONE_VALUE) {
            return Err(GpuMapError::TombstoneValueReserved {
                value: TOMBSTONE_VALUE,
            });
        }

        let len = entries.len() as u32;
        let requested = self.slab.len() + len;
        if requested > self.slab.capacity() {
            return Err(GpuMapError::CapacityExceeded {
                capacity: self.slab.capacity(),
                requested,
            });
        }

        self.input.write(&self.queue, entries);
        let merge_len = self.bulk_put.execute(
            &self.device,
            &self.queue,
            &self.slab,
            &self.input,
            &self.merge,
            &self.merge_meta,
            len,
        )?;
        self.update_len(merge_len);
        Ok(())
    }

    pub fn bulk_get(&self, keys: &[u32]) -> Vec<Option<u32>> {
        self.bulk_get
            .execute(&self.device, &self.queue, &self.slab, keys)
    }

    pub fn bulk_delete(&self, keys: &[u32]) {
        self.bulk_delete
            .execute(&self.device, &self.queue, &self.slab, keys);
    }

    pub fn delete(&self, key: u32) {
        self.bulk_delete(std::slice::from_ref(&key));
    }

    pub fn put(&mut self, key: u32, value: u32) -> Result<(), GpuMapError> {
        let entry = KvEntry { key, value };
        self.bulk_put(std::slice::from_ref(&entry))
    }

    pub fn get(&self, key: u32) -> Option<u32> {
        self.bulk_get(&[key]).into_iter().next().unwrap_or(None)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuMapError {
    CapacityExceeded { capacity: u32, requested: u32 },
    TombstoneValueReserved { value: u32 },
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
        results[idx].value = slab[lo].value;
        results[idx].found = 1u;
    } else {
        results[idx].value = 0u;
        results[idx].found = 0u;
    }
}
"#;

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

fn readback_vec<T: Pod>(device: &wgpu::Device, buffer: &wgpu::Buffer) -> Vec<T> {
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
    fn bulk_delete_clears_values() {
        let mut map = pollster::block_on(GpuSortedMap::new(8)).unwrap();
        let entries = [
            KvEntry { key: 1, value: 10 },
            KvEntry { key: 2, value: 20 },
            KvEntry { key: 3, value: 30 },
        ];
        map.bulk_put(&entries).unwrap();
        map.bulk_delete(&[1, 3]);
        let results = map.bulk_get(&[1, 2, 3]);
        assert_eq!(results, vec![None, Some(20), None]);
    }

    #[test]
    fn delete_single_key() {
        let mut map = pollster::block_on(GpuSortedMap::new(4)).unwrap();
        map.put(9, 99).unwrap();
        map.delete(9);
        assert_eq!(map.get(9), None);
    }

    #[test]
    fn put_rejects_tombstone_value() {
        let mut map = pollster::block_on(GpuSortedMap::new(4)).unwrap();
        let err = map.put(1, 0xFFFF_FFFF).unwrap_err();
        assert!(matches!(err, super::GpuMapError::TombstoneValueReserved { .. }));
    }
}
