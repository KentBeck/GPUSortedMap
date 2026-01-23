use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug, Default)]
pub struct KvEntry {
    pub key: u32,
    pub value: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug, Default)]
pub struct SlabMeta {
    pub len: u32,
    pub capacity: u32,
    pub _pad: [u32; 2],
}

pub struct GpuSortedMap {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub slab_buffer: wgpu::Buffer,
    pub input_buffer: wgpu::Buffer,
    pub meta_buffer: wgpu::Buffer,
    capacity: u32,
    len: u32,
    host_slab: Vec<KvEntry>,
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

        Ok(Self {
            device,
            queue,
            slab_buffer,
            input_buffer,
            meta_buffer,
            capacity,
            len: 0,
            host_slab: Vec::with_capacity(capacity as usize),
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

        let mut incoming_map = std::collections::BTreeMap::new();
        for entry in entries {
            incoming_map.insert(entry.key, entry.value);
        }
        let incoming: Vec<KvEntry> = incoming_map
            .into_iter()
            .map(|(key, value)| KvEntry { key, value })
            .collect();

        let mut merged = Vec::with_capacity(self.host_slab.len() + incoming.len());
        let mut i = 0;
        let mut j = 0;
        while i < self.host_slab.len() && j < incoming.len() {
            let current = self.host_slab[i];
            let next = incoming[j];
            if current.key < next.key {
                merged.push(current);
                i += 1;
            } else if current.key > next.key {
                merged.push(next);
                j += 1;
            } else {
                merged.push(next);
                i += 1;
                j += 1;
            }
        }
        if i < self.host_slab.len() {
            merged.extend_from_slice(&self.host_slab[i..]);
        }
        if j < incoming.len() {
            merged.extend_from_slice(&incoming[j..]);
        }

        let requested = merged.len() as u32;
        if requested > self.capacity {
            return Err(GpuMapError::CapacityExceeded {
                capacity: self.capacity,
                requested,
            });
        }

        self.host_slab = merged;
        self.update_len(requested);
        self.queue.write_buffer(
            &self.slab_buffer,
            0,
            bytemuck::cast_slice(&self.host_slab),
        );
        Ok(())
    }

    pub fn bulk_get(&self, keys: &[u32]) -> Vec<Option<u32>> {
        keys.iter()
            .map(|key| {
                self.host_slab
                    .binary_search_by_key(key, |entry| entry.key)
                    .ok()
                    .map(|idx| self.host_slab[idx].value)
            })
            .collect()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuMapError {
    CapacityExceeded { capacity: u32, requested: u32 },
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
}
