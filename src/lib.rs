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
}

#[cfg(test)]
mod tests {
    use super::GpuSortedMap;

    #[test]
    fn creates_gpu_sorted_map() {
        let map = pollster::block_on(GpuSortedMap::new(1024));
        assert!(map.is_ok(), "GpuSortedMap::new should succeed");
    }
}
