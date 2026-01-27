use bytemuck::{Pod, Zeroable};
use std::marker::PhantomData;

use crate::pipelines::utils::create_buffer_with_data;

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
        let meta_buffer = create_buffer_with_data(
            device,
            "slab-meta-buffer",
            wgpu::BufferUsages::UNIFORM
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            &[meta],
        );

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

#[cfg(test)]
mod tests {
    use super::{GpuArray, SlabMeta};
    use crate::pipelines::utils::readback_vec;

    async fn create_device_queue() -> (wgpu::Device, wgpu::Queue) {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::LowPower,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .expect("no suitable GPU adapters found");
        adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("gpu-array-test-device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .expect("failed to request device")
    }

    fn readback_gpu_array<T: bytemuck::Pod>(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        source: &wgpu::Buffer,
        byte_len: u64,
    ) -> Vec<T> {
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu-array-readback"),
            size: byte_len,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("gpu-array-readback-encoder"),
        });
        encoder.copy_buffer_to_buffer(source, 0, &readback, 0, byte_len);
        queue.submit(Some(encoder.finish()));

        readback_vec::<T>(device, &readback)
    }

    #[test]
    fn creates_with_capacity() {
        let (device, _queue) = pollster::block_on(create_device_queue());
        let array = GpuArray::<u32>::new(
            &device,
            4,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            "test-buffer",
        );
        assert_eq!(array.capacity(), 4);
        assert_eq!(array.len(), 0);
    }

    #[test]
    fn update_len_clamps_and_updates_meta() {
        let (device, queue) = pollster::block_on(create_device_queue());
        let mut array = GpuArray::<u32>::new(
            &device,
            4,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            "test-buffer",
        );

        array.update_len(&queue, 10);
        assert_eq!(array.len(), 4);

        let meta = readback_gpu_array::<SlabMeta>(
            &device,
            &queue,
            array.meta_buffer(),
            std::mem::size_of::<SlabMeta>() as u64,
        );
        assert_eq!(meta.len(), 1);
        assert_eq!(meta[0].len, 4);
        assert_eq!(meta[0].capacity, 4);
    }

    #[test]
    fn write_persists_data() {
        let (device, queue) = pollster::block_on(create_device_queue());
        let array = GpuArray::<u32>::new(
            &device,
            4,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            "test-buffer",
        );

        let data = [10_u32, 20, 30, 40];
        array.write(&queue, &data);

        let readback = readback_gpu_array::<u32>(
            &device,
            &queue,
            array.buffer(),
            (data.len() * std::mem::size_of::<u32>()) as u64,
        );
        assert_eq!(readback, data);
    }

    #[test]
    fn write_empty_does_nothing() {
        let (device, queue) = pollster::block_on(create_device_queue());
        let array = GpuArray::<u32>::new(
            &device,
            4,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            "test-buffer",
        );
        array.write(&queue, &[]);
    }
}
