use bytemuck::Pod;

pub fn create_buffer_with_data<T: Pod>(
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

pub fn readback_vec<T: Pod>(device: &wgpu::Device, buffer: &wgpu::Buffer) -> Vec<T> {
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

pub(crate) fn readback_single<T: Pod>(device: &wgpu::Device, buffer: &wgpu::Buffer) -> T {
    readback_vec::<T>(device, buffer)
        .first()
        .copied()
        .expect("readback failed to return data")
}
