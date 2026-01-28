mod gpu_array;
mod pipelines;

use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

use crate::gpu_array::{GpuArray, GpuStorage};
use crate::pipelines::{BulkDeletePipeline, BulkGetPipeline, BulkPutPipeline, MergeMeta};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug, Default)]
pub struct KvEntry {
    pub key: u32,
    pub value: u32,
}

const TOMBSTONE_VALUE: u32 = 0xFFFF_FFFF;

pub struct GpuSortedMap {
    queue: Arc<wgpu::Queue>,
    slab: GpuArray<KvEntry>,
    input: GpuArray<KvEntry>,
    merge: GpuArray<KvEntry>,
    merge_meta: GpuStorage<MergeMeta>,
    bulk_get: BulkGetPipeline,
    bulk_delete: BulkDeletePipeline,
    bulk_put: BulkPutPipeline,
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
        let device = Arc::new(device);
        let queue = Arc::new(queue);

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

        let bulk_get = BulkGetPipeline::new(Arc::clone(&device), Arc::clone(&queue));
        let bulk_delete = BulkDeletePipeline::new(Arc::clone(&device), Arc::clone(&queue));
        let bulk_put = BulkPutPipeline::new(Arc::clone(&device), Arc::clone(&queue));

        Ok(Self {
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

    pub fn bulk_get(&self, keys: &[u32]) -> Vec<Option<u32>> {
        self.bulk_get.execute(&self.slab, keys)
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
            &self.slab,
            &self.input,
            &self.merge,
            &self.merge_meta,
            len,
        )?;
        self.update_len(merge_len);
        Ok(())
    }

    pub fn bulk_delete(&mut self, keys: &[u32]) {
        self.bulk_delete.execute(&self.slab, keys);
    }

    pub fn get(&self, key: u32) -> Option<u32> {
        self.bulk_get(&[key]).into_iter().next().unwrap_or(None)
    }

    pub fn put(&mut self, key: u32, value: u32) -> Result<(), GpuMapError> {
        let entry = KvEntry { key, value };
        self.bulk_put(std::slice::from_ref(&entry))
    }

    pub fn delete(&mut self, key: u32) {
        self.bulk_delete(std::slice::from_ref(&key));
    }

    pub fn capacity(&self) -> u32 {
        self.slab.capacity()
    }

    pub fn len(&self) -> u32 {
        self.slab.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn update_len(&mut self, new_len: u32) {
        self.slab.update_len(&self.queue, new_len);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuMapError {
    CapacityExceeded { capacity: u32, requested: u32 },
    TombstoneValueReserved { value: u32 },
}

impl std::fmt::Display for GpuMapError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuMapError::CapacityExceeded { capacity, requested } => {
                write!(
                    f,
                    "Capacity exceeded: requested {} entries but capacity is {}",
                    requested, capacity
                )
            }
            GpuMapError::TombstoneValueReserved { value } => {
                write!(
                    f,
                    "Value 0x{:08X} is reserved as tombstone marker and cannot be used",
                    value
                )
            }
        }
    }
}

impl std::error::Error for GpuMapError {}

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

    #[test]
    fn bulk_get_empty_keys() {
        let map = pollster::block_on(GpuSortedMap::new(10)).unwrap();
        let results = map.bulk_get(&[]);
        assert!(results.is_empty());
    }

    #[test]
    fn bulk_delete_empty_keys() {
        let mut map = pollster::block_on(GpuSortedMap::new(10)).unwrap();
        map.put(1, 10).unwrap();
        map.bulk_delete(&[]);
        assert_eq!(map.get(1), Some(10));
    }

    #[test]
    fn bulk_put_internal_capacity_exceeded() {
        // Map capacity 4. High-level check: len(0) + 3 <= 4. OK.
        // Internal check: padded_len(3) -> 4. 4 <= 4. OK.
        // Wait, if I use entries.len() = 5, high-level check (5 <= 4) fails.
        // If I use capacity 6 and entries 5.
        // High-level: 5 <= 6. OK.
        // Internal: next_power_of_two(5) = 8. 8 > 6. FAIL.
        let mut map = pollster::block_on(GpuSortedMap::new(6)).unwrap();
        let entries = vec![KvEntry { key: 1, value: 1 }; 5];
        let res = map.bulk_put(&entries);
        assert!(matches!(res, Err(super::GpuMapError::CapacityExceeded { .. })));
    }

    #[test]
    fn is_empty_returns_true_for_new_map() {
        let map = pollster::block_on(GpuSortedMap::new(10)).unwrap();
        assert!(map.is_empty());
        assert_eq!(map.len(), 0);
    }

    #[test]
    fn is_empty_returns_false_after_insert() {
        let mut map = pollster::block_on(GpuSortedMap::new(10)).unwrap();
        map.put(1, 10).unwrap();
        assert!(!map.is_empty());
        assert_eq!(map.len(), 1);
    }
}
