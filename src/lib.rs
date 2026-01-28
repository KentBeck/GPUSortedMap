//! GPU-accelerated sorted key/value store built on wgpu.
//!
//! # Quick start
//!
//! ```rust
//! use gpusorted_map::{Capacity, GpuSortedMap, Key, KvEntry, Value};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let mut map = pollster::block_on(GpuSortedMap::new(Capacity::new(1024)))?;
//! map.bulk_put(&[
//!     KvEntry { key: Key::new(1), value: Value::new(10) },
//!     KvEntry { key: Key::new(2), value: Value::new(20) },
//! ])?;
//! assert_eq!(map.get(Key::new(1)), Some(Value::new(10)));
//! # Ok(())
//! # }
//! ```

mod gpu_array;
mod pipelines;

use bytemuck::{Pod, Zeroable};
use std::collections::HashSet;
use std::sync::Arc;

use crate::gpu_array::{GpuArray, GpuStorage};
use crate::pipelines::{
    BulkDeletePipeline, BulkGetPipeline, BulkPutPipeline, MergeMeta, RangeScanPipeline,
};

/// Key wrapper to distinguish keys from other `u32` values.
#[repr(transparent)]
#[derive(Clone, Copy, Pod, Zeroable, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Key(pub u32);

impl Key {
    pub const fn new(value: u32) -> Self {
        Self(value)
    }
}

impl From<u32> for Key {
    fn from(value: u32) -> Self {
        Self(value)
    }
}

impl From<Key> for u32 {
    fn from(value: Key) -> Self {
        value.0
    }
}

/// Value wrapper to distinguish values from other `u32` values.
#[repr(transparent)]
#[derive(Clone, Copy, Pod, Zeroable, Debug, Default, PartialEq, Eq)]
pub struct Value(pub u32);

impl Value {
    pub const fn new(value: u32) -> Self {
        Self(value)
    }
}

impl From<u32> for Value {
    fn from(value: u32) -> Self {
        Self(value)
    }
}

impl From<Value> for u32 {
    fn from(value: Value) -> Self {
        value.0
    }
}

/// Capacity wrapper to distinguish sizes from other `u32` values.
#[repr(transparent)]
#[derive(Clone, Copy, Pod, Zeroable, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct Capacity(pub u32);

impl Capacity {
    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    pub const fn as_u32(self) -> u32 {
        self.0
    }
}

impl From<u32> for Capacity {
    fn from(value: u32) -> Self {
        Self(value)
    }
}

impl From<Capacity> for u32 {
    fn from(value: Capacity) -> Self {
        value.0
    }
}

/// Length wrapper to distinguish sizes from other `u32` values.
#[repr(transparent)]
#[derive(Clone, Copy, Pod, Zeroable, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct Length(pub u32);

impl Length {
    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    pub const fn as_u32(self) -> u32 {
        self.0
    }
}

impl From<u32> for Length {
    fn from(value: u32) -> Self {
        Self(value)
    }
}

impl From<Length> for u32 {
    fn from(value: Length) -> Self {
        value.0
    }
}

/// Key/value pair stored in the GPU slab.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug, Default, PartialEq, Eq)]
pub struct KvEntry {
    pub key: Key,
    pub value: Value,
}

const TOMBSTONE_VALUE: Value = Value(0xFFFF_FFFF);

/// GPU-backed sorted map with batched operations.
pub struct GpuSortedMap {
    queue: Arc<wgpu::Queue>,
    slab: GpuArray<KvEntry>,
    input: GpuArray<KvEntry>,
    merge: GpuArray<KvEntry>,
    merge_meta: GpuStorage<MergeMeta>,
    bulk_get: BulkGetPipeline,
    bulk_delete: BulkDeletePipeline,
    bulk_put: BulkPutPipeline,
    range_scan: RangeScanPipeline,
    live_len: Length,
}

impl GpuSortedMap {
    /// Create a new map with the given slab capacity.
    pub async fn new(capacity: Capacity) -> Result<Self, wgpu::RequestDeviceError> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let adapter = match instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
        {
            Some(adapter) => adapter,
            None => instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::LowPower,
                    compatible_surface: None,
                    force_fallback_adapter: true,
                })
                .await
                .expect("no suitable GPU adapters found (including fallback)"),
        };

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
        let range_scan = RangeScanPipeline::new(Arc::clone(&device), Arc::clone(&queue));

        Ok(Self {
            queue,
            slab,
            input,
            merge,
            merge_meta,
            bulk_get,
            bulk_delete,
            bulk_put,
            range_scan,
            live_len: Length::new(0),
        })
    }

    /// Batch lookup of keys.
    pub fn bulk_get(&self, keys: &[Key]) -> Vec<Option<Value>> {
        self.bulk_get.execute(&self.slab, keys)
    }

    /// Batch insert/update of entries.
    pub fn bulk_put(&mut self, entries: &[KvEntry]) -> Result<(), GpuMapError> {
        if entries.is_empty() {
            return Ok(());
        }

        if entries.iter().any(|entry| entry.value == TOMBSTONE_VALUE) {
            return Err(GpuMapError::TombstoneValueReserved {
                value: TOMBSTONE_VALUE,
            });
        }

        let unique_keys = unique_keys_from_entries(entries).map_err(|key| {
            GpuMapError::DuplicateKeys { key }
        })?;
        let existing = self.count_existing_keys(&unique_keys);
        let net_new = unique_keys.len().saturating_sub(existing) as u32;
        let requested = Length::new(self.slab.len().0 + net_new);
        if requested.0 > self.slab.capacity().0 {
            return Err(GpuMapError::CapacityExceeded {
                capacity: self.slab.capacity(),
                requested,
            });
        }

        self.input.write(&self.queue, entries);
        let len = Length::new(entries.len() as u32);
        let merge_len = self.bulk_put.execute(
            &self.slab,
            &self.input,
            &self.merge,
            &self.merge_meta,
            len.0,
        )?;
        self.update_len(Length::new(merge_len));
        self.live_len = Length::new(self.live_len.0 + net_new);
        Ok(())
    }

    /// Batch delete of keys.
    pub fn bulk_delete(&mut self, keys: &[Key]) {
        if keys.is_empty() {
            return;
        }
        let unique_keys = unique_keys(keys);
        let existing = self.count_existing_keys(&unique_keys);
        self.bulk_delete.execute(&self.slab, &unique_keys);
        self.live_len = Length::new(self.live_len.0.saturating_sub(existing as u32));
    }

    /// Single-key lookup convenience wrapper over `bulk_get`.
    pub fn get(&self, key: Key) -> Option<Value> {
        self.bulk_get(&[key]).into_iter().next().unwrap_or(None)
    }

    /// Single-key insert/update convenience wrapper over `bulk_put`.
    pub fn put(&mut self, key: Key, value: Value) -> Result<(), GpuMapError> {
        let entry = KvEntry { key, value };
        self.bulk_put(std::slice::from_ref(&entry))
    }

    /// Single-key delete convenience wrapper over `bulk_delete`.
    pub fn delete(&mut self, key: Key) {
        self.bulk_delete(std::slice::from_ref(&key));
    }

    /// Returns entries with keys in `[from_key, to_key)`.
    pub fn range(&self, from_key: Key, to_key: Key) -> Vec<KvEntry> {
        self.range_scan
            .execute(&self.slab, from_key, to_key)
            .into_iter()
            .filter(|entry| entry.value != TOMBSTONE_VALUE)
            .collect()
    }

    /// Iterator over entries with keys in `[from_key, to_key)`.
    pub fn range_iter(&self, from_key: Key, to_key: Key) -> std::vec::IntoIter<KvEntry> {
        self.range(from_key, to_key).into_iter()
    }

    /// Total slab capacity.
    pub fn capacity(&self) -> Capacity {
        self.slab.capacity()
    }

    /// Current number of live entries (tombstones are excluded).
    pub fn len(&self) -> Length {
        self.live_len
    }

    #[must_use]
    /// Returns true if the map is empty.
    pub fn is_empty(&self) -> bool {
        self.live_len.0 == 0
    }

    /// Update the stored slab length metadata.
    ///
    /// This does not change the live entry count returned by `len()`.
    pub fn update_len(&mut self, new_len: Length) {
        self.slab.update_len(&self.queue, new_len);
    }

    fn count_existing_keys(&self, keys: &[Key]) -> usize {
        if keys.is_empty() {
            return 0;
        }
        self.bulk_get(keys).iter().filter(|v| v.is_some()).count()
    }
}

fn unique_keys_from_entries(entries: &[KvEntry]) -> Result<Vec<Key>, Key> {
    let mut seen = HashSet::with_capacity(entries.len());
    let mut keys = Vec::with_capacity(entries.len());
    for entry in entries {
        if !seen.insert(entry.key) {
            return Err(entry.key);
        }
        keys.push(entry.key);
    }
    Ok(keys)
}

fn unique_keys(keys: &[Key]) -> Vec<Key> {
    let mut seen = HashSet::with_capacity(keys.len());
    let mut out = Vec::with_capacity(keys.len());
    for &key in keys {
        if seen.insert(key) {
            out.push(key);
        }
    }
    out
}

/// Errors returned by `GpuSortedMap` operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuMapError {
    CapacityExceeded {
        capacity: Capacity,
        requested: Length,
    },
    TombstoneValueReserved {
        value: Value,
    },
    DuplicateKeys {
        key: Key,
    },
}

impl std::fmt::Display for GpuMapError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuMapError::CapacityExceeded {
                capacity,
                requested,
            } => {
                write!(
                    f,
                    "Capacity exceeded: requested {} entries but capacity is {}",
                    requested.0, capacity.0
                )
            }
            GpuMapError::TombstoneValueReserved { value } => {
                write!(
                    f,
                    "Value 0x{:08X} is reserved as tombstone marker and cannot be used",
                    value.0
                )
            }
            GpuMapError::DuplicateKeys { key } => {
                write!(f, "Duplicate key in batch: {}", key.0)
            }
        }
    }
}

impl std::error::Error for GpuMapError {}

#[cfg(test)]
mod tests {
    use super::{Capacity, GpuSortedMap, Key, KvEntry, Length, Value};

    fn k(value: u32) -> Key {
        Key::new(value)
    }

    fn v(value: u32) -> Value {
        Value::new(value)
    }

    #[test]
    fn creates_gpu_sorted_map() {
        let map = pollster::block_on(GpuSortedMap::new(Capacity::new(1024)));
        assert!(map.is_ok(), "GpuSortedMap::new should succeed");
    }

    #[test]
    fn put_then_get() {
        let mut map = pollster::block_on(GpuSortedMap::new(Capacity::new(8))).unwrap();
        let entries = [
            KvEntry {
                key: k(42),
                value: v(7),
            },
            KvEntry {
                key: k(7),
                value: v(9),
            },
            KvEntry {
                key: k(13),
                value: v(1),
            },
        ];
        map.bulk_put(&entries).unwrap();

        let results = map.bulk_get(&[k(7), k(13), k(42), k(99)]);
        assert_eq!(results, vec![Some(v(9)), Some(v(1)), Some(v(7)), None]);
    }

    #[test]
    fn single_put_then_get() {
        let mut map = pollster::block_on(GpuSortedMap::new(Capacity::new(4))).unwrap();
        map.put(k(5), v(11)).unwrap();
        assert_eq!(map.get(k(5)), Some(v(11)));
    }

    #[test]
    fn single_get_missing_key() {
        let map = pollster::block_on(GpuSortedMap::new(Capacity::new(4))).unwrap();
        assert_eq!(map.get(k(9)), None);
    }

    #[test]
    fn bulk_delete_clears_values() {
        let mut map = pollster::block_on(GpuSortedMap::new(Capacity::new(8))).unwrap();
        let entries = [
            KvEntry {
                key: k(1),
                value: v(10),
            },
            KvEntry {
                key: k(2),
                value: v(20),
            },
            KvEntry {
                key: k(3),
                value: v(30),
            },
        ];
        map.bulk_put(&entries).unwrap();
        map.bulk_delete(&[k(1), k(3)]);
        let results = map.bulk_get(&[k(1), k(2), k(3)]);
        assert_eq!(results, vec![None, Some(v(20)), None]);
    }

    #[test]
    fn delete_single_key() {
        let mut map = pollster::block_on(GpuSortedMap::new(Capacity::new(4))).unwrap();
        map.put(k(9), v(99)).unwrap();
        map.delete(k(9));
        assert_eq!(map.get(k(9)), None);
    }

    #[test]
    fn range_returns_half_open_interval() {
        let mut map = pollster::block_on(GpuSortedMap::new(Capacity::new(16))).unwrap();
        map.bulk_put(&[
            KvEntry {
                key: k(1),
                value: v(10),
            },
            KvEntry {
                key: k(2),
                value: v(20),
            },
            KvEntry {
                key: k(3),
                value: v(30),
            },
            KvEntry {
                key: k(4),
                value: v(40),
            },
        ])
        .unwrap();

        let entries = map.range(k(2), k(4));
        let keys: Vec<Key> = entries.iter().map(|entry| entry.key).collect();
        assert_eq!(keys, vec![k(2), k(3)]);
    }

    #[test]
    fn range_empty_when_from_equals_to() {
        let mut map = pollster::block_on(GpuSortedMap::new(Capacity::new(16))).unwrap();
        map.bulk_put(&[
            KvEntry {
                key: k(1),
                value: v(10),
            },
            KvEntry {
                key: k(2),
                value: v(20),
            },
        ])
        .unwrap();
        assert!(map.range(k(2), k(2)).is_empty());
    }

    #[test]
    fn range_empty_when_from_greater_than_to() {
        let mut map = pollster::block_on(GpuSortedMap::new(Capacity::new(16))).unwrap();
        map.bulk_put(&[
            KvEntry {
                key: k(1),
                value: v(10),
            },
            KvEntry {
                key: k(2),
                value: v(20),
            },
        ])
        .unwrap();
        assert!(map.range(k(3), k(1)).is_empty());
    }

    #[test]
    fn range_empty_on_empty_map() {
        let map = pollster::block_on(GpuSortedMap::new(Capacity::new(16))).unwrap();
        assert!(map.range(k(0), k(10)).is_empty());
    }

    #[test]
    fn range_outside_bounds_is_empty() {
        let mut map = pollster::block_on(GpuSortedMap::new(Capacity::new(16))).unwrap();
        map.bulk_put(&[
            KvEntry {
                key: k(10),
                value: v(10),
            },
            KvEntry {
                key: k(20),
                value: v(20),
            },
        ])
        .unwrap();
        assert!(map.range(k(0), k(5)).is_empty());
        assert!(map.range(k(30), k(40)).is_empty());
    }

    #[test]
    fn range_clamps_to_existing_keys() {
        let mut map = pollster::block_on(GpuSortedMap::new(Capacity::new(16))).unwrap();
        map.bulk_put(&[
            KvEntry {
                key: k(5),
                value: v(50),
            },
            KvEntry {
                key: k(10),
                value: v(100),
            },
            KvEntry {
                key: k(15),
                value: v(150),
            },
        ])
        .unwrap();
        let keys: Vec<Key> = map.range(k(0), k(100)).iter().map(|e| e.key).collect();
        assert_eq!(keys, vec![k(5), k(10), k(15)]);
    }

    #[test]
    fn range_starts_between_keys() {
        let mut map = pollster::block_on(GpuSortedMap::new(Capacity::new(16))).unwrap();
        map.bulk_put(&[
            KvEntry {
                key: k(10),
                value: v(100),
            },
            KvEntry {
                key: k(20),
                value: v(200),
            },
            KvEntry {
                key: k(30),
                value: v(300),
            },
        ])
        .unwrap();
        let keys: Vec<Key> = map.range(k(15), k(30)).iter().map(|e| e.key).collect();
        assert_eq!(keys, vec![k(20)]);
    }

    #[test]
    fn range_excludes_tombstones() {
        let mut map = pollster::block_on(GpuSortedMap::new(Capacity::new(16))).unwrap();
        map.bulk_put(&[
            KvEntry {
                key: k(1),
                value: v(10),
            },
            KvEntry {
                key: k(2),
                value: v(20),
            },
            KvEntry {
                key: k(3),
                value: v(30),
            },
        ])
        .unwrap();
        map.delete(k(2));
        let keys: Vec<Key> = map.range(k(1), k(4)).iter().map(|e| e.key).collect();
        assert_eq!(keys, vec![k(1), k(3)]);
    }

    #[test]
    fn put_rejects_tombstone_value() {
        let mut map = pollster::block_on(GpuSortedMap::new(Capacity::new(4))).unwrap();
        let err = map.put(k(1), v(0xFFFF_FFFF)).unwrap_err();
        assert!(matches!(
            err,
            super::GpuMapError::TombstoneValueReserved { .. }
        ));
    }

    #[test]
    fn bulk_get_empty_keys() {
        let map = pollster::block_on(GpuSortedMap::new(Capacity::new(10))).unwrap();
        let results = map.bulk_get(&[]);
        assert!(results.is_empty());
    }

    #[test]
    fn bulk_delete_empty_keys() {
        let mut map = pollster::block_on(GpuSortedMap::new(Capacity::new(10))).unwrap();
        map.put(k(1), v(10)).unwrap();
        map.bulk_delete(&[]);
        assert_eq!(map.get(k(1)), Some(v(10)));
    }

    #[test]
    fn bulk_put_internal_capacity_exceeded() {
        // Map capacity 4. High-level check: len(0) + 3 <= 4. OK.
        // Internal check: padded_len(3) -> 4. 4 <= 4. OK.
        // Wait, if I use entries.len() = 5, high-level check (5 <= 4) fails.
        // If I use capacity 6 and entries 5.
        // High-level: 5 <= 6. OK.
        // Internal: next_power_of_two(5) = 8. 8 > 6. FAIL.
        let mut map = pollster::block_on(GpuSortedMap::new(Capacity::new(6))).unwrap();
        let entries = [
            KvEntry {
                key: k(1),
                value: v(1),
            },
            KvEntry {
                key: k(2),
                value: v(2),
            },
            KvEntry {
                key: k(3),
                value: v(3),
            },
            KvEntry {
                key: k(4),
                value: v(4),
            },
            KvEntry {
                key: k(5),
                value: v(5),
            },
        ];
        let res = map.bulk_put(&entries);
        assert!(matches!(
            res,
            Err(super::GpuMapError::CapacityExceeded { .. })
        ));
    }

    #[test]
    fn is_empty_returns_true_for_new_map() {
        let map = pollster::block_on(GpuSortedMap::new(Capacity::new(10))).unwrap();
        assert!(map.is_empty());
        assert_eq!(map.len(), Length::new(0));
    }

    #[test]
    fn is_empty_returns_false_after_insert() {
        let mut map = pollster::block_on(GpuSortedMap::new(Capacity::new(10))).unwrap();
        map.put(k(1), v(10)).unwrap();
        assert!(!map.is_empty());
        assert_eq!(map.len(), Length::new(1));
    }

    #[test]
    fn update_overwrites_existing_key() {
        let mut map = pollster::block_on(GpuSortedMap::new(Capacity::new(8))).unwrap();
        map.put(k(1), v(10)).unwrap();
        map.put(k(1), v(20)).unwrap();

        assert_eq!(map.get(k(1)), Some(v(20)));
        let entries = map.range(k(1), k(2));
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].value, v(20));
        assert_eq!(map.len(), Length::new(1));
    }

    #[test]
    fn delete_reduces_live_len() {
        let mut map = pollster::block_on(GpuSortedMap::new(Capacity::new(8))).unwrap();
        map.bulk_put(&[
            KvEntry {
                key: k(1),
                value: v(10),
            },
            KvEntry {
                key: k(2),
                value: v(20),
            },
        ])
        .unwrap();
        map.bulk_delete(&[k(1), k(2)]);
        assert_eq!(map.len(), Length::new(0));
        assert!(map.is_empty());
    }

    #[test]
    fn duplicate_keys_in_bulk_put_is_error_and_atomic() {
        let mut map = pollster::block_on(GpuSortedMap::new(Capacity::new(8))).unwrap();
        map.put(k(1), v(10)).unwrap();

        let entries = [
            KvEntry {
                key: k(2),
                value: v(20),
            },
            KvEntry {
                key: k(2),
                value: v(30),
            },
        ];
        let err = map.bulk_put(&entries).unwrap_err();
        assert!(matches!(err, super::GpuMapError::DuplicateKeys { .. }));

        assert_eq!(map.get(k(1)), Some(v(10)));
        assert_eq!(map.get(k(2)), None);
        assert_eq!(map.len(), Length::new(1));
    }
}
