use bytemuck::{Pod, Zeroable};

pub const TOMBSTONE_VALUE: u32 = 0xFFFF_FFFF;

pub const BULK_GET_BIND_SLAB: u32 = 0;
pub const BULK_GET_BIND_SLAB_META: u32 = 1;
pub const BULK_GET_BIND_KEYS: u32 = 2;
pub const BULK_GET_BIND_KEYS_META: u32 = 3;
pub const BULK_GET_BIND_RESULTS: u32 = 4;

pub const BULK_DELETE_BIND_SLAB: u32 = 0;
pub const BULK_DELETE_BIND_SLAB_META: u32 = 1;
pub const BULK_DELETE_BIND_KEYS: u32 = 2;
pub const BULK_DELETE_BIND_KEYS_META: u32 = 3;

pub const BULK_SORT_BIND_INPUT: u32 = 0;
pub const BULK_SORT_BIND_PARAMS: u32 = 1;

pub const BULK_DEDUP_BIND_INPUT: u32 = 0;
pub const BULK_DEDUP_BIND_PARAMS: u32 = 1;
pub const BULK_DEDUP_BIND_META: u32 = 2;

pub const BULK_MERGE_BIND_SLAB: u32 = 0;
pub const BULK_MERGE_BIND_INPUT: u32 = 1;
pub const BULK_MERGE_BIND_OUTPUT: u32 = 2;
pub const BULK_MERGE_BIND_SLAB_META: u32 = 3;
pub const BULK_MERGE_BIND_INPUT_META: u32 = 4;
pub const BULK_MERGE_BIND_MERGE_META: u32 = 5;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug, Default)]
pub struct MergeMeta {
    pub len: u32,
    pub _pad: [u32; 3],
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
