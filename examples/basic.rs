use gpusorted_map::{GpuSortedMap, KvEntry};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut map = pollster::block_on(GpuSortedMap::new(1024))?;

    map.bulk_put(&[
        KvEntry { key: 1, value: 10 },
        KvEntry { key: 2, value: 20 },
        KvEntry { key: 3, value: 30 },
    ])?;

    assert_eq!(map.get(1), Some(10));
    assert_eq!(map.get(2), Some(20));

    map.delete(2);
    assert_eq!(map.get(2), None);

    Ok(())
}
