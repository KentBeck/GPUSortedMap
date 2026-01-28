use gpusorted_map::{Capacity, GpuSortedMap, Key, KvEntry, Value};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut map = pollster::block_on(GpuSortedMap::new(Capacity::new(1024)))?;

    map.bulk_put(&[
        KvEntry {
            key: Key::new(1),
            value: Value::new(10),
        },
        KvEntry {
            key: Key::new(2),
            value: Value::new(20),
        },
        KvEntry {
            key: Key::new(3),
            value: Value::new(30),
        },
    ])?;

    assert_eq!(map.get(Key::new(1)), Some(Value::new(10)));
    assert_eq!(map.get(Key::new(2)), Some(Value::new(20)));

    map.delete(Key::new(2));
    assert_eq!(map.get(Key::new(2)), None);

    Ok(())
}
