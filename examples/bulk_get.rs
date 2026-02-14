use gpusorted_map::{Capacity, GpuSortedMap, Key, KvEntry, Value};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut map = pollster::block_on(GpuSortedMap::new(Capacity::new(128)))?;

    map.bulk_put(&[
        KvEntry {
            key: Key::new(10),
            value: Value::new(100),
        },
        KvEntry {
            key: Key::new(20),
            value: Value::new(200),
        },
        KvEntry {
            key: Key::new(30),
            value: Value::new(300),
        },
    ])?;

    let keys = [Key::new(20), Key::new(10), Key::new(99)];
    let values = map.bulk_get(&keys);
    assert_eq!(
        values,
        vec![Some(Value::new(200)), Some(Value::new(100)), None]
    );

    println!("bulk_get results: {:?}", values);
    Ok(())
}
