use gpusorted_map::{Capacity, GpuSortedMap, Key, KvEntry, Value};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut map = pollster::block_on(GpuSortedMap::new(Capacity::new(128)))?;

    map.bulk_put(&[
        KvEntry {
            key: Key::new(5),
            value: Value::new(50),
        },
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

    let entries = map.range(Key::new(10), Key::new(30));
    let keys: Vec<Key> = entries.iter().map(|entry| entry.key).collect();
    assert_eq!(keys, vec![Key::new(10), Key::new(20)]);

    println!("range keys: {:?}", keys);
    Ok(())
}
