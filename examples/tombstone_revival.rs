use gpusorted_map::{Capacity, GpuSortedMap, Key, Value};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut map = pollster::block_on(GpuSortedMap::new(Capacity::new(64)))?;

    map.put(Key::new(42), Value::new(1))?;
    assert_eq!(map.get(Key::new(42)), Some(Value::new(1)));

    map.delete(Key::new(42));
    assert_eq!(map.get(Key::new(42)), None);

    // Re-put of same key revives entry with new value.
    map.put(Key::new(42), Value::new(2))?;
    assert_eq!(map.get(Key::new(42)), Some(Value::new(2)));

    println!("revived key 42 -> {:?}", map.get(Key::new(42)));
    Ok(())
}
