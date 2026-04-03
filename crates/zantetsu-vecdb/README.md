# zantetsu-vecdb

Semantic title mapping via embedded vector search.

## Features

- **HNSW Index**: Approximate nearest neighbor search for fast semantic matching
- **Hybrid Search**: Combines semantic similarity with lexical matching
- **SQLite Storage**: Persistent vector index with embedded database
- **Title Normalization**: Maps extracted titles to canonical anime names

## Usage

```rust
use zantetsu_vecdb::VecDb;

let db = VecDb::open("anime_vectors.db").unwrap();

// Search by semantic similarity
let results = db.search("spy x family", 5).unwrap();
for (title, score) in &results {
    println!("{}: {:.2}", title, score);
}
```

## License

MIT
