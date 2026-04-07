# zantetsu-ffi

Multi-language FFI bindings for Zantetsu.

## Crates

- [`zantetsu`](https://crates.io/crates/zantetsu) - unified API surface
- [`zantetsu-core`](https://crates.io/crates/zantetsu-core) - parsing engine
- [`zantetsu-vecdb`](https://crates.io/crates/zantetsu-vecdb) - canonical title matching
- [`zantetsu-trainer`](https://crates.io/crates/zantetsu-trainer) - training workflows
- [`zantetsu-ffi`](https://crates.io/crates/zantetsu-ffi) - Node/Python/C bindings

## Features

| Feature  | Description |
| -------- | ----------- |
| `node`   | Node.js bindings via napi-rs |
| `python` | Python bindings via PyO3 |

## Usage

### Node.js

```toml
[dependencies]
zantetsu-ffi = { version = "0.1.4", features = ["node"] }
```

```rust
use zantetsu_ffi::HeuristicParserNode;

let parser = HeuristicParserNode::new()?;
let result = parser.parse("[SubsPlease] Spy x Family - 01 (1080p).mkv".to_string())?;
```

### Python

```toml
[dependencies]
zantetsu-ffi = { version = "0.1.4", features = ["python"] }
```

```rust
use zantetsu_ffi::HeuristicParserPy;

let parser = HeuristicParserPy::new()?;
let result = parser.parse("[SubsPlease] Spy x Family - 01 (1080p).mkv".to_string())?;
```

## License

MIT
