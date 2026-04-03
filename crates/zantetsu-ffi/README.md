# zantetsu-ffi

Multi-language FFI bindings for Zantetsu.

## Features

| Feature  | Description |
| -------- | ----------- |
| `node`   | Node.js bindings via napi-rs |
| `python` | Python bindings via PyO3 |

## Usage

### Node.js

```toml
[dependencies]
zantetsu-ffi = { version = "0.1", features = ["node"] }
```

```rust
use zantetsu_ffi::HeuristicParserNode;

let parser = HeuristicParserNode::new()?;
let result = parser.parse("[SubsPlease] Spy x Family - 01 (1080p).mkv")?;
```

### Python

```toml
[dependencies]
zantetsu-ffi = { version = "0.1", features = ["python"] }
```

```rust
use zantetsu_ffi::HeuristicParserPy;

let parser = HeuristicParserPy::new()?;
let result = parser.parse("[SubsPlease] Spy x Family - 01 (1080p).mkv")?;
```

## License

MIT