# zantetsu

Unified public API for Zantetsu.

## Features

- Parse anime filenames through the heuristic and neural parser stack
- Match parsed titles through local Kitsu dumps or a remote endpoint
- Re-export the core types needed by downstream applications

## Crates

- [`zantetsu`](https://crates.io/crates/zantetsu) - unified API surface
- [`zantetsu-core`](https://crates.io/crates/zantetsu-core) - parsing engine
- [`zantetsu-vecdb`](https://crates.io/crates/zantetsu-vecdb) - canonical title matching
- [`zantetsu-trainer`](https://crates.io/crates/zantetsu-trainer) - training workflows
- [`zantetsu-ffi`](https://crates.io/crates/zantetsu-ffi) - Node/Python/C bindings

## Usage

```rust
use zantetsu::{MatchSource, TitleMatcher, Zantetsu};

let engine = Zantetsu::new().unwrap();
let parsed = engine.parse("[SubsPlease] Spy x Family - 01 (1080p).mkv").unwrap();

let matcher = TitleMatcher::new(MatchSource::remote_endpoint("https://graphql.anilist.co")).unwrap();
let best = matcher.match_title(parsed.title.as_deref().unwrap()).unwrap();

assert!(best.is_some());
```

## License

MIT
