# zantetsu-core

ML-based anime metadata extraction and normalization engine.

## Features

- **Heuristic Parser**: Fast regex-based extraction for production use (~92% accuracy)
- **Neural CRF Parser**: Candle-based Viterbi decoder for maximum accuracy
- **Quality Scoring**: Configurable profiles for release validation
- **Zero-copy**: Sub-millisecond parsing with minimal allocations

## Usage

```rust
use zantetsu_core::{HeuristicParser, ParseResult};

let parser = HeuristicParser::default();
let result: ParseResult = parser.parse("[SubsPlease] Spy x Family - 01 (1080p).mkv").unwrap();

assert_eq!(result.title, Some("Spy x Family".to_string()));
assert_eq!(result.resolution, Some("FHD1080".to_string()));
assert_eq!(result.group, Some("SubsPlease".to_string()));
```

## Supported Formats

- Sub-group notation: `[Group] Title - Ep.ext`
- Scene naming: `Title.S01E01.1080p.WEB-DL.AAC2.0.H.264.ext`
- Batch paths: `.../Group/Title/Group Title - Ep.ext`
- Multi-episode: `Ep01-Ep05`, `Ep01-05`
- Subtitle variants, FLAC, Blu-Ray, WEB, DVD, TV sources

## License

MIT
