# Zantetsu — LLM Usage Guide

This document explains how to use the **zantetsu** crate (an anime filename metadata extraction library) so that LLMs can generate correct code for it.

## What Zantetsu Does

Zantetsu parses unstructured anime torrent/filename strings and extracts structured metadata: title, episode number, release group, resolution, video codec, audio codec, media source, year, CRC32, and more.

Input: `"[SubsPlease] Sousou no Frieren - 08 (1080p) [ABCD1234].mkv"`
Output: `ParseResult { title: Some("Sousou no Frieren"), group: Some("SubsPlease"), episode: Some(Single(8)), resolution: Some(FHD1080), ... }`

## Crate Structure

| Crate | Purpose |
|---|---|
| `zantetsu-core` | Main parsing engine (heuristic + neural CRF) |
| `zantetsu-vecdb` | Vector DB for semantic title search (stub, not yet available) |
| `zantetsu-ffi` | FFI bindings for Node.js (napi-rs) and Python (PyO3) |
| `bindings/node` | TypeScript/Node.js wrapper with pure-JS fallback |

## Quick Start

### Rust

Add to `Cargo.toml`:
```toml
[dependencies]
zantetsu-core = "0.1"
```

Parse a filename:
```rust
use zantetsu_core::{parse, ParseResult};

let result: ParseResult = parse("[SubsPlease] Sousou no Frieren - 08 (1080p) [ABCD1234].mkv")?;
println!("{}", result.title.unwrap()); // "Sousou no Frieren"
```

### Node.js

```typescript
import { parse, HeuristicParser } from 'zantetsu';

// Convenience (uses a default parser singleton)
const result = parse('[SubsPlease] Sousou no Frieren - 08 (1080p) [ABCD1234].mkv');
console.log(result.title); // "Sousou no Frieren"

// Or create your own parser instance
const parser = new HeuristicParser();
const results = parser.parseBatch([
  '[SubsPlease] Sousou no Frieren - 08 (1080p) [ABCD1234].mkv',
  '[Erai-raws] Jujutsu Kaisen S2 - 12 [1080p][Multiple Subtitle].mkv',
]);
```

---

## Core API

### Convenience Functions (Rust)

```rust
use zantetsu_core::{parse, parse_with_mode, ParseMode, ParseResult};

// Parse with default settings (Auto mode — tries heuristic first)
let result: ParseResult = parse("filename.mkv")?;

// Parse with a specific mode
let result: ParseResult = parse_with_mode("filename.mkv", ParseMode::Light)?;
```

### Parser Struct (Rust)

For more control, use the `Parser` struct directly:

```rust
use zantetsu_core::{Parser, ParserConfig, ParseMode};

// Default config
let parser = Parser::default()?;
let result = parser.parse("[SubsPlease] Frieren - 08 (1080p).mkv")?;

// Custom config
let config = ParserConfig::new()
    .with_mode(ParseMode::Light)
    .with_confidence_threshold(0.7)
    .with_neural(false);

let parser = Parser::new(config)?;
let result = parser.parse("filename.mkv")?;
```

### HeuristicParser (Rust)

Direct access to the fast regex-based parser (no ML overhead):

```rust
use zantetsu_core::HeuristicParser;

let parser = HeuristicParser::new()?;
let result = parser.parse("[SubsPlease] Frieren - 08 (1080p).mkv")?;
```

### NeuralParser (Rust)

Direct access to the Neural CRF parser (requires model weights):

```rust
use zantetsu_core::NeuralParser;

let mut parser = NeuralParser::new()?;
parser.init_model()?; // Loads model weights from default path
let result = parser.parse("filename.mkv")?;
```

---

## Types Reference

### `ParseResult`

The primary output type. All fields except `input`, `confidence`, and `parse_mode` are `Option<T>`.

```rust
pub struct ParseResult {
    pub input: String,           // Original input
    pub title: Option<String>,   // Normalized anime title
    pub group: Option<String>,   // Release group (e.g., "SubsPlease")
    pub episode: Option<EpisodeSpec>,
    pub season: Option<u32>,
    pub resolution: Option<Resolution>,
    pub video_codec: Option<VideoCodec>,
    pub audio_codec: Option<AudioCodec>,
    pub source: Option<MediaSource>,
    pub year: Option<u16>,
    pub crc32: Option<String>,   // Hex string
    pub extension: Option<String>,
    pub version: Option<u8>,     // e.g., v2 = 2
    pub confidence: f32,         // [0.0, 1.0]
    pub parse_mode: ParseMode,
}
```

Methods:
- `has_title() -> bool` — true if a title was extracted
- `has_metadata() -> bool` — true if any metadata beyond the title was found

### `EpisodeSpec`

```rust
pub enum EpisodeSpec {
    Single(u32),              // "08"
    Range(u32, u32),          // "01-12"
    Multi(Vec<u32>),          // "01, 03, 05"
    Version { episode: u32, version: u8 }, // "12v2"
}
```

Display: `Single(8)` → `"08"`, `Range(1, 12)` → `"01-12"`, `Version { episode: 12, version: 2 }` → `"12v2"`

### `Resolution`

```rust
pub enum Resolution {
    SD480,    // 480p  — score: 0.25
    HD720,    // 720p  — score: 0.50
    FHD1080,  // 1080p — score: 0.85
    UHD2160,  // 2160p — score: 1.00
}
```

### `VideoCodec`

```rust
pub enum VideoCodec {
    H264,    // score: 0.60
    HEVC,    // score: 0.85
    AV1,     // score: 1.00
    VP9,     // score: 0.70
    MPEG4,   // score: 0.20
}
```

### `AudioCodec`

```rust
pub enum AudioCodec {
    FLAC,    // score: 0.95
    AAC,     // score: 0.60
    Opus,    // score: 0.70
    AC3,     // score: 0.50
    DTS,     // score: 0.75
    MP3,     // score: 0.30
    Vorbis,  // score: 0.45
    TrueHD,  // score: 1.00
    EAAC,    // score: 0.55
}
```

### `MediaSource`

```rust
pub enum MediaSource {
    BluRayRemux, // score: 1.00
    BluRay,      // score: 0.90
    WebDL,       // score: 0.75
    WebRip,      // score: 0.65
    HDTV,        // score: 0.50
    DVD,         // score: 0.40
    LaserDisc,   // score: 0.30
    VHS,         // score: 0.15
}
```

### `ParseMode`

```rust
pub enum ParseMode {
    Full,   // Neural CRF inference (requires model weights)
    Light,  // Regex + scene rules only (fast, no ML)
    Auto,   // Default — tries heuristic, falls back to neural if confidence is low
}
```

---

## Quality Scoring

Zantetsu includes a scoring engine to rank parsed releases by quality.

```rust
use zantetsu_core::{QualityScores, QualityProfile, ClientContext, DeviceType, NetworkQuality, QualityProfile};

// Build scores from parsed metadata
let scores = QualityScores::from_metadata(
    result.resolution,
    result.video_codec,
    result.audio_codec,
    result.source,
    0.8, // group trust score [0.0, 1.0]
);

// Compute weighted score with default profile
let profile = QualityProfile::default();
let quality: f32 = scores.compute(&profile); // returns [0.0, 1.0]
```

### QualityProfile (Default Weights)

| Dimension | Weight |
|---|---|
| Resolution | 0.35 |
| Video Codec | 0.25 |
| Audio Codec | 0.15 |
| Source | 0.15 |
| Group Trust | 0.10 |

### ClientContext (Device/Network-Aware Scoring)

```rust
use zantetsu_core::{ClientContext, DeviceType, NetworkQuality, VideoCodec};

let ctx = ClientContext {
    device_type: DeviceType::Mobile,
    network: NetworkQuality::Limited,
    hw_decode_codecs: vec![VideoCodec::H264, VideoCodec::HEVC],
};

// Adjusts scores based on device/network constraints
let adjusted = ctx.adjust_score(scores, Some(VideoCodec::HEVC));
let final_score = adjusted.compute(&profile);
```

### DeviceType

```rust
pub enum DeviceType {
    Desktop,   // No penalty
    Laptop,    // Slight preference for 1080p over 4K
    Mobile,    // Strong preference for 720p
    TV,        // Preference for highest resolution
    Embedded,  // SD/720p cap
}
```

### NetworkQuality

```rust
pub enum NetworkQuality {
    Unlimited,  // No constraints
    Broadband,  // Slight penalty for 4K remux
    Limited,    // Strong penalty for large files
    Offline,    // Only locally cached files
}
```

---

## Error Handling

All fallible operations return `zantetsu_core::Result<T>`, which is `std::result::Result<T, ZantetsuError>`.

```rust
use zantetsu_core::{ZantetsuError, Result};

match parse("") {
    Ok(result) => { /* use result */ }
    Err(ZantetsuError::EmptyInput) => { /* input was empty */ }
    Err(ZantetsuError::ParseFailed { input }) => { /* could not extract metadata */ }
    Err(ZantetsuError::ModelLoadError(msg)) => { /* neural model not available */ }
    Err(e) => { /* other error */ }
}
```

### `ZantetsuError` Variants

| Variant | When |
|---|---|
| `EmptyInput` | Input is empty or whitespace-only |
| `ParseFailed { input }` | Parser could not extract any metadata |
| `RegexError` | Internal regex compilation failure |
| `ModelLoadError(String)` | Failed to load neural model weights |
| `InferenceError(String)` | Neural model inference failed |
| `InvalidContext(String)` | Invalid scoring context |
| `NeuralParser(String)` | Neural parser error |
| `CandleError(String)` | ML framework error |

---

## TypeScript API Reference

### Exports

```typescript
import {
  parse,
  parseBatch,
  HeuristicParser,
  isUsingNativeModule,
  // Types
  ParseResult,
  EpisodeSpec,
  Resolution,
  VideoCodec,
  AudioCodec,
  MediaSource,
  ParseMode,
  HeuristicParserOptions,
} from 'zantetsu';
```

### Functions

```typescript
// Parse a single filename with default parser
const result: ParseResult = parse('[SubsPlease] Frieren - 08 (1080p).mkv');

// Parse multiple filenames
const results: ParseResult[] = parseBatch(['file1.mkv', 'file2.mkv']);

// Check if native addon is loaded
const native: boolean = isUsingNativeModule();
```

### Class: HeuristicParser

```typescript
const parser = new HeuristicParser({ debug: true });
const result: ParseResult = parser.parse('filename.mkv');
const results: ParseResult[] = parser.parseBatch(['a.mkv', 'b.mkv']);
```

### TypeScript Types

```typescript
type Resolution = 'SD480' | 'HD720' | 'FHD1080' | 'UHD2160';
type VideoCodec = 'H264' | 'HEVC' | 'AV1' | 'VP9' | 'MPEG4';
type AudioCodec = 'FLAC' | 'AAC' | 'Opus' | 'AC3' | 'DTS' | 'MP3' | 'Vorbis' | 'TrueHD' | 'EAAC';
type MediaSource = 'BluRayRemux' | 'BluRay' | 'WebDL' | 'WebRip' | 'HDTV' | 'DVD' | 'LaserDisc' | 'VHS';
type ParseMode = 'Full' | 'Light' | 'Auto';

type EpisodeSpec =
  | { type: 'single'; episode: number }
  | { type: 'range'; start: number; end: number }
  | { type: 'multi'; episodes: number[] }
  | { type: 'versioned'; episode: number; version: number };

interface ParseResult {
  input: string;
  title: string | null;
  group: string | null;
  episode: EpisodeSpec | null;
  season: number | null;
  resolution: Resolution | null;
  video_codec: VideoCodec | null;
  audio_codec: AudioCodec | null;
  source: MediaSource | null;
  year: number | null;
  crc32: string | null;
  extension: string | null;
  version: number | null;
  confidence: number;
  parse_mode: ParseMode;
}
```

Note: In the TypeScript API, enum values are returned as plain strings (e.g., `"FHD1080"`, `"HEVC"`), not Rust enum variants.

---

## Common Patterns

### Parse and Score a Release

```rust
use zantetsu_core::{parse, QualityScores, QualityProfile};

let result = parse("[SubsPlease] Frieren - 08 (1080p FLAC) [ABCD1234].mkv")?;
let scores = QualityScores::from_metadata(
    result.resolution, result.video_codec,
    result.audio_codec, result.source, 0.8,
);
let quality = scores.compute(&QualityProfile::default());
println!("{}: quality={:.2}", result.title.as_deref().unwrap_or("?"), quality);
```

### Batch Parse with Error Handling

```rust
use zantetsu_core::{parse, ZantetsuError};

let filenames = vec![
    "[SubsPlease] Frieren - 08 (1080p).mkv",
    "",
    "[Erai-raws] Jujutsu Kaisen - 12 [1080p].mkv",
];

for fname in &filenames {
    match parse(fname) {
        Ok(r) => println!("{} -> {:?}", fname, r.title),
        Err(ZantetsuError::EmptyInput) => eprintln!("skipped empty input"),
        Err(e) => eprintln!("error parsing {}: {}", fname, e),
    }
}
```

### Use Light Mode Only (No ML)

```rust
use zantetsu_core::{Parser, ParserConfig, ParseMode};

let parser = Parser::new(
    ParserConfig::new()
        .with_mode(ParseMode::Light)
        .with_neural(false)
)?;
let result = parser.parse("filename.mkv")?;
```

---

## Key Conventions for Code Generation

1. **Always use `?` or `.unwrap()` on `parse()` calls** — they return `Result<ParseResult>`.
2. **Fields are `Option<T>`** — always check for `None` before accessing metadata fields.
3. **`EpisodeSpec` is a discriminated union** — match on all variants (`Single`, `Range`, `Multi`, `Version`).
4. **In TypeScript, enums are plain strings** — compare with string literals like `'FHD1080'`.
5. **`HeuristicParser::new()` can fail** — it compiles regexes at construction time.
6. **`NeuralParser` requires model weights** — call `init_model()` before `parse()`, or use `Parser` with `Auto` mode for automatic fallback.
7. **Quality scores are `[0.0, 1.0]`** — higher is better. Missing dimensions default to `0.5` (neutral).
