# Zantetsu — The Omniscient Metadata Engine

<p align="center">
  <strong>Ultra-fast, intelligent library for anime metadata extraction and normalization</strong>
</p>

<p align="center">
  <a href="https://github.com/kokoro/zantetsu/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"></a>
  <a href="https://rust-lang.org"><img src="https://img.shields.io/badge/Rust-1.85+-dea584.svg?logo=rust" alt="Rust 1.85+"></a>
</p>

---

## Vision

Zantetsu transforms chaotic, unstructured media data (torrent names, file names, release group conventions) into perfectly organized, actionable metadata. All inference runs entirely on your machine — **local-first, zero cloud dependencies**.

## Features

- **ML-Based Parsing**: Neural CRF model with Viterbi decoding for accurate metadata extraction
- **Heuristic Fallback**: Fast regex-based parsing for constrained environments
- **Semantic Search**: HNSW vector index with hybrid (semantic + lexical) search
- **Quality Scoring**: Configurable quality profiles for release validation
- **FFI Bindings**: Native bindings for TypeScript, Python, and C/C++
- **Self-Improving**: RLAIF training loop for autonomous model improvement

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Consumer Applications                 │
│         (TypeScript / Python / C++ / CLI)               │
└───────────────┬─────────────────────────┬───────────────┘
                │         FFI             │
┌───────────────▼─────────────────────────▼───────────────┐
│                    zantetsu-ffi                         │
│              (napi-rs / PyO3 / cbindgen)                │
└───────────────┬─────────────────────────┬───────────────┘
                │                         │
┌───────────────▼───────────┐ ┌───────────▼────────────────┐
│      zantetsu-core        │ │      zantetsu-vecdb        │
│  ┌─────────────────────┐  │ │  ┌─────────────────────┐   │
│  │  Neural CRF Engine  │  │ │  │  HNSW Vector Index  │   │
│  │  (candle + Viterbi) │  │ │  │  (all-MiniLM-L6-v2) │   │
│  ├─────────────────────┤  │ │  ├─────────────────────┤   │
│  │  Heuristic Fallback │  │ │  │  Hybrid Search      │   │
│  ├─────────────────────┤  │ │  ├─────────────────────┤   │
│  │  Scoring Engine     │  │ │  │  SQLite Cache       │   │
│  └─────────────────────┘  │ │  └─────────────────────┘   │
└───────────────────────────┘ └────────────────────────────┘
                │                         │
┌───────────────▼─────────────────────────▼────────────────┐
│                  zantetsu-trainer                        │
│  ┌──────────────────┐  ┌──────────────────────────────┐  │
│  │  Dump Syncer     │  │  RLAIF Training Loop         │  │
│  │  (Kitsu/AniList) │  │  (candle fine-tuning)        │  │
│  └──────────────────┘  └──────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

## Crates

| Crate | Description |
|-------|-------------|
| `zantetsu-core` | ML parser + heuristic fallback + scoring |
| `zantetsu-vecdb` | Semantic vector search with HNSW |
| `zantetsu-trainer` | Model training and RLAIF workflows |
| `zantetsu-ffi` | Multi-language bindings |

## Tech Stack

- **Runtime**: Rust 1.85+
- **ML Inference**: Candle
- **Vector Search**: HNSW (all-MiniLM-L6-v2)
- **Storage**: SQLite (rusqlite)
- **Async**: Tokio
- **CLI**: Clap

## Getting Started

```bash
# Build the project
cargo build --release

# Run the CLI
cargo run --release -- --help
```

## Design Principles

1. **Local-First**: All operations run on your machine
2. **Performance Obsession**: Sub-millisecond parsing, zero-copy where possible
3. **Correctness Over Heuristics**: ML-backed parsing replaces fragile regex
4. **Universal Interop**: One engine, multiple languages
5. **Self-Improving**: Autonomous model improvement via RLAIF

## License

[MIT](LICENSE)
