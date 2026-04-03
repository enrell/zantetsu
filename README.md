# Zantetsu — The Omniscient Metadata Engine

<p align="center">
  <strong>Ultra-fast, intelligent library for anime metadata extraction and normalization</strong>
</p>

<p align="center">
  <a href="https://github.com/enrell/zantetsu/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"></a>
  <a href="https://rust-lang.org"><img src="https://img.shields.io/badge/Rust-1.85+-dea584.svg?logo=rust" alt="Rust 1.85+"></a>
</p>

---

## Vision

Zantetsu transforms chaotic, unstructured media data (torrent names, file names, release group conventions) into perfectly organized, actionable metadata. All inference runs entirely on your machine — **local-first, zero cloud dependencies**.

## Features

- **Multi-Model Parsing**: Heuristic parser (production-ready) + Neural CRF (DistilBERT) + Character CNN (in development)
- **RAD-Augmented Training**: Reinforcement Learning with Augmented Data — 430k+ synthetic samples with character-level augmentations
- **Semantic Search**: HNSW vector index with hybrid (semantic + lexical) search
- **AnimeDB Validation**: Ground-truth validation against AnimeDB API for accuracy measurement
- **Quality Scoring**: Configurable quality profiles for release validation
- **FFI Bindings**: Native bindings for TypeScript, Python, and C/C++
- **Self-Improving**: RLAIF training loop for autonomous model improvement

## Benchmarks

Latest benchmark run (`tools/benchmark_compare.py`, March 2026) on 148 tricky anime filenames:

| Parser | Avg Score | Min Score |
|--------|-----------|-----------|
| **Zantetsu (Heuristic)** | **92.38%** | **79.09%** |
| Python Torrent Tools (PTT) | 86.66% | 36.36% |
| Release Title Normalizer (RTN) | 84.94% | 27.27% |
| Zantetsu (Neural CRF, early) | 50.39% | 18.18% |

**Detailed Heuristic Parser Accuracy:**
- Group extraction: 100%
- Title extraction: 92.6%
- Episode extraction: 89.2%
- Resolution extraction: 95.9%
- Perfect parses: 16.9%

**AnimeDB Validation**: 58% match rate on sample data (title extraction vs ground truth)

## Training Data

- **431,249 samples** generated from 22,179 unique anime titles via AnimeDB API
- **RAD augmentations**: character substitution, masking, noise injection, case variation, spacing variation
- Dataset: `data/training/rad_dataset_50k.jsonl` (413MB)

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
│  │  Heuristic Parser   │  │ │  │  HNSW Vector Index  │   │
│  │  (regex + rules)    │  │ │  │  (all-MiniLM-L6-v2) │   │
│  ├─────────────────────┤  │ │  ├─────────────────────┤   │
│  │  Neural CRF Engine  │  │ │  │  Hybrid Search      │   │
│  │  (DistilBERT+Viterbi)│ │ │  ├─────────────────────┤   │
│  ├─────────────────────┤  │ │  │  SQLite Cache       │   │
│  │  Character CNN      │  │ │  └─────────────────────┘   │
│  │  (CNN+BiLSTM+CRF)   │  │ │                             │
│  ├─────────────────────┤  │ │                             │
│  │  Scoring Engine     │  │ │                             │
│  └─────────────────────┘  │ │                             │
└───────────────────────────┘ └─────────────────────────────┘
                │                         │
┌───────────────▼─────────────────────────▼────────────────┐
│                  zantetsu-trainer                        │
│  ┌──────────────────┐  ┌──────────────────────────────┐  │
│  │  Dump Syncer     │  │  RLAIF Training Loop         │  │
│  │  (Kitsu/AniList) │  │  (candle fine-tuning)        │  │
│  └──────────────────┘  └──────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
                │
┌───────────────▼──────────────────────────────────────────┐
│                  AnimeDB API (local)                     │
│  ┌──────────────────┐  ┌──────────────────────────────┐  │
│  │  Title Validation│  │  Data Generation (RAD)       │  │
│  │  (ground truth)  │  │  (430k+ samples)             │  │
│  └──────────────────┘  └──────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

## Crates & Tools

| Component | Type | Description |
|-----------|------|-------------|
| `zantetsu` | Crate | **Main entry point** — unified API combining all parsers |
| `zantetsu-core` | Crate | ML parser + heuristic fallback + scoring |
| `zantetsu-vecdb` | Crate | Semantic vector search with HNSW |
| `zantetsu-trainer` | Crate | Model training and RLAIF workflows |
| `zantetsu-ffi` | Crate | Multi-language bindings |
| `kitsu-sync` | Tool | Kitsu database dump downloader/importer |
| `benchmark-compare` | Tool | Parser performance comparison |

## Tech Stack

- **Runtime**: Rust 1.85+
- **ML Inference**: Candle (DistilBERT) + Character CNN (in development)
- **Vector Search**: HNSW (all-MiniLM-L6-v2)
- **Storage**: SQLite (rusqlite)
- **Async**: Tokio
- **CLI**: Clap
- **Training**: PyTorch + RAD augmentations → ONNX export

## Getting Started

### Prerequisites

- Rust 1.85+ with Cargo
- Python 3.10+ (for training/tools)
- [uv](https://github.com/astral-sh/uv) (recommended for Python dependency management)
- PostgreSQL 12+ (or Docker) for anime metadata
- ROCm/CUDA GPU (recommended for training)

### Quick Start

```bash
# Install Python dependencies
uv sync

# Build the project
cargo build --release

# Download and import Kitsu anime database
# Using Docker PostgreSQL (password: root)
cargo run -p kitsu-sync -- -P root reset

# Or with custom database
cargo run -p kitsu-sync -- -H localhost -U postgres -P mypassword reset

# Run tests
cargo test --workspace
```

### Training Character CNN

```bash
# Train with RAD-augmented data (GPU)
python tools/train_char_cnn.py --epochs 20 --batch_size 64

# Train with subset for debugging (CPU)
python tools/train_char_cnn.py --epochs 3 --batch_size 16 --max_samples 10000 --no_crf

# Validate parser against AnimeDB
python tools/validate_with_anidb.py

# Generate more training data
python tools/generate_rad_data.py
```

### Database Setup

Zantetsu uses the Kitsu anime database for semantic search and ground truth:

```bash
# Using Docker
export KITSU_DB_PASSWORD=root
cargo run -p kitsu-sync -- reset

# Or using the shell script
KITSU_DB_PASSWORD=root ./tools/kitsu-db-sync.sh reset
```

See [tools/kitsu-sync/README.md](tools/kitsu-sync/README.md) for detailed setup instructions.

## Design Principles

1. **Local-First**: All operations run on your machine
2. **Performance Obsession**: Sub-millisecond parsing, zero-copy where possible
3. **Correctness Over Heuristics**: ML-backed parsing replaces fragile regex
4. **Universal Interop**: One engine, multiple languages
5. **Self-Improving**: Autonomous model improvement via RLAIF
6. **No LLMs**: Top performance achieved through traditional ML + data augmentation

## License

[MIT](LICENSE)
