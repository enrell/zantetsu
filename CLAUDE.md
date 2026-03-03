# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Zantetsu is an ultra-fast metadata extraction library for anime filenames, built in Rust with multi-language FFI bindings. It combines ML-based parsing (Neural CRF with Candle) with a heuristic fallback, and includes a vector database for semantic search. The project uses a workspace layout with 4 main crates plus tools and bindings.

**Key crates:**
- `zantetsu-core`: Main parsing engine (ML + heuristic)
- `zantetsu-vecdb`: HNSW vector index for semantic search
- `zantetsu-trainer`: RLAIF training pipeline
- `zantetsu-ffi`: FFI bindings for Node.js (napi-rs) and Python (PyO3)

**Tools:**
- `kitsu-sync`: Downloads/imports Kitsu anime database dumps
- `benchmark-compare`: Performance comparison tool
- Various Python training/data generation scripts in `tools/`

## Development Prerequisites

- Rust 1.85+ (edition 2024)
- Python 3.10+ with `uv` for Python dependencies
- PostgreSQL 12+ (for Kitsu database integration) — see `tools/kitsu-sync/README.md`
- Node.js (for bindings development)

## Build, Test, and Lint Commands

### Workspace Build & Test

```bash
# Build entire workspace
cargo build
cargo build --release

# Run all tests
cargo test --workspace

# Run tests for specific crate
cargo test -p zantetsu-core
cargo test -p zantetsu-vecdb
cargo test -p zantetsu-trainer
cargo test -p zantetsu-ffi

# Run a single test by exact name
cargo test empty_input_errors
cargo test subsplease_standard_format

# Run tests matching a pattern
cargo test episode

# Run tests without capturing output (debugging)
cargo test -- --nocapture

# Doc tests
cargo test --doc

# Benchmark
cargo bench -p zantetsu-core

# Lint with Clippy (treat warnings as errors)
cargo clippy --workspace -- -D warnings

# Format check
cargo fmt --check

# Format and fix
cargo fmt
```

### Node.js Bindings (bindings/node)

```bash
cd bindings/node

# Install dependencies
npm install

# Build TypeScript
npm run build

# Run all tests (Jest)
npm test

# Run single test by name pattern
npm test -- --testNamePattern="should parse"

# Watch mode
npm test -- --watch

# Clean build artifacts
npm run clean
```

### Python Dependencies & Scripts

```bash
# Install Python dependencies with uv
uv sync

# Run Python training/data scripts
python tools/generate_training_data.py
python tools/visualize_predictions.py
python tools/validate_gemini_output.py
python check_model.py
```

### Database Setup (Kitsu)

The `kitsu-sync` tool imports the Kitsu anime database:

```bash
# Using Rust CLI
cargo run -p kitsu-sync -- reset

# With Docker PostgreSQL (default password: root)
cargo run -p kitsu-sync -- -P root reset

# With custom connection
cargo run -p kitsu-sync -- -H localhost -U postgres -P mypassword reset

# Or using shell script
KITSU_DB_PASSWORD=root ./tools/kitsu-db-sync.sh reset
```

See `tools/kitsu-sync/README.md` for complete setup details.

## Architecture & Code Structure

```
zantetsu/
├── Cargo.toml              # Workspace root (members: 4 crates + 2 tools)
├── README.md               # Project overview, benchmarks, vision
├── AGENTS.md               # Detailed coding standards & conventions
├── pyproject.toml          # Python dependencies (torch, transformers, etc.)
├── crates/
│   ├── zantetsu-core/     # Parsing engine
│   │   ├── src/
│   │   │   ├── parser/    # HeuristicParser, NeuralParser, Tokenizer
│   │   │   ├── crf/       # Neural CRF with Viterbi
│   │   │   ├── scoring/   # Quality scoring engine
│   │   │   ├── types/     # Data structures (ParseResult, EpisodeSpec, etc.)
│   │   │   └── error.rs   # Error types (ZantetsuError)
│   │   └── Cargo.toml
│   ├── zantetsu-vecdb/    # Vector database with SQLite + HNSW
│   │   └── src/lib.rs
│   ├── zantetsu-trainer/  # Model training & RLAIF loop
│   │   └── src/
│   │       ├── model.rs   # Model definition
│   │       ├── trainer.rs # Training logic
│   │       └── data.rs    # Data processing
│   ├── zantetsu-ffi/      # FFI layer (napi-rs, PyO3, cbindgen)
│   │   ├── src/lib.rs     # Feature-gated bindings
│   │   └── src/node.rs    # Node.js bindings
│   └── (no tests in workspace root — tests are per-crate)
├── bindings/
│   ├── node/              # TypeScript wrapper around zantetsu-ffi
│   │   ├── src/
│   │   │   ├── index.ts   # Main export
│   │   │   ├── types.d.ts # TypeScript definitions
│   │   │   └── index.test.ts
│   │   └── package.json
│   └── python/            # Python bindings (planned, PyO3)
├── tools/
│   ├── kitsu-sync/        # Kitsu database sync (Rust CLI)
│   ├── benchmark-compare/ # Performance comparison tool
│   ├── annotate/          # Data annotation utilities
│   ├── train_ner/         # NER training scripts
│   └── *.py               # Various Python helper scripts
└── data/                  # Generated data and model checkpoints
```

**Data Flow:**
1. Input: Unstructured anime filename
2. Parser (core): Try heuristic parser first (fast), fallback to neural CRF (accurate)
3. Output: Structured `ParseResult` with title, episode, group, resolution, etc.
4. VecDB: Use extracted title for semantic search against Kitsu anime titles
5. Trainer: Use labeled data to improve neural model via RLAIF

**FFI Layers:**
- TypeScript: `bindings/node` uses `napi-rs` to call into `zantetsu-ffi`
- Python: PyO3 bindings are in `zantetsu-ffi` (feature-gated)
- C/C++: cbindgen support (see `zantetsu-ffi`)

## Key Dependencies

**Rust:**
- `candle-*` (ML inference)
- `rusqlite` (SQLite)
- `thiserror`/`anyhow` (error handling)
- `tokio` (async)
- `serde` (serialization)
- `tracing` (logging)
- `clap` (CLI)

**Node.js:**
- TypeScript 5.6+, Jest for testing

**Python:**
- torch, transformers, safetensors (training)
- datasets (data loading)

## Important Conventions

**See `AGENTS.md` for complete coding standards.** Key points:

- **Error handling**: Use `thiserror` for library errors, `anyhow` for application errors. Always define `pub type Result<T> = std::result::Result<T, ZantetsuError>;`
- **Logging**: Use `tracing` macros (not `println!`)
- **Naming**: `snake_case` for Rust functions/variables; `PascalCase` for types.
- **Imports order**: std → external → workspace crates → local.
- **Testing**: Inline `#[cfg(test)]` modules; use helper functions in `mod tests` block.
- **Serialization**: `#[derive(Serialize, Deserialize)]` on public types.
- **Clippy**: Pass with `-D warnings` (treat warnings as errors).
- **Formatting**: Run `cargo fmt` before committing.

**Node.js:** ES modules with `.js` extension in imports; Jest tests with `describe`/`it`; JSDoc for public API.

## Common Development Tasks

### Running a Single Test

```bash
cargo test exact_test_name
cargo test episode  # pattern match
cargo test -- --nocapture  # see output
```

### Building FFI for Specific Target

```bash
cargo build -p zantetsu-ffi --features node
cargo build -p zantetsu-ffi --features python
```

### Iterating Quickly

```bash
# Fast compile check (skip linking)
cargo check -p zantetsu-core
cargo check -p zantetsu-ffi --features node

# Watch mode (if using cargo-watch)
cargo watch -x check -x test
```

### Working with the Kitsu Database

```bash
# Check status
./tools/kitsu-db-sync.sh status

# Download only
cargo run -p kitsu-sync -- download

# Import only (requires existing dump)
cargo run -p kitsu-sync -- import

# Configure custom DB via env vars
export KITSU_DB_HOST=localhost
export KITSU_DB_PORT=5432
export KITSU_DB_NAME=kitsu_development
export KITSU_DB_USER=postgres
export KITSU_DB_PASSWORD=secret
cargo run -p kitsu-sync -- reset
```

### Adding a New Parser (Heuristic)

1. Add regex pattern in `crates/zantetsu-core/src/parser/heuristic.rs`
2. Add extraction logic in the match arms
3. Add test cases with diverse filename formats
4. Update `ParseResult` in `crates/zantetsu-core/src/types/` if new fields needed

### Adding FFI Functions

1. Add feature flag in `crates/zantetsu-ffi/Cargo.toml`
2. Implement wrapper in `crates/zantetsu-ffi/src/node.rs` with `#[no_mangle]`
3. Update `bindings/node/src/types.d.ts` with TypeScript types
4. Add tests in `bindings/node/src/index.test.ts`

### Data Generation for Training

```bash
uv sync
python tools/generate_training_data.py  # Generates synthetic examples
python tools/generate_50k_training.py  # Large dataset
python tools/fetch_nyaa_titles.py      # Scrape real titles
```

See individual script headers for usage.

## Environment Configuration

Key environment variables:
- `KITSU_DB_*`: Database connection for Kitsu sync
- `RUST_LOG`: Logging level (e.g., `debug`, `info`)
- `RAYON_NUM_THREADS`: Control parallelism in Rust

## Notices

- This is a **local-first** project — no cloud dependencies
- All inference runs on your machine
- Neural model is early-stage; heuristic parser is the default
- Kitsu database dumps are for development/personal use only (do not redistribute)

## Reference Files

- `README.md` — Project vision, features, quick start, benchmarks
- `AGENTS.md` **Read this** for comprehensive coding standards, style guidelines, and detailed architecture explanation
- `tools/kitsu-sync/README.md` — Database setup and troubleshooting
- Crate-level `README.md` files (if present) for specific component documentation
