# zantetsu-trainer

Background sync and RLAIF training loop for Zantetsu.

## Crates

- [`zantetsu`](https://crates.io/crates/zantetsu) - unified API surface
- [`zantetsu-core`](https://crates.io/crates/zantetsu-core) - parsing engine
- [`zantetsu-vecdb`](https://crates.io/crates/zantetsu-vecdb) - canonical title matching
- [`zantetsu-trainer`](https://crates.io/crates/zantetsu-trainer) - training workflows
- [`zantetsu-ffi`](https://crates.io/crates/zantetsu-ffi) - Node/Python/C bindings

## Features

- **Data Sync**: Downloads anime metadata from Kitsu database
- **RLAIF Loop**: Reinforcement learning from AI feedback for model improvement
- **Candle Fine-tuning**: Native Rust model training pipeline
- **Evaluation**: Automated scoring against ground truth data

## Usage

```bash
# Run the trainer binary
cargo run -p zantetsu-trainer --bin train -- --help
```

```rust
use zantetsu_trainer::Trainer;

let trainer = Trainer::default();
assert!(trainer.batch_size > 0);
```

## License

MIT
