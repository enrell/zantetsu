# zantetsu-trainer

Background sync and RLAIF training loop for Zantetsu.

## Features

- **Data Sync**: Downloads anime metadata from Kitsu database
- **RLAIF Loop**: Reinforcement learning from AI feedback for model improvement
- **Candle Fine-tuning**: Native Rust model training pipeline
- **Evaluation**: Automated scoring against ground truth data

## Usage

```bash
# Sync Kitsu database and train model
cargo run -p zantetsu-trainer -- train

# Evaluate current model
cargo run -p zantetsu-trainer -- eval
```

## License

MIT
