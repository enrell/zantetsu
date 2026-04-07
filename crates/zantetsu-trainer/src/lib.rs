//! # Zantetsu Trainer
//!
//! Training pipeline for the neural CRF model.
//! Includes data loading, model training, and model export.
//!
//! Crates:
//! - [`zantetsu`](https://docs.rs/zantetsu) - unified API surface
//! - [`zantetsu-core`](https://docs.rs/zantetsu-core) - parsing engine
//! - [`zantetsu-vecdb`](https://docs.rs/zantetsu-vecdb) - canonical title matching
//! - [`zantetsu-trainer`](https://docs.rs/zantetsu-trainer) - training workflows
//! - [`zantetsu-ffi`](https://docs.rs/zantetsu-ffi) - Node/Python/C bindings
//!
//! Use this crate when you need dataset bootstrapping, model training, or evaluation workflows.

pub mod data;
pub mod model;
pub mod trainer;

pub use data::{BIO_LABELS, CharVocab, TrainingExample, load_bio_dataset};
pub use model::{CrfModel, NUM_LABELS, viterbi_decode};
pub use trainer::{Trainer, run_training};
