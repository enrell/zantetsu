//! # Zantetsu Trainer
//!
//! Training pipeline for the neural CRF model.
//! Includes data loading, model training, and model export.

pub mod data;
pub mod model;
pub mod trainer;

pub use data::{BIO_LABELS, CharVocab, TrainingExample, load_bio_dataset};
pub use model::{CrfModel, NUM_LABELS, viterbi_decode};
pub use trainer::{Trainer, run_training};
