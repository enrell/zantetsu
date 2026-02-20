//! # Zantetsu Trainer
//!
//! Training pipeline for the neural CRF model.
//! Includes data loading, model training, and model export.

pub mod data;
pub mod model;
pub mod trainer;

pub use data::{load_bio_dataset, CharVocab, TrainingExample, BIO_LABELS};
pub use model::{viterbi_decode, CrfModel, NUM_LABELS};
pub use trainer::{run_training, Trainer};
