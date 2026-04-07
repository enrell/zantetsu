//! # Zantetsu Core
//!
//! The heart of the Zantetsu metadata engine. Provides ML-based and heuristic
//! anime metadata extraction, quality scoring, and structured data types.
//!
//! Crates:
//! - [`zantetsu`](https://docs.rs/zantetsu) - unified API surface
//! - [`zantetsu-core`](https://docs.rs/zantetsu-core) - parsing engine
//! - [`zantetsu-vecdb`](https://docs.rs/zantetsu-vecdb) - canonical title matching
//! - [`zantetsu-trainer`](https://docs.rs/zantetsu-trainer) - training workflows
//! - [`zantetsu-ffi`](https://docs.rs/zantetsu-ffi) - Node/Python/C bindings
//!
//! ## Quick Start
//!
//! ```rust
//! use zantetsu_core::parser::HeuristicParser;
//!
//! let parser = HeuristicParser::new().unwrap();
//! let result = parser.parse("[SubsPlease] Jujutsu Kaisen - 24 (1080p) [A1B2C3D4].mkv").unwrap();
//!
//! assert_eq!(result.title.as_deref(), Some("Jujutsu Kaisen"));
//! assert_eq!(result.group.as_deref(), Some("SubsPlease"));
//! ```
//!
//! See the crate README on crates.io for feature overview and supported filename patterns.
pub mod crf;
pub mod error;
pub mod parser;
pub mod scoring;
pub mod types;

// Re-export primary API
pub use error::{Result, ZantetsuError};
pub use parser::{
    BioTag, HeuristicParser, NeuralParser, Parser, ParserConfig, Tokenizer, ViterbiDecoder,
};
pub use scoring::{ClientContext, DeviceType, NetworkQuality, QualityProfile, QualityScores};
pub use types::{
    AudioCodec, EpisodeSpec, MediaSource, ParseMode, ParseResult, Resolution, VideoCodec,
};
