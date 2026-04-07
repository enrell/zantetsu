//! # Zantetsu
//!
//! Ultra-fast, intelligent library for anime metadata extraction and normalization.
//!
//! ## Features
//!
//! - **Heuristic Parsing**: Regex-based parsing for fast, reliable extraction
//! - **Neural CRF**: DistilBERT + CRF model for accurate sequence labeling
//! - **Character CNN**: CNN + BiLSTM + CRF for robust character-level parsing (in development)
//! - **Semantic Search**: HNSW vector index for title matching
//! - **Quality Scoring**: Configurable quality profiles for release validation
//!
//! ## Quick Start
//!
//! ```rust
//! use zantetsu::{EpisodeSpec, Zantetsu};
//!
//! let engine = Zantetsu::new()?;
//! let result = engine.parse("[SubsPlease] Cowboy Bebop - 01 [1080p][HEVC].mkv")?;
//!
//! assert_eq!(result.title.as_deref(), Some("Cowboy Bebop"));
//! assert_eq!(result.group.as_deref(), Some("SubsPlease"));
//! assert_eq!(result.episode, Some(EpisodeSpec::Single(1)));
//! assert!(result.resolution.is_some());
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## Architecture
//!
//! Zantetsu combines multiple parsing strategies:
//!
//! 1. **Heuristic Parser** — Fast regex-based parsing (production-ready, 92.38% accuracy)
//! 2. **Neural CRF** — DistilBERT + CRF with Viterbi decoding (early stage)
//! 3. **Character CNN** — Lightweight CNN + BiLSTM + CRF with RAD augmentations (in development)
//!
//! The engine automatically selects the best parser based on availability and confidence.
//!
//! ## Crates
//!
//! - `zantetsu-core` — Parsing engine (heuristic + neural + character CNN)
//! - `zantetsu-vecdb` — Canonical title matching via Kitsu dumps or remote endpoints
//! - `zantetsu-trainer` — Model training and RLAIF workflows
//! - `zantetsu-ffi` — Multi-language bindings (TypeScript, Python, C/C++)

pub use zantetsu_core::error::{Result, ZantetsuError};
pub use zantetsu_core::parser::{HeuristicParser, NeuralParser};
pub use zantetsu_core::scoring::{QualityProfile, QualityScores};
pub use zantetsu_core::types::{
    AudioCodec, EpisodeSpec, MediaSource, ParseMode, ParseResult, Resolution, VideoCodec,
};
pub use zantetsu_vecdb::{
    AnimeIds, AnimeTitleMatch, MatchProvider, MatchSource, MatchResult, MatcherError,
    TitleMatcher, default_kitsu_dump_dir,
};

/// Main entry point for the Zantetsu parsing engine.
///
/// Combines heuristic, neural, and character-based parsing with semantic search.
pub struct Zantetsu {
    heuristic: HeuristicParser,
    neural: Option<NeuralParser>,
}

impl Zantetsu {
    /// Create a new Zantetsu engine instance.
    ///
    /// The heuristic parser is always initialized. The neural parser is
    /// initialized lazily if model weights are available.
    pub fn new() -> Result<Self> {
        let heuristic = HeuristicParser::new()?;
        let neural = NeuralParser::new().ok();

        Ok(Self { heuristic, neural })
    }

    /// Parse an anime filename using the best available parser.
    ///
    /// Tries the heuristic parser first (fast), then falls back to the neural
    /// parser if available and the heuristic result is low confidence.
    pub fn parse(&self, input: &str) -> Result<ParseResult> {
        let heuristic_result = self.heuristic.parse(input)?;

        if heuristic_result.confidence >= 0.8 {
            return Ok(heuristic_result);
        }

        if let Some(ref neural) = self.neural {
            if let Ok(neural_result) = neural.parse(input) {
                if neural_result.confidence > heuristic_result.confidence {
                    return Ok(neural_result);
                }
            }
        }

        Ok(heuristic_result)
    }

    /// Parse using only the heuristic parser.
    pub fn parse_heuristic(&self, input: &str) -> Result<ParseResult> {
        self.heuristic.parse(input)
    }

    /// Parse using only the neural parser (if available).
    pub fn parse_neural(&self, input: &str) -> Result<ParseResult> {
        match &self.neural {
            Some(neural) => neural.parse(input),
            None => Err(ZantetsuError::NeuralParser(
                "Neural parser not available".into(),
            )),
        }
    }

    /// Score a parse result using the given quality profile.
    pub fn score(&self, result: &ParseResult, _profile: &QualityProfile) -> QualityScores {
        QualityScores::from_metadata(
            result.resolution,
            result.video_codec,
            result.audio_codec,
            result.source,
            result.confidence,
        )
    }

    /// Check if the neural parser is available.
    pub fn has_neural_parser(&self) -> bool {
        self.neural.is_some()
    }
}

impl Default for Zantetsu {
    fn default() -> Self {
        Self::new().expect("Failed to create Zantetsu engine")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_standard() {
        let engine = Zantetsu::new().unwrap();
        let result = engine
            .parse("[SubsPlease] Cowboy Bebop - 01 [1080p][HEVC].mkv")
            .unwrap();

        assert_eq!(result.group.as_deref(), Some("SubsPlease"));
        assert_eq!(result.title.as_deref(), Some("Cowboy Bebop"));
        assert_eq!(result.episode, Some(EpisodeSpec::Single(1)));
    }

    #[test]
    fn test_parse_empty_input() {
        let engine = Zantetsu::new().unwrap();
        let result = engine.parse("");
        assert!(result.is_err());
    }

    #[test]
    fn test_heuristic_only() {
        let engine = Zantetsu::new().unwrap();
        let result = engine.parse_heuristic("[Erai-raws] One Piece - 1071 [720p].mkv");
        assert!(result.is_ok());
    }
}
