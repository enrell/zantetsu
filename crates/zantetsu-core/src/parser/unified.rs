//! # Unified Parser Interface
//!
//! Provides a unified API for parsing anime filenames with automatic
//! mode selection and fallback handling.

use crate::error::Result;
use crate::parser::heuristic::HeuristicParser;
use crate::parser::neural::NeuralParser;
use crate::types::{ParseMode, ParseResult};

/// Configuration for the parser.
#[derive(Debug, Clone)]
pub struct ParserConfig {
    /// Which parsing mode to use
    pub mode: ParseMode,
    /// Confidence threshold for neural parser (below this, falls back to heuristic)
    pub confidence_threshold: f32,
    /// Whether to enable neural parser
    pub enable_neural: bool,
}

impl Default for ParserConfig {
    fn default() -> Self {
        Self {
            mode: ParseMode::Auto,
            confidence_threshold: 0.6,
            enable_neural: true,
        }
    }
}

impl ParserConfig {
    /// Create a new parser configuration with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the parse mode.
    pub fn with_mode(mut self, mode: ParseMode) -> Self {
        self.mode = mode;
        self
    }

    /// Set the confidence threshold for neural parser fallback.
    pub fn with_confidence_threshold(mut self, threshold: f32) -> Self {
        self.confidence_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Enable or disable neural parser.
    pub fn with_neural(mut self, enabled: bool) -> Self {
        self.enable_neural = enabled;
        self
    }
}

/// Unified parser that handles both heuristic and neural parsing with automatic fallback.
pub struct Parser {
    config: ParserConfig,
    heuristic: HeuristicParser,
    neural: Option<NeuralParser>,
}

impl Parser {
    /// Create a new parser with the given configuration.
    pub fn new(config: ParserConfig) -> Result<Self> {
        let heuristic = HeuristicParser::new()?;

        let neural = if config.enable_neural {
            match NeuralParser::new() {
                Ok(mut parser) => {
                    // Try to initialize model - if it fails, we'll fall back to heuristic
                    let _ = parser.init_model();
                    Some(parser)
                }
                Err(_) => None,
            }
        } else {
            None
        };

        Ok(Self {
            config,
            heuristic,
            neural,
        })
    }

    /// Create a new parser with default configuration.
    pub fn default() -> Result<Self> {
        Self::new(ParserConfig::default())
    }

    /// Parse a filename using the configured mode.
    ///
    /// # Arguments
    /// * `input` - The filename or torrent name to parse
    ///
    /// # Returns
    /// A `ParseResult` containing extracted metadata
    ///
    /// # Examples
    /// ```
    /// use zantetsu_core::parser::Parser;
    ///
    /// let parser = Parser::default().unwrap();
    /// let result = parser.parse("[SubsPlease] Jujutsu Kaisen - 24 (1080p) [A1B2C3D4].mkv").unwrap();
    ///
    /// assert_eq!(result.title.as_deref(), Some("Jujutsu Kaisen"));
    /// assert_eq!(result.group.as_deref(), Some("SubsPlease"));
    /// ```
    pub fn parse(&self, input: &str) -> Result<ParseResult> {
        match self.config.mode {
            ParseMode::Full => self.parse_full(input),
            ParseMode::Light => self.parse_light(input),
            ParseMode::Auto => self.parse_auto(input),
        }
    }

    /// Parse using the neural CRF model (ParseMode::Full).
    fn parse_full(&self, input: &str) -> Result<ParseResult> {
        if let Some(ref neural) = self.neural {
            neural.parse(input)
        } else {
            // Neural parser not available, fall back to heuristic
            let mut result = self.heuristic.parse(input)?;
            result.parse_mode = ParseMode::Light; // Mark as fallback
            Ok(result)
        }
    }

    /// Parse using the heuristic regex parser (ParseMode::Light).
    fn parse_light(&self, input: &str) -> Result<ParseResult> {
        self.heuristic.parse(input)
    }

    /// Parse with automatic mode selection.
    ///
    /// Strategy:
    /// 1. Try neural parser first
    /// 2. If neural parser confidence is below threshold, try heuristic
    /// 3. Return the result with higher confidence
    fn parse_auto(&self, input: &str) -> Result<ParseResult> {
        // Try neural parser first
        if let Some(ref neural) = self.neural {
            match neural.parse(input) {
                Ok(neural_result) => {
                    if neural_result.confidence >= self.config.confidence_threshold {
                        return Ok(neural_result);
                    }

                    // Neural result below threshold, try heuristic
                    match self.heuristic.parse(input) {
                        Ok(heuristic_result) => {
                            if heuristic_result.confidence > neural_result.confidence {
                                let mut result = heuristic_result;
                                result.parse_mode = ParseMode::Light;
                                return Ok(result);
                            } else {
                                return Ok(neural_result);
                            }
                        }
                        Err(_) => {
                            // Heuristic failed, return neural result anyway
                            return Ok(neural_result);
                        }
                    }
                }
                Err(_) => {
                    // Neural parser failed, fall back to heuristic
                    return self.heuristic.parse(input);
                }
            }
        }

        // No neural parser available, use heuristic
        self.heuristic.parse(input)
    }

    /// Check if the neural parser is available.
    pub fn has_neural(&self) -> bool {
        self.neural.is_some()
    }

    /// Get the parser configuration.
    pub fn config(&self) -> &ParserConfig {
        &self.config
    }
}

/// Convenience function to parse a filename with default settings.
pub fn parse(input: &str) -> Result<ParseResult> {
    let parser = Parser::default()?;
    parser.parse(input)
}

/// Parse with a specific mode.
pub fn parse_with_mode(input: &str, mode: ParseMode) -> Result<ParseResult> {
    let config = ParserConfig::new().with_mode(mode);
    let parser = Parser::new(config)?;
    parser.parse(input)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parser_creation() {
        let parser = Parser::default();
        assert!(parser.is_ok());
    }

    #[test]
    fn test_parser_config() {
        let config = ParserConfig::new()
            .with_mode(ParseMode::Light)
            .with_confidence_threshold(0.7)
            .with_neural(false);

        assert_eq!(config.mode, ParseMode::Light);
        assert_eq!(config.confidence_threshold, 0.7);
        assert!(!config.enable_neural);
    }

    #[test]
    fn test_parse_light_mode() {
        let config = ParserConfig::new().with_mode(ParseMode::Light);
        let parser = Parser::new(config).unwrap();

        let result = parser
            .parse("[SubsPlease] Jujutsu Kaisen - 24 (1080p) [A1B2C3D4].mkv")
            .unwrap();

        assert_eq!(result.group.as_deref(), Some("SubsPlease"));
        assert_eq!(result.title.as_deref(), Some("Jujutsu Kaisen"));
        assert_eq!(result.parse_mode, ParseMode::Light);
    }

    #[test]
    fn test_parse_auto_mode() {
        let config = ParserConfig::new().with_mode(ParseMode::Auto);
        let parser = Parser::new(config).unwrap();

        let result = parser
            .parse("[SubsPlease] Jujutsu Kaisen - 24 (1080p).mkv")
            .unwrap();

        // Should extract basic metadata
        assert!(result.group.is_some());
        assert!(result.resolution.is_some());
    }

    #[test]
    fn test_parse_empty() {
        let parser = Parser::default().unwrap();
        let result = parser.parse("");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_convenience_function() {
        let result = parse("[Erai-raws] Test Anime - 01 (720p).mp4");
        assert!(result.is_ok());

        let parsed = result.unwrap();
        assert_eq!(parsed.group.as_deref(), Some("Erai-raws"));
        assert_eq!(parsed.extension.as_deref(), Some("mp4"));
    }

    #[test]
    fn test_parse_with_mode() {
        let result = parse_with_mode("[Test] Anime - 01.mkv", ParseMode::Light);
        assert!(result.is_ok());
    }

    #[test]
    fn test_confidence_threshold_clamping() {
        let config = ParserConfig::new().with_confidence_threshold(1.5);
        assert_eq!(config.confidence_threshold, 1.0);

        let config = ParserConfig::new().with_confidence_threshold(-0.5);
        assert_eq!(config.confidence_threshold, 0.0);
    }
}
