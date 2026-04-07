//! # Unified Parser Interface
//!
//! Provides a unified API for parsing anime filenames with automatic
//! mode selection and fallback handling.

use crate::error::{Result, ZantetsuError};
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

fn is_usable_text(value: &Option<String>) -> bool {
    value
        .as_ref()
        .map(|v| {
            let trimmed = v.trim();
            !trimmed.is_empty() && trimmed.len() >= 2
        })
        .unwrap_or(false)
}

fn normalize_whitespace(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn normalize_candidate_text(value: &str) -> Option<String> {
    let cleaned = value
        .replace('.', " ")
        .replace('_', " ")
        .replace(['[', ']', '(', ')', '{', '}'], " ")
        .trim_matches(|c: char| {
            matches!(c, '[' | ']' | '(' | ')' | '{' | '}' | ' ' | '.' | '_' | '-')
        })
        .to_string();

    let cleaned = normalize_whitespace(&cleaned);
    (!cleaned.is_empty()).then_some(cleaned)
}

fn normalized_metadata_token(token: &str) -> String {
    token
        .trim_matches(|c: char| !c.is_ascii_alphanumeric())
        .to_ascii_lowercase()
}

fn looks_like_metadata_token(token: &str) -> bool {
    let normalized = normalized_metadata_token(token);
    match normalized.as_str() {
        "480p" | "480i" | "720p" | "720i" | "1080p" | "1080i" | "2160p" | "2160i" | "4k"
        | "bluray" | "bd" | "webdl" | "webrip" | "dvd" | "hdtv" | "remux" | "hevc" | "x264"
        | "x265" | "h264" | "h265" | "av1" | "vp9" | "aac" | "flac" | "opus" | "ac3" | "dts"
        | "mp3" | "mkv" | "mp4" | "avi" | "batch" | "complete" => true,
        _ if normalized.starts_with('v') && normalized[1..].chars().all(|c| c.is_ascii_digit()) => {
            true
        }
        _ if normalized.len() == 8 && normalized.chars().all(|c| c.is_ascii_hexdigit()) => true,
        _ => false,
    }
}

fn looks_like_metadata_noise(value: &str) -> bool {
    let parts: Vec<&str> = value.split_whitespace().collect();
    !parts.is_empty() && parts.iter().all(|part| looks_like_metadata_token(part))
}

fn clean_title_candidate(value: &str) -> Option<String> {
    let cleaned = normalize_candidate_text(value)?;
    let mut parts: Vec<&str> = cleaned.split_whitespace().collect();

    while matches!(parts.last(), Some(last) if looks_like_metadata_token(last)) {
        parts.pop();
    }

    let cleaned = normalize_whitespace(&parts.join(" "));
    if cleaned.len() < 2 || looks_like_metadata_noise(&cleaned) {
        return None;
    }

    Some(cleaned)
}

fn clean_group_candidate(value: &str) -> Option<String> {
    let cleaned = normalize_candidate_text(value)?;
    if cleaned.len() < 2 || looks_like_metadata_noise(&cleaned) {
        return None;
    }

    Some(cleaned)
}

fn strip_group_prefix(title: &str, group: &str) -> String {
    for prefix in [
        format!("{group} - "),
        format!("{group}: "),
        format!("{group} "),
    ] {
        if title
            .to_ascii_lowercase()
            .starts_with(&prefix.to_ascii_lowercase())
            && title.len() > prefix.len()
        {
            return title[prefix.len()..].trim().to_string();
        }
    }

    title.to_string()
}

fn sanitize_result(mut result: ParseResult) -> ParseResult {
    result.group = result.group.as_deref().and_then(clean_group_candidate);
    result.title = result.title.as_deref().and_then(clean_title_candidate);

    if let (Some(group), Some(title)) = (result.group.as_ref(), result.title.as_ref()) {
        let stripped = strip_group_prefix(title, group);
        if stripped != *title {
            result.title = clean_title_candidate(&stripped);
        }
    }

    if matches!((&result.group, &result.title), (Some(group), Some(title)) if group.eq_ignore_ascii_case(title))
    {
        result.group = None;
    }

    result
}

fn is_heuristic_complete(result: &ParseResult) -> bool {
    result.title.is_some() && result.group.is_some() && result.episode.is_some()
}

fn fuse_results(
    mut heuristic: ParseResult,
    neural: &ParseResult,
    neural_fill_threshold: f32,
) -> ParseResult {
    heuristic = sanitize_result(heuristic);
    let neural = sanitize_result(neural.clone());

    if neural.confidence < neural_fill_threshold {
        return heuristic;
    }

    if (!is_usable_text(&heuristic.title)
        || heuristic
            .title
            .as_ref()
            .is_some_and(|title| looks_like_metadata_noise(title)))
        && is_usable_text(&neural.title)
    {
        heuristic.title = neural.title.clone();
    }
    if !is_usable_text(&heuristic.group) && is_usable_text(&neural.group) {
        heuristic.group = neural.group.clone();
    }

    if heuristic.episode.is_none() {
        heuristic.episode = neural.episode.clone();
    }
    if heuristic.season.is_none() {
        heuristic.season = neural.season;
    }
    if heuristic.resolution.is_none() {
        heuristic.resolution = neural.resolution;
    }
    if heuristic.video_codec.is_none() {
        heuristic.video_codec = neural.video_codec;
    }
    if heuristic.audio_codec.is_none() {
        heuristic.audio_codec = neural.audio_codec;
    }
    if heuristic.source.is_none() {
        heuristic.source = neural.source;
    }
    if heuristic.year.is_none() {
        heuristic.year = neural.year;
    }
    if heuristic.crc32.is_none() {
        heuristic.crc32 = neural.crc32.clone();
    }
    if heuristic.extension.is_none() {
        heuristic.extension = neural.extension.clone();
    }
    if heuristic.version.is_none() {
        heuristic.version = neural.version;
    }

    heuristic.confidence = heuristic
        .confidence
        .max((heuristic.confidence + neural.confidence * 0.35).clamp(0.0, 1.0));

    sanitize_result(heuristic)
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
    #[allow(clippy::should_implement_trait)]
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
            neural.parse(input).map(sanitize_result)
        } else {
            // Neural parser not available, fall back to heuristic
            let mut result = sanitize_result(self.heuristic.parse(input)?);
            result.parse_mode = ParseMode::Light; // Mark as fallback
            Ok(result)
        }
    }

    /// Parse using the heuristic regex parser (ParseMode::Light).
    fn parse_light(&self, input: &str) -> Result<ParseResult> {
        self.heuristic.parse(input).map(sanitize_result)
    }

    /// Parse with automatic mode selection.
    ///
    /// Strategy:
    /// 1. Try neural parser first
    /// 2. If neural parser confidence is below threshold, try heuristic
    /// 3. Return the result with higher confidence
    fn parse_auto(&self, input: &str) -> Result<ParseResult> {
        let mut heuristic_result = sanitize_result(self.heuristic.parse(input)?);
        heuristic_result.parse_mode = ParseMode::Auto;

        let Some(neural) = self.neural.as_ref() else {
            return Ok(heuristic_result);
        };

        if heuristic_result.confidence >= self.config.confidence_threshold
            && is_heuristic_complete(&heuristic_result)
        {
            return Ok(heuristic_result);
        }

        match neural.parse(input).map(sanitize_result) {
            Ok(neural_result) => {
                if neural_result.confidence > 0.90
                    && neural_result.confidence > heuristic_result.confidence + 0.20
                    && is_usable_text(&neural_result.title)
                {
                    return Ok(neural_result);
                }

                Ok(fuse_results(
                    heuristic_result,
                    &neural_result,
                    self.config.confidence_threshold,
                ))
            }
            Err(ZantetsuError::EmptyInput) => Err(ZantetsuError::EmptyInput),
            Err(_) => Ok(heuristic_result),
        }
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
        assert_eq!(result.parse_mode, ParseMode::Auto);
    }

    #[test]
    fn test_fusion_prefers_heuristic_and_fills_missing() {
        let mut heuristic = ParseResult::new("x", ParseMode::Light);
        heuristic.group = Some("SubsPlease".into());
        heuristic.confidence = 0.62;

        let mut neural = ParseResult::new("x", ParseMode::Full);
        neural.title = Some("Jujutsu Kaisen".into());
        neural.group = Some("SP".into());
        neural.confidence = 0.82;

        let merged = fuse_results(heuristic, &neural, 0.6);

        assert_eq!(merged.group.as_deref(), Some("SubsPlease"));
        assert_eq!(merged.title.as_deref(), Some("Jujutsu Kaisen"));
        assert!(merged.confidence >= 0.62);
    }

    #[test]
    fn test_fusion_ignores_low_confidence_neural_fill() {
        let mut heuristic = ParseResult::new("x", ParseMode::Light);
        heuristic.title = Some("Stable Title".into());
        heuristic.confidence = 0.70;

        let mut neural = ParseResult::new("x", ParseMode::Full);
        neural.group = Some("NoisyGroup".into());
        neural.confidence = 0.40;

        let merged = fuse_results(heuristic.clone(), &neural, 0.6);
        assert_eq!(merged.group, heuristic.group);
        assert_eq!(merged.title, heuristic.title);
    }

    #[test]
    fn test_sanitize_result_strips_group_prefix_and_metadata_noise() {
        let mut result = ParseResult::new("x", ParseMode::Full);
        result.group = Some("SubsPlease".into());
        result.title = Some("[SubsPlease] Frieren 1080p HEVC".into());

        let cleaned = sanitize_result(result);

        assert_eq!(cleaned.group.as_deref(), Some("SubsPlease"));
        assert_eq!(cleaned.title.as_deref(), Some("Frieren"));
    }

    #[test]
    fn test_fusion_fills_additional_metadata_fields() {
        let mut heuristic = ParseResult::new("x", ParseMode::Light);
        heuristic.title = Some("Frieren".into());
        heuristic.confidence = 0.62;

        let mut neural = ParseResult::new("x", ParseMode::Full);
        neural.title = Some("Frieren".into());
        neural.source = Some(crate::types::MediaSource::WebDL);
        neural.resolution = Some(crate::types::Resolution::FHD1080);
        neural.confidence = 0.82;

        let merged = fuse_results(heuristic, &neural, 0.6);

        assert_eq!(merged.source, Some(crate::types::MediaSource::WebDL));
        assert_eq!(merged.resolution, Some(crate::types::Resolution::FHD1080));
    }

    #[test]
    fn test_is_usable_text_filters_noise() {
        assert!(!is_usable_text(&Some("".into())));
        assert!(!is_usable_text(&Some("v".into())));
        assert!(is_usable_text(&Some("ab".into())));
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
