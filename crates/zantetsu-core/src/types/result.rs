use serde::{Deserialize, Serialize};

use super::episode::EpisodeSpec;
use super::quality::{AudioCodec, MediaSource, ParseMode, Resolution, VideoCodec};

/// The primary output of the Zantetsu parsing engine.
///
/// Contains all metadata extracted from an anime torrent/file name,
/// along with confidence and provenance information.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParseResult {
    /// Original input string.
    pub input: String,

    /// Extracted anime title (normalized).
    pub title: Option<String>,

    /// Release group name (e.g., "SubsPlease", "Erai-raws").
    pub group: Option<String>,

    /// Episode specification.
    pub episode: Option<EpisodeSpec>,

    /// Season number.
    pub season: Option<u32>,

    /// Video resolution.
    pub resolution: Option<Resolution>,

    /// Video codec.
    pub video_codec: Option<VideoCodec>,

    /// Audio codec.
    pub audio_codec: Option<AudioCodec>,

    /// Media source.
    pub source: Option<MediaSource>,

    /// Release year.
    pub year: Option<u16>,

    /// CRC32 checksum (hex string).
    pub crc32: Option<String>,

    /// File extension (without leading dot).
    pub extension: Option<String>,

    /// Release version (e.g., v2 = 2).
    pub version: Option<u8>,

    /// Confidence score in `[0.0, 1.0]` from the parsing engine.
    pub confidence: f32,

    /// Which parse mode produced this result.
    pub parse_mode: ParseMode,
}

impl ParseResult {
    /// Creates a new empty `ParseResult` for the given input.
    #[must_use]
    pub fn new(input: impl Into<String>, parse_mode: ParseMode) -> Self {
        Self {
            input: input.into(),
            title: None,
            group: None,
            episode: None,
            season: None,
            resolution: None,
            video_codec: None,
            audio_codec: None,
            source: None,
            year: None,
            crc32: None,
            extension: None,
            version: None,
            confidence: 0.0,
            parse_mode,
        }
    }

    /// Returns `true` if the result extracted at least a title.
    #[must_use]
    pub fn has_title(&self) -> bool {
        self.title.is_some()
    }

    /// Returns `true` if any metadata beyond the title was extracted.
    #[must_use]
    pub fn has_metadata(&self) -> bool {
        self.episode.is_some()
            || self.season.is_some()
            || self.resolution.is_some()
            || self.video_codec.is_some()
            || self.audio_codec.is_some()
            || self.source.is_some()
    }
}

impl std::fmt::Display for ParseResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ParseResult(")?;
        if let Some(ref title) = self.title {
            write!(f, "title={title:?}")?;
        }
        if let Some(ref ep) = self.episode {
            write!(f, ", ep={ep}")?;
        }
        if let Some(ref res) = self.resolution {
            write!(f, ", res={res}")?;
        }
        write!(f, ", conf={:.2}", self.confidence)?;
        write!(f, ", mode={}", self.parse_mode)?;
        write!(f, ")")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_parse_result_is_empty() {
        let result = ParseResult::new("test input", ParseMode::Light);
        assert_eq!(result.input, "test input");
        assert!(!result.has_title());
        assert!(!result.has_metadata());
        assert_eq!(result.confidence, 0.0);
        assert_eq!(result.parse_mode, ParseMode::Light);
    }

    #[test]
    fn parse_result_has_title() {
        let mut result = ParseResult::new("test", ParseMode::Light);
        assert!(!result.has_title());
        result.title = Some("Jujutsu Kaisen".into());
        assert!(result.has_title());
    }

    #[test]
    fn parse_result_has_metadata() {
        let mut result = ParseResult::new("test", ParseMode::Light);
        assert!(!result.has_metadata());
        result.resolution = Some(Resolution::FHD1080);
        assert!(result.has_metadata());
    }

    #[test]
    fn parse_result_display() {
        let mut result = ParseResult::new("test", ParseMode::Light);
        result.title = Some("Jujutsu Kaisen".into());
        result.episode = Some(EpisodeSpec::Single(24));
        result.resolution = Some(Resolution::FHD1080);
        result.confidence = 0.95;
        let display = result.to_string();
        assert!(display.contains("Jujutsu Kaisen"));
        assert!(display.contains("1080p"));
        assert!(display.contains("0.95"));
    }

    #[test]
    fn parse_result_serialization_roundtrip() {
        let mut result = ParseResult::new("test input", ParseMode::Light);
        result.title = Some("One Piece".into());
        result.group = Some("SubsPlease".into());
        result.episode = Some(EpisodeSpec::Single(1084));
        result.resolution = Some(Resolution::FHD1080);
        result.video_codec = Some(VideoCodec::H264);
        result.audio_codec = Some(AudioCodec::AAC);
        result.source = Some(MediaSource::WebDL);
        result.year = Some(2024);
        result.crc32 = Some("DEADBEEF".into());
        result.extension = Some("mkv".into());
        result.version = Some(2);
        result.confidence = 0.92;

        let json = serde_json::to_string_pretty(&result).unwrap();
        let back: ParseResult = serde_json::from_str(&json).unwrap();

        assert_eq!(result, back);
    }
}
