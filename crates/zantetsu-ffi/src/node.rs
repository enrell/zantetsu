//! Node.js/TypeScript bindings using NAPI.
//!
//! This module provides the primary interface for using Zantetsu
//! from Node.js applications via the `zantetsu` npm package.

use napi::bindgen_prelude::*;
use zantetsu_core::{
    types::{
        AudioCodec, EpisodeSpec, MediaSource, ParseMode, Resolution, VideoCodec,
    },
    HeuristicParser, ParseResult,
};

/// NAPI wrapper for the HeuristicParser.
///
/// Create an instance to parse anime filenames using the fast
/// regex-based heuristic engine.
///
/// # Example
///
/// ```js
/// const { HeuristicParser } = require('zantetsu');
///
/// const parser = new HeuristicParser();
/// const result = parser.parse('[SubsPlease] Jujutsu Kaisen - 24 (1080p) [A1B2C3D4].mkv');
///
/// console.log(result.title); // 'Jujutsu Kaisen'
/// console.log(result.episode); // 24
/// console.log(result.resolution); // 'FHD1080'
/// ```
#[napi]
pub struct HeuristicParserNode {
    inner: HeuristicParser,
}

#[napi]
impl HeuristicParserNode {
    /// Creates a new HeuristicParser instance.
    ///
    /// # Errors
    ///
    /// Returns a `JsError` if regex compilation fails (should never happen
    /// with the static patterns defined internally).
    #[napi(constructor)]
    pub fn new() -> Result<Self> {
        let inner = HeuristicParser::new().map_err(|e| {
            Error::new(
                Status::GenericFailure,
                format!("failed to create parser: {}", e),
            )
        })?;
        Ok(Self { inner })
    }

    /// Parses an anime filename/torrent name using heuristic patterns.
    ///
    /// # Arguments
    ///
    /// * `input` - The filename string to parse
    ///
    /// # Errors
    ///
    /// Returns a `JsError` if the input is empty or whitespace-only.
    #[napi]
    pub fn parse(&self, input: String) -> Result<ParseResultNode> {
        let result = self.inner.parse(&input).map_err(|e| {
            Error::new(
                Status::GenericFailure,
                format!("parse error: {}", e),
            )
        })?;
        Ok(ParseResultNode::from(result))
    }
}

/// NAPI wrapper for ParseResult.
///
/// Represents the structured output of parsing an anime filename,
/// containing extracted metadata like title, episode, resolution, etc.
#[napi]
pub struct ParseResultNode {
    /// Original input string
    pub input: String,
    /// Extracted anime title (normalized)
    pub title: Option<String>,
    /// Release group name (e.g., "SubsPlease", "Erai-raws")
    pub group: Option<String>,
    /// Episode specification
    pub episode: Option<EpisodeSpecNode>,
    /// Season number
    pub season: Option<u32>,
    /// Video resolution
    pub resolution: Option<String>,
    /// Video codec
    pub video_codec: Option<String>,
    /// Audio codec
    pub audio_codec: Option<String>,
    /// Media source
    pub source: Option<String>,
    /// Release year
    pub year: Option<u16>,
    /// CRC32 checksum (hex string)
    pub crc32: Option<String>,
    /// File extension (without leading dot)
    pub extension: Option<String>,
    /// Release version (e.g., v2 = 2)
    pub version: Option<u8>,
    /// Confidence score in `[0.0, 1.0]`
    pub confidence: f32,
    /// Parse mode used
    pub parse_mode: String,
}

impl From<ParseResult> for ParseResultNode {
    fn from(result: ParseResult) -> Self {
        Self {
            input: result.input,
            title: result.title,
            group: result.group,
            episode: result.episode.map(EpisodeSpecNode::from),
            season: result.season,
            resolution: result.resolution.map(|r| resolution_to_string(r)),
            video_codec: result.video_codec.map(|v| vcodec_to_string(v)),
            audio_codec: result.audio_codec.map(|a| acodec_to_string(a)),
            source: result.source.map(|s| source_to_string(s)),
            year: result.year,
            crc32: result.crc32,
            extension: result.extension,
            version: result.version,
            confidence: result.confidence,
            parse_mode: parse_mode_to_string(result.parse_mode),
        }
    }
}

/// Episode specification supporting complex numbering schemes.
#[napi]
pub enum EpisodeSpecNode {
    /// Single episode: "01", "12", "1084"
    Single(u32),
    /// Episode range: "01-12", "01~12"
    Range(u32, u32),
    /// Multiple discrete episodes: "01, 03, 05"
    Multi(Vec<u32>),
    /// Versioned episode: "12v2"
    Versioned {
        /// The episode number
        episode: u32,
        /// The version number (e.g., v2 = 2)
        version: u8,
    },
}

impl From<EpisodeSpec> for EpisodeSpecNode {
    fn from(spec: EpisodeSpec) -> Self {
        match spec {
            EpisodeSpec::Single(ep) => Self::Single(ep),
            EpisodeSpec::Range(start, end) => Self::Range(start, end),
            EpisodeSpec::Multi(eps) => Self::Multi(eps),
            EpisodeSpec::Version { episode, version } => Self::Versioned { episode, version },
        }
    }
}

// Helper functions for converting enum variants to strings

fn resolution_to_string(res: Resolution) -> String {
    match res {
        Resolution::SD480 => "SD480".to_string(),
        Resolution::HD720 => "HD720".to_string(),
        Resolution::FHD1080 => "FHD1080".to_string(),
        Resolution::UHD2160 => "UHD2160".to_string(),
    }
}

fn vcodec_to_string(vc: VideoCodec) -> String {
    match vc {
        VideoCodec::H264 => "H264".to_string(),
        VideoCodec::HEVC => "HEVC".to_string(),
        VideoCodec::AV1 => "AV1".to_string(),
        VideoCodec::VP9 => "VP9".to_string(),
        VideoCodec::MPEG4 => "MPEG4".to_string(),
    }
}

fn acodec_to_string(ac: AudioCodec) -> String {
    match ac {
        AudioCodec::FLAC => "FLAC".to_string(),
        AudioCodec::AAC => "AAC".to_string(),
        AudioCodec::Opus => "Opus".to_string(),
        AudioCodec::AC3 => "AC3".to_string(),
        AudioCodec::DTS => "DTS".to_string(),
        AudioCodec::MP3 => "MP3".to_string(),
        AudioCodec::Vorbis => "Vorbis".to_string(),
        AudioCodec::TrueHD => "TrueHD".to_string(),
        AudioCodec::EAAC => "EAAC".to_string(),
    }
}

fn source_to_string(src: MediaSource) -> String {
    match src {
        MediaSource::BluRayRemux => "BluRayRemux".to_string(),
        MediaSource::BluRay => "BluRay".to_string(),
        MediaSource::WebDL => "WebDL".to_string(),
        MediaSource::WebRip => "WebRip".to_string(),
        MediaSource::HDTV => "HDTV".to_string(),
        MediaSource::DVD => "DVD".to_string(),
        MediaSource::LaserDisc => "LaserDisc".to_string(),
        MediaSource::VHS => "VHS".to_string(),
    }
}

fn parse_mode_to_string(mode: ParseMode) -> String {
    match mode {
        ParseMode::Full => "Full".to_string(),
        ParseMode::Light => "Light".to_string(),
        ParseMode::Auto => "Auto".to_string(),
    }
}

/// Main entry point for the zantetsu Node.js package.
///
/// Provides the HeuristicParser for fast regex-based parsing
/// of anime filenames without ML overhead.
#[napi]
pub fn get_parser_class() -> HeuristicParserNode {
    // This is a marker function - actual class is exported via macro
    HeuristicParserNode::new().expect("failed to create parser")
}
