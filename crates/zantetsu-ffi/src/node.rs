//! Node.js/TypeScript bindings using NAPI.
//!
//! This module provides the primary interface for using Zantetsu
//! from Node.js applications via the `zantetsu` npm package.

use napi::bindgen_prelude::*;
use napi_derive::napi;
use zantetsu_core::{
    HeuristicParser, ParseResult,
    types::{AudioCodec, EpisodeSpec, MediaSource, ParseMode, Resolution, VideoCodec},
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
/// console.log(result.episode); // '24'
/// console.log(result.resolution); // 'FHD1080'
/// ```
#[napi]
pub struct HeuristicParserNode {
    inner: HeuristicParser,
}

#[napi]
impl HeuristicParserNode {
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

    #[napi]
    pub fn parse(&self, input: String) -> Result<ParseResultNode> {
        let result = self
            .inner
            .parse(&input)
            .map_err(|e| Error::new(Status::GenericFailure, format!("parse error: {}", e)))?;
        Ok(ParseResultNode::from(result))
    }
}

#[napi]
pub struct ParseResultNode {
    pub input: String,
    pub title: Option<String>,
    pub group: Option<String>,
    pub episode: Option<String>,
    pub season: Option<u32>,
    pub resolution: Option<String>,
    pub video_codec: Option<String>,
    pub audio_codec: Option<String>,
    pub source: Option<String>,
    pub year: Option<u16>,
    pub crc32: Option<String>,
    pub extension: Option<String>,
    pub version: Option<u8>,
    pub confidence: f64,
    pub parse_mode: String,
}

impl From<ParseResult> for ParseResultNode {
    fn from(result: ParseResult) -> Self {
        Self {
            input: result.input,
            title: result.title,
            group: result.group,
            episode: result.episode.map(episode_to_string),
            season: result.season,
            resolution: result.resolution.map(resolution_to_string),
            video_codec: result.video_codec.map(vcodec_to_string),
            audio_codec: result.audio_codec.map(acodec_to_string),
            source: result.source.map(source_to_string),
            year: result.year,
            crc32: result.crc32,
            extension: result.extension,
            version: result.version,
            confidence: result.confidence as f64,
            parse_mode: parse_mode_to_string(result.parse_mode),
        }
    }
}

fn episode_to_string(spec: EpisodeSpec) -> String {
    match spec {
        EpisodeSpec::Single(ep) => format!("{}", ep),
        EpisodeSpec::Range(start, end) => format!("{}-{}", start, end),
        EpisodeSpec::Multi(eps) => eps
            .iter()
            .map(|e| e.to_string())
            .collect::<Vec<_>>()
            .join(","),
        EpisodeSpec::Version { episode, version } => format!("{}v{}", episode, version),
    }
}

fn resolution_to_string(res: Resolution) -> String {
    match res {
        Resolution::SD480 => "SD480".into(),
        Resolution::HD720 => "HD720".into(),
        Resolution::FHD1080 => "FHD1080".into(),
        Resolution::UHD2160 => "UHD2160".into(),
    }
}

fn vcodec_to_string(vc: VideoCodec) -> String {
    match vc {
        VideoCodec::H264 => "H264".into(),
        VideoCodec::HEVC => "HEVC".into(),
        VideoCodec::AV1 => "AV1".into(),
        VideoCodec::VP9 => "VP9".into(),
        VideoCodec::MPEG4 => "MPEG4".into(),
    }
}

fn acodec_to_string(ac: AudioCodec) -> String {
    match ac {
        AudioCodec::FLAC => "FLAC".into(),
        AudioCodec::AAC => "AAC".into(),
        AudioCodec::Opus => "Opus".into(),
        AudioCodec::AC3 => "AC3".into(),
        AudioCodec::DTS => "DTS".into(),
        AudioCodec::MP3 => "MP3".into(),
        AudioCodec::Vorbis => "Vorbis".into(),
        AudioCodec::TrueHD => "TrueHD".into(),
        AudioCodec::EAAC => "EAAC".into(),
    }
}

fn source_to_string(src: MediaSource) -> String {
    match src {
        MediaSource::BluRayRemux => "BluRayRemux".into(),
        MediaSource::BluRay => "BluRay".into(),
        MediaSource::WebDL => "WebDL".into(),
        MediaSource::WebRip => "WebRip".into(),
        MediaSource::HDTV => "HDTV".into(),
        MediaSource::DVD => "DVD".into(),
        MediaSource::LaserDisc => "LaserDisc".into(),
        MediaSource::VHS => "VHS".into(),
    }
}

fn parse_mode_to_string(mode: ParseMode) -> String {
    match mode {
        ParseMode::Full => "Full".into(),
        ParseMode::Light => "Light".into(),
        ParseMode::Auto => "Auto".into(),
    }
}
