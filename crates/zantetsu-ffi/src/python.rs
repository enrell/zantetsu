//! Python bindings using PyO3.
//!
//! This module provides the primary interface for using Zantetsu
//! from Python applications via the `zantetsu` PyPI package.

use pyo3::prelude::*;
use zantetsu_core::{
    HeuristicParser, ParseResult,
    types::{AudioCodec, EpisodeSpec, MediaSource, ParseMode, Resolution, VideoCodec},
};

/// PyO3 wrapper for the HeuristicParser.
///
/// Create an instance to parse anime filenames using the fast
/// regex-based heuristic engine.
///
/// # Example
///
/// ```python
/// from zantetsu import HeuristicParser
///
/// parser = HeuristicParser()
/// result = parser.parse('[SubsPlease] Jujutsu Kaisen - 24 (1080p) [A1B2C3D4].mkv')
///
/// print(result.title)      # 'Jujutsu Kaisen'
/// print(result.episode)    # 24
/// print(result.resolution) # 'FHD1080'
/// ```
#[pyclass(name = "HeuristicParser")]
pub struct HeuristicParserPy {
    inner: HeuristicParser,
}

#[pymethods]
impl HeuristicParserPy {
    #[new]
    fn new() -> PyResult<Self> {
        let inner = HeuristicParser::new()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Parses an anime filename/torrent name using heuristic patterns.
    ///
    /// # Arguments
    ///
    /// * `input` - The filename string to parse
    ///
    /// # Returns
    ///
    /// A `ParseResult` object containing extracted metadata.
    fn parse(&self, input: String) -> PyResult<ParseResultPy> {
        let result = self
            .inner
            .parse(&input)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(ParseResultPy::from(result))
    }
}

/// PyO3 wrapper for ParseResult.
///
/// Represents the structured output of parsing an anime filename,
/// containing extracted metadata like title, episode, resolution, etc.
#[pyclass(name = "ParseResult")]
#[derive(Clone)]
pub struct ParseResultPy {
    #[pyo3(get)]
    pub input: String,
    #[pyo3(get)]
    pub title: Option<String>,
    #[pyo3(get)]
    pub group: Option<String>,
    #[pyo3(get)]
    pub episode: Option<String>,
    #[pyo3(get)]
    pub season: Option<u32>,
    #[pyo3(get)]
    pub resolution: Option<String>,
    #[pyo3(get)]
    pub video_codec: Option<String>,
    #[pyo3(get)]
    pub audio_codec: Option<String>,
    #[pyo3(get)]
    pub source: Option<String>,
    #[pyo3(get)]
    pub year: Option<u16>,
    #[pyo3(get)]
    pub crc32: Option<String>,
    #[pyo3(get)]
    pub extension: Option<String>,
    #[pyo3(get)]
    pub version: Option<u8>,
    #[pyo3(get)]
    pub confidence: f32,
    #[pyo3(get)]
    pub parse_mode: String,
}

impl From<ParseResult> for ParseResultPy {
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
            confidence: result.confidence,
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
        Resolution::SD480 => "SD480",
        Resolution::HD720 => "HD720",
        Resolution::FHD1080 => "FHD1080",
        Resolution::UHD2160 => "UHD2160",
    }
    .into()
}

fn vcodec_to_string(vc: VideoCodec) -> String {
    match vc {
        VideoCodec::H264 => "H264",
        VideoCodec::HEVC => "HEVC",
        VideoCodec::AV1 => "AV1",
        VideoCodec::VP9 => "VP9",
        VideoCodec::MPEG4 => "MPEG4",
    }
    .into()
}

fn acodec_to_string(ac: AudioCodec) -> String {
    match ac {
        AudioCodec::FLAC => "FLAC",
        AudioCodec::AAC => "AAC",
        AudioCodec::Opus => "Opus",
        AudioCodec::AC3 => "AC3",
        AudioCodec::DTS => "DTS",
        AudioCodec::MP3 => "MP3",
        AudioCodec::Vorbis => "Vorbis",
        AudioCodec::TrueHD => "TrueHD",
        AudioCodec::EAAC => "EAAC",
    }
    .into()
}

fn source_to_string(src: MediaSource) -> String {
    match src {
        MediaSource::BluRayRemux => "BluRayRemux",
        MediaSource::BluRay => "BluRay",
        MediaSource::WebDL => "WebDL",
        MediaSource::WebRip => "WebRip",
        MediaSource::HDTV => "HDTV",
        MediaSource::DVD => "DVD",
        MediaSource::LaserDisc => "LaserDisc",
        MediaSource::VHS => "VHS",
    }
    .into()
}

fn parse_mode_to_string(mode: ParseMode) -> String {
    match mode {
        ParseMode::Full => "Full",
        ParseMode::Light => "Light",
        ParseMode::Auto => "Auto",
    }
    .into()
}

/// Python module definition.
pub fn pymodule(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<HeuristicParserPy>()?;
    m.add_class::<ParseResultPy>()?;
    Ok(())
}
