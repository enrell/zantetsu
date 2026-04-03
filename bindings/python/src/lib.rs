use pyo3::prelude::*;
use zantetsu_core::{
    types::{AudioCodec, EpisodeSpec, MediaSource, ParseMode, Resolution, VideoCodec},
    HeuristicParser, ParseResult,
};

#[pyclass(name = "HeuristicParser")]
struct HeuristicParserPy {
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

    fn parse(&self, input: String) -> PyResult<ParseResultPy> {
        let result = self
            .inner
            .parse(&input)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(ParseResultPy::from(result))
    }
}

#[pyclass(name = "ParseResult")]
#[derive(Clone)]
struct ParseResultPy {
    #[pyo3(get)]
    input: String,
    #[pyo3(get)]
    title: Option<String>,
    #[pyo3(get)]
    group: Option<String>,
    #[pyo3(get)]
    episode: Option<String>,
    #[pyo3(get)]
    season: Option<u32>,
    #[pyo3(get)]
    resolution: Option<String>,
    #[pyo3(get)]
    video_codec: Option<String>,
    #[pyo3(get)]
    audio_codec: Option<String>,
    #[pyo3(get)]
    source: Option<String>,
    #[pyo3(get)]
    year: Option<u16>,
    #[pyo3(get)]
    crc32: Option<String>,
    #[pyo3(get)]
    extension: Option<String>,
    #[pyo3(get)]
    version: Option<u8>,
    #[pyo3(get)]
    confidence: f32,
    #[pyo3(get)]
    parse_mode: String,
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

#[pymodule]
fn _zantetsu(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<HeuristicParserPy>()?;
    m.add_class::<ParseResultPy>()?;
    Ok(())
}
