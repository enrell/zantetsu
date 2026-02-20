use std::fmt;

use serde::{Deserialize, Serialize};

/// Video resolution enum with quality scoring.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Resolution {
    /// 480p — Standard Definition
    SD480,
    /// 720p — High Definition
    HD720,
    /// 1080p — Full HD
    FHD1080,
    /// 2160p — Ultra HD / 4K
    UHD2160,
}

impl Resolution {
    /// Returns a normalized quality score in `[0.0, 1.0]`.
    #[must_use]
    pub fn score(self) -> f32 {
        match self {
            Self::SD480 => 0.25,
            Self::HD720 => 0.50,
            Self::FHD1080 => 0.85,
            Self::UHD2160 => 1.00,
        }
    }
}

impl fmt::Display for Resolution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SD480 => write!(f, "480p"),
            Self::HD720 => write!(f, "720p"),
            Self::FHD1080 => write!(f, "1080p"),
            Self::UHD2160 => write!(f, "2160p"),
        }
    }
}

/// Video codec enum with quality scoring.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VideoCodec {
    H264,
    HEVC,
    AV1,
    VP9,
    MPEG4,
}

impl VideoCodec {
    /// Returns a normalized quality score in `[0.0, 1.0]`.
    #[must_use]
    pub fn score(self) -> f32 {
        match self {
            Self::AV1 => 1.00,
            Self::HEVC => 0.85,
            Self::VP9 => 0.70,
            Self::H264 => 0.60,
            Self::MPEG4 => 0.20,
        }
    }
}

impl fmt::Display for VideoCodec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::H264 => write!(f, "H.264"),
            Self::HEVC => write!(f, "HEVC"),
            Self::AV1 => write!(f, "AV1"),
            Self::VP9 => write!(f, "VP9"),
            Self::MPEG4 => write!(f, "MPEG-4"),
        }
    }
}

/// Audio codec enum with quality scoring.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AudioCodec {
    FLAC,
    AAC,
    Opus,
    AC3,
    DTS,
    MP3,
    Vorbis,
    TrueHD,
    EAAC,
}

impl AudioCodec {
    /// Returns a normalized quality score in `[0.0, 1.0]`.
    #[must_use]
    pub fn score(self) -> f32 {
        match self {
            Self::TrueHD => 1.00,
            Self::FLAC => 0.95,
            Self::DTS => 0.75,
            Self::Opus => 0.70,
            Self::AAC => 0.60,
            Self::EAAC => 0.55,
            Self::AC3 => 0.50,
            Self::Vorbis => 0.45,
            Self::MP3 => 0.30,
        }
    }
}

impl fmt::Display for AudioCodec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FLAC => write!(f, "FLAC"),
            Self::AAC => write!(f, "AAC"),
            Self::Opus => write!(f, "Opus"),
            Self::AC3 => write!(f, "AC3"),
            Self::DTS => write!(f, "DTS"),
            Self::MP3 => write!(f, "MP3"),
            Self::Vorbis => write!(f, "Vorbis"),
            Self::TrueHD => write!(f, "TrueHD"),
            Self::EAAC => write!(f, "E-AAC+"),
        }
    }
}

/// Media source enum with quality scoring.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MediaSource {
    BluRayRemux,
    BluRay,
    WebDL,
    WebRip,
    HDTV,
    DVD,
    LaserDisc,
    VHS,
}

impl MediaSource {
    /// Returns a normalized quality score in `[0.0, 1.0]`.
    #[must_use]
    pub fn score(self) -> f32 {
        match self {
            Self::BluRayRemux => 1.00,
            Self::BluRay => 0.90,
            Self::WebDL => 0.75,
            Self::WebRip => 0.65,
            Self::HDTV => 0.50,
            Self::DVD => 0.40,
            Self::LaserDisc => 0.30,
            Self::VHS => 0.15,
        }
    }
}

impl fmt::Display for MediaSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BluRayRemux => write!(f, "Blu-ray Remux"),
            Self::BluRay => write!(f, "Blu-ray"),
            Self::WebDL => write!(f, "WEB-DL"),
            Self::WebRip => write!(f, "WEBRip"),
            Self::HDTV => write!(f, "HDTV"),
            Self::DVD => write!(f, "DVD"),
            Self::LaserDisc => write!(f, "LaserDisc"),
            Self::VHS => write!(f, "VHS"),
        }
    }
}

/// Parse mode selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ParseMode {
    /// Full Neural CRF inference (requires model weights).
    Full,
    /// Lightweight regex + scene rules (no ML overhead).
    Light,
    /// Auto-select based on available resources.
    Auto,
}

impl Default for ParseMode {
    fn default() -> Self {
        Self::Auto
    }
}

impl fmt::Display for ParseMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Full => write!(f, "Full (Neural CRF)"),
            Self::Light => write!(f, "Light (Heuristic)"),
            Self::Auto => write!(f, "Auto"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolution_score_ordering() {
        assert!(Resolution::UHD2160.score() > Resolution::FHD1080.score());
        assert!(Resolution::FHD1080.score() > Resolution::HD720.score());
        assert!(Resolution::HD720.score() > Resolution::SD480.score());
    }

    #[test]
    fn resolution_display() {
        assert_eq!(Resolution::FHD1080.to_string(), "1080p");
        assert_eq!(Resolution::UHD2160.to_string(), "2160p");
    }

    #[test]
    fn video_codec_score_ordering() {
        assert!(VideoCodec::AV1.score() > VideoCodec::HEVC.score());
        assert!(VideoCodec::HEVC.score() > VideoCodec::VP9.score());
        assert!(VideoCodec::VP9.score() > VideoCodec::H264.score());
        assert!(VideoCodec::H264.score() > VideoCodec::MPEG4.score());
    }

    #[test]
    fn audio_codec_score_ordering() {
        assert!(AudioCodec::TrueHD.score() > AudioCodec::FLAC.score());
        assert!(AudioCodec::FLAC.score() > AudioCodec::DTS.score());
        assert!(AudioCodec::DTS.score() > AudioCodec::Opus.score());
        assert!(AudioCodec::Opus.score() > AudioCodec::AAC.score());
        assert!(AudioCodec::AAC.score() > AudioCodec::MP3.score());
    }

    #[test]
    fn media_source_score_ordering() {
        assert!(MediaSource::BluRayRemux.score() > MediaSource::BluRay.score());
        assert!(MediaSource::BluRay.score() > MediaSource::WebDL.score());
        assert!(MediaSource::WebDL.score() > MediaSource::WebRip.score());
        assert!(MediaSource::WebRip.score() > MediaSource::HDTV.score());
    }

    #[test]
    fn parse_mode_default_is_auto() {
        assert_eq!(ParseMode::default(), ParseMode::Auto);
    }

    #[test]
    fn quality_enum_serialization_roundtrip() {
        let res = Resolution::FHD1080;
        let json = serde_json::to_string(&res).unwrap();
        let back: Resolution = serde_json::from_str(&json).unwrap();
        assert_eq!(res, back);

        let vc = VideoCodec::HEVC;
        let json = serde_json::to_string(&vc).unwrap();
        let back: VideoCodec = serde_json::from_str(&json).unwrap();
        assert_eq!(vc, back);

        let ac = AudioCodec::FLAC;
        let json = serde_json::to_string(&ac).unwrap();
        let back: AudioCodec = serde_json::from_str(&json).unwrap();
        assert_eq!(ac, back);

        let src = MediaSource::BluRay;
        let json = serde_json::to_string(&src).unwrap();
        let back: MediaSource = serde_json::from_str(&json).unwrap();
        assert_eq!(src, back);
    }
}
