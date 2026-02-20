use serde::{Deserialize, Serialize};

use crate::types::{AudioCodec, MediaSource, Resolution, VideoCodec};

/// Default quality profile weights.
pub const WEIGHT_RESOLUTION: f32 = 0.35;
pub const WEIGHT_VIDEO_CODEC: f32 = 0.25;
pub const WEIGHT_AUDIO_CODEC: f32 = 0.15;
pub const WEIGHT_SOURCE: f32 = 0.15;
pub const WEIGHT_GROUP_TRUST: f32 = 0.10;

/// Quality profile defining the relative importance of each dimension.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityProfile {
    pub resolution_weight: f32,
    pub video_codec_weight: f32,
    pub audio_codec_weight: f32,
    pub source_weight: f32,
    pub group_trust_weight: f32,
}

impl Default for QualityProfile {
    fn default() -> Self {
        Self {
            resolution_weight: WEIGHT_RESOLUTION,
            video_codec_weight: WEIGHT_VIDEO_CODEC,
            audio_codec_weight: WEIGHT_AUDIO_CODEC,
            source_weight: WEIGHT_SOURCE,
            group_trust_weight: WEIGHT_GROUP_TRUST,
        }
    }
}

impl QualityProfile {
    /// Validates that all weights sum to approximately 1.0.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        let sum = self.resolution_weight
            + self.video_codec_weight
            + self.audio_codec_weight
            + self.source_weight
            + self.group_trust_weight;
        (sum - 1.0).abs() < 0.01
    }
}

/// Scores for individual quality dimensions of a parsed file.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QualityScores {
    /// Resolution score `[0.0, 1.0]`.
    pub resolution: Option<f32>,
    /// Video codec score `[0.0, 1.0]`.
    pub video_codec: Option<f32>,
    /// Audio codec score `[0.0, 1.0]`.
    pub audio_codec: Option<f32>,
    /// Source score `[0.0, 1.0]`.
    pub source: Option<f32>,
    /// Group trust score `[0.0, 1.0]`.
    pub group_trust: f32,
}

impl QualityScores {
    /// Builds scores from parsed metadata.
    #[must_use]
    pub fn from_metadata(
        resolution: Option<Resolution>,
        video_codec: Option<VideoCodec>,
        audio_codec: Option<AudioCodec>,
        source: Option<MediaSource>,
        group_trust: f32,
    ) -> Self {
        Self {
            resolution: resolution.map(|r| r.score()),
            video_codec: video_codec.map(|v| v.score()),
            audio_codec: audio_codec.map(|a| a.score()),
            source: source.map(|s| s.score()),
            group_trust,
        }
    }

    /// Computes the weighted quality score using the given profile.
    /// Missing dimensions contribute 0.5 (neutral) to avoid penalizing
    /// files where metadata is simply absent.
    #[must_use]
    pub fn compute(&self, profile: &QualityProfile) -> f32 {
        let res = self.resolution.unwrap_or(0.5);
        let vc = self.video_codec.unwrap_or(0.5);
        let ac = self.audio_codec.unwrap_or(0.5);
        let src = self.source.unwrap_or(0.5);

        profile.resolution_weight * res
            + profile.video_codec_weight * vc
            + profile.audio_codec_weight * ac
            + profile.source_weight * src
            + profile.group_trust_weight * self.group_trust
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_profile_is_valid() {
        let profile = QualityProfile::default();
        assert!(profile.is_valid());
    }

    #[test]
    fn invalid_profile_detected() {
        let profile = QualityProfile {
            resolution_weight: 0.5,
            video_codec_weight: 0.5,
            audio_codec_weight: 0.5,
            source_weight: 0.5,
            group_trust_weight: 0.5,
        };
        assert!(!profile.is_valid());
    }

    #[test]
    fn quality_scores_full_metadata() {
        let scores = QualityScores::from_metadata(
            Some(Resolution::FHD1080),
            Some(VideoCodec::HEVC),
            Some(AudioCodec::FLAC),
            Some(MediaSource::BluRay),
            0.8,
        );
        let profile = QualityProfile::default();
        let score = scores.compute(&profile);

        // Expected:
        // 0.35 * 0.85 (1080p) + 0.25 * 0.85 (HEVC) + 0.15 * 0.95 (FLAC) + 0.15 * 0.90 (BluRay) + 0.10 * 0.8
        let expected = 0.35 * 0.85 + 0.25 * 0.85 + 0.15 * 0.95 + 0.15 * 0.90 + 0.10 * 0.8;
        assert!((score - expected).abs() < 0.001, "score={score}, expected={expected}");
    }

    #[test]
    fn quality_scores_missing_metadata_uses_neutral() {
        let scores = QualityScores::from_metadata(None, None, None, None, 0.5);
        let profile = QualityProfile::default();
        let score = scores.compute(&profile);
        // All dimensions use 0.5 neutral
        assert!((score - 0.5).abs() < 0.001);
    }

    #[test]
    fn quality_scores_partial_metadata() {
        let scores = QualityScores::from_metadata(
            Some(Resolution::UHD2160),
            None,
            None,
            Some(MediaSource::BluRayRemux),
            0.9,
        );
        let profile = QualityProfile::default();
        let score = scores.compute(&profile);

        let expected =
            0.35 * 1.0 + 0.25 * 0.5 + 0.15 * 0.5 + 0.15 * 1.0 + 0.10 * 0.9;
        assert!((score - expected).abs() < 0.001, "score={score}, expected={expected}");
    }
}
