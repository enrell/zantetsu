use serde::{Deserialize, Serialize};

use crate::types::VideoCodec;

use super::profile::QualityScores;

/// Device type affects resolution preference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DeviceType {
    /// Desktop computer — no penalty.
    Desktop,
    /// Laptop — slight preference for 1080p over 4K.
    Laptop,
    /// Mobile device — strong preference for 720p.
    Mobile,
    /// Television — preference for highest resolution.
    TV,
    /// Embedded device — SD/720p cap.
    Embedded,
}

/// Network quality affects bitrate tolerance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NetworkQuality {
    /// No bandwidth constraints.
    Unlimited,
    /// Broadband — slight penalty for 4K remux.
    Broadband,
    /// Limited connection — strong penalty for large files.
    Limited,
    /// Offline — only locally cached files.
    Offline,
}

/// Client context for dynamic score adjustment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientContext {
    /// Device type affects resolution preference.
    pub device_type: DeviceType,
    /// Network condition affects bitrate tolerance.
    pub network: NetworkQuality,
    /// Hardware-supported video codecs on the client.
    pub hw_decode_codecs: Vec<VideoCodec>,
}

impl Default for ClientContext {
    fn default() -> Self {
        Self {
            device_type: DeviceType::Desktop,
            network: NetworkQuality::Unlimited,
            hw_decode_codecs: vec![VideoCodec::H264, VideoCodec::HEVC],
        }
    }
}

impl ClientContext {
    /// Applies context-aware multipliers to the quality scores.
    ///
    /// Returns the adjusted final score.
    #[must_use]
    pub fn adjust_score(
        &self,
        mut scores: QualityScores,
        file_video_codec: Option<VideoCodec>,
    ) -> QualityScores {
        // Device-type resolution adjustment
        if let Some(ref mut res_score) = scores.resolution {
            let multiplier = self.resolution_multiplier(*res_score);
            *res_score *= multiplier;
        }

        // Network penalty (applied as a global modifier to all scores)
        let network_mult = self.network_multiplier();
        if let Some(ref mut res) = scores.resolution {
            *res *= network_mult;
        }
        if let Some(ref mut vc) = scores.video_codec {
            *vc *= network_mult;
        }

        // Hardware decoding penalty
        if let Some(codec) = file_video_codec {
            if !self.hw_decode_codecs.contains(&codec) {
                // Massive penalty: codec not hardware-decodable
                scores.video_codec = scores.video_codec.map(|s| s * 0.1);
            }
        }

        scores
    }

    /// Returns a resolution multiplier based on device type.
    fn resolution_multiplier(&self, res_score: f32) -> f32 {
        match self.device_type {
            DeviceType::Desktop | DeviceType::TV => 1.0,
            DeviceType::Laptop => {
                if res_score > 0.9 {
                    // 4K gets slightly penalized on laptops
                    0.85
                } else {
                    1.0
                }
            }
            DeviceType::Mobile => {
                if res_score > 0.6 {
                    // Anything above 720p gets penalized on mobile
                    0.6
                } else {
                    1.0
                }
            }
            DeviceType::Embedded => {
                if res_score > 0.5 {
                    // Embedded caps at 720p effective preference
                    0.5
                } else {
                    1.0
                }
            }
        }
    }

    /// Returns a network quality multiplier.
    fn network_multiplier(&self) -> f32 {
        match self.network {
            NetworkQuality::Unlimited => 1.0,
            NetworkQuality::Broadband => 0.9,
            NetworkQuality::Limited => 0.3,
            NetworkQuality::Offline => 1.0, // No penalty; file is already local
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{AudioCodec, MediaSource, Resolution};

    fn make_scores(
        resolution: Option<Resolution>,
        video_codec: Option<VideoCodec>,
    ) -> QualityScores {
        QualityScores::from_metadata(
            resolution,
            video_codec,
            Some(AudioCodec::AAC),
            Some(MediaSource::WebDL),
            0.7,
        )
    }

    #[test]
    fn desktop_unlimited_no_penalty() {
        let ctx = ClientContext::default();
        let scores = make_scores(Some(Resolution::UHD2160), Some(VideoCodec::H264));
        let adjusted = ctx.adjust_score(scores.clone(), Some(VideoCodec::H264));

        // Desktop + Unlimited + H264 (in hw_decode_codecs) → no penalty
        assert_eq!(adjusted.resolution, scores.resolution);
    }

    #[test]
    fn mobile_penalizes_high_resolution() {
        let ctx = ClientContext {
            device_type: DeviceType::Mobile,
            network: NetworkQuality::Unlimited,
            hw_decode_codecs: vec![VideoCodec::H264, VideoCodec::HEVC],
        };

        let scores = make_scores(Some(Resolution::FHD1080), Some(VideoCodec::H264));
        let adjusted = ctx.adjust_score(scores, Some(VideoCodec::H264));

        // 1080p score (0.85) is > 0.6 threshold → multiplied by 0.6
        let expected = 0.85 * 0.6;
        assert!(
            (adjusted.resolution.unwrap() - expected).abs() < 0.001,
            "got {}, expected {}",
            adjusted.resolution.unwrap(),
            expected
        );
    }

    #[test]
    fn limited_network_penalizes_all() {
        let ctx = ClientContext {
            device_type: DeviceType::Desktop,
            network: NetworkQuality::Limited,
            hw_decode_codecs: vec![VideoCodec::H264],
        };

        let scores = make_scores(Some(Resolution::FHD1080), Some(VideoCodec::H264));
        let adjusted = ctx.adjust_score(scores, Some(VideoCodec::H264));

        // Limited network → 0.3 multiplier on resolution and video codec
        let expected_res = 0.85 * 0.3;
        assert!((adjusted.resolution.unwrap() - expected_res).abs() < 0.001);
    }

    #[test]
    fn unsupported_codec_massive_penalty() {
        let ctx = ClientContext {
            device_type: DeviceType::Desktop,
            network: NetworkQuality::Unlimited,
            hw_decode_codecs: vec![VideoCodec::H264], // AV1 NOT listed
        };

        let scores = make_scores(Some(Resolution::FHD1080), Some(VideoCodec::AV1));
        let adjusted = ctx.adjust_score(scores, Some(VideoCodec::AV1));

        // AV1 score (1.0) * 0.1 = 0.1
        assert!((adjusted.video_codec.unwrap() - 0.1).abs() < 0.001);
    }

    #[test]
    fn default_context_is_desktop_unlimited() {
        let ctx = ClientContext::default();
        assert_eq!(ctx.device_type, DeviceType::Desktop);
        assert_eq!(ctx.network, NetworkQuality::Unlimited);
        assert!(ctx.hw_decode_codecs.contains(&VideoCodec::H264));
        assert!(ctx.hw_decode_codecs.contains(&VideoCodec::HEVC));
    }
}
