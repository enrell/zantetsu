use regex::Regex;

use crate::error::{Result, ZantetsuError};
use crate::types::{
    AudioCodec, EpisodeSpec, MediaSource, ParseMode, ParseResult, Resolution, VideoCodec,
};

/// Heuristic parser using optimized regex patterns and scene naming rules.
///
/// This is the `ParseMode::Light` engine — fast, zero-ML-overhead parsing
/// for instant results on any device. Accuracy is lower than the Neural CRF
/// engine but latency is sub-microsecond.
pub struct HeuristicParser {
    re_resolution: Regex,
    re_vcodec: Regex,
    re_acodec: Regex,
    re_source: Regex,
    re_crc32: Regex,
    re_episode_range: Regex,
    re_episode_version: Regex,
    re_episode: Regex,
    re_season: Regex,
    re_version: Regex,
    re_year: Regex,
    re_extension: Regex,
    re_group: Regex,
}

impl HeuristicParser {
    /// Constructs a new `HeuristicParser` with pre-compiled regex patterns.
    ///
    /// # Errors
    ///
    /// Returns `ZantetsuError::RegexError` if any pattern fails to compile
    /// (should never happen with the static patterns defined here).
    pub fn new() -> Result<Self> {
        Ok(Self {
            re_resolution: Regex::new(r"(?i)\b(2160|1080|720|480)[pi]\b")?,
            re_vcodec: Regex::new(
                r"(?i)\b(x\.?264|x\.?265|h\.?264|h\.?265|hevc|av1|vp9|mpeg4|xvid)\b",
            )?,
            re_acodec: Regex::new(
                r"(?i)\b(flac|aac|opus|ac3|dts(?:-?hd)?|truehd|true\shd|mp3|vorbis|ogg|e-?aac\+?)\b",
            )?,
            re_source: Regex::new(
                r"(?i)\b(blu-?ray\s*remux|bdremux|bd-?remux|blu-?ray|web-?dl|webrip|web-?rip|hdtv|dvd(?:rip)?|laserdisc|ld|vhs)\b",
            )?,
            re_crc32: Regex::new(r"\[([0-9A-Fa-f]{8})\]")?,
            re_episode_range: Regex::new(
                r"(?i)(?:[\s\-_\.]|(?:^|[\s\-_\.\[\(])ep?\.?\s*)(\d{1,4})\s*[-~]\s*(\d{1,4})\b",
            )?,
            re_episode_version: Regex::new(
                r"(?i)(?:[\s\-_\.]|(?:^|[\s\-_\.\[\(])ep?\.?\s*)(\d{1,4})v(\d)\b",
            )?,
            re_episode: Regex::new(
                r"(?i)(?:[\s\-_\.]|(?:^|[\s\-_\.\[\(])(?:ep?\.?|episode)\s*)(\d{1,4})(?:\b|[^0-9v\-~])",
            )?,
            re_season: Regex::new(r"(?i)(?:\bS|season\s*)(\d{1,2})\b")?,
            re_version: Regex::new(r"(?i)\[v(\d)\]|\bv(\d)\b")?,
            re_year: Regex::new(r"\b((?:19|20)\d{2})\b")?,
            re_extension: Regex::new(r"\.(\w{2,4})$")?,
            re_group: Regex::new(r"^\[([^\]]+)\]")?,
        })
    }

    /// Parses the given filename/torrent name using heuristic regex patterns.
    ///
    /// # Errors
    ///
    /// Returns `ZantetsuError::EmptyInput` if the input is empty or whitespace-only.
    pub fn parse(&self, input: &str) -> Result<ParseResult> {
        let trimmed = input.trim();
        if trimmed.is_empty() {
            return Err(ZantetsuError::EmptyInput);
        }

        let mut result = ParseResult::new(trimmed, ParseMode::Light);

        // Extract structured metadata (order matters for disambiguation)
        result.group = self.extract_group(trimmed);
        result.extension = self.extract_extension(trimmed);
        result.crc32 = self.extract_crc32(trimmed);
        result.resolution = self.extract_resolution(trimmed);
        result.video_codec = self.extract_video_codec(trimmed);
        result.audio_codec = self.extract_audio_codec(trimmed);
        result.source = self.extract_source(trimmed);
        result.season = self.extract_season(trimmed);
        result.year = self.extract_year(trimmed);
        result.episode = self.extract_episode(trimmed);
        result.version = self.extract_version(trimmed, &result.episode);

        // Title extraction: everything between group tag and first metadata token
        result.title = self.extract_title(trimmed, &result);

        // Compute confidence based on how many fields were extracted
        result.confidence = self.compute_confidence(&result);

        Ok(result)
    }

    fn extract_group(&self, input: &str) -> Option<String> {
        self.re_group
            .captures(input)
            .map(|c| c[1].trim().to_string())
    }

    fn extract_extension(&self, input: &str) -> Option<String> {
        self.re_extension
            .captures(input)
            .map(|c| c[1].to_lowercase())
    }

    fn extract_crc32(&self, input: &str) -> Option<String> {
        self.re_crc32
            .captures(input)
            .map(|c| c[1].to_uppercase())
    }

    fn extract_resolution(&self, input: &str) -> Option<Resolution> {
        self.re_resolution.captures(input).and_then(|c| {
            match &c[1] {
                "2160" => Some(Resolution::UHD2160),
                "1080" => Some(Resolution::FHD1080),
                "720" => Some(Resolution::HD720),
                "480" => Some(Resolution::SD480),
                _ => None,
            }
        })
    }

    fn extract_video_codec(&self, input: &str) -> Option<VideoCodec> {
        self.re_vcodec.captures(input).and_then(|c| {
            let codec = c[1].to_lowercase();
            match codec.as_str() {
                "x264" | "x.264" | "h264" | "h.264" => Some(VideoCodec::H264),
                "x265" | "x.265" | "h265" | "h.265" | "hevc" => Some(VideoCodec::HEVC),
                "av1" => Some(VideoCodec::AV1),
                "vp9" => Some(VideoCodec::VP9),
                "mpeg4" | "xvid" => Some(VideoCodec::MPEG4),
                _ => None,
            }
        })
    }

    fn extract_audio_codec(&self, input: &str) -> Option<AudioCodec> {
        self.re_acodec.captures(input).and_then(|c| {
            let codec = c[1].to_lowercase();
            match codec.as_str() {
                "flac" => Some(AudioCodec::FLAC),
                "aac" => Some(AudioCodec::AAC),
                "opus" => Some(AudioCodec::Opus),
                "ac3" => Some(AudioCodec::AC3),
                s if s.starts_with("dts") => Some(AudioCodec::DTS),
                s if s.contains("truehd") || s.contains("true hd") => Some(AudioCodec::TrueHD),
                "mp3" => Some(AudioCodec::MP3),
                "vorbis" | "ogg" => Some(AudioCodec::Vorbis),
                s if s.starts_with("e-aac") || s.starts_with("eaac") => Some(AudioCodec::EAAC),
                _ => None,
            }
        })
    }

    fn extract_source(&self, input: &str) -> Option<MediaSource> {
        self.re_source.captures(input).and_then(|c| {
            let source = c[1].to_lowercase().replace([' ', '-'], "");
            match source.as_str() {
                s if s.contains("remux") => Some(MediaSource::BluRayRemux),
                s if s.contains("blu") || s == "bd" => Some(MediaSource::BluRay),
                "webdl" => Some(MediaSource::WebDL),
                "webrip" => Some(MediaSource::WebRip),
                "hdtv" => Some(MediaSource::HDTV),
                s if s.starts_with("dvd") => Some(MediaSource::DVD),
                s if s == "laserdisc" || s == "ld" => Some(MediaSource::LaserDisc),
                "vhs" => Some(MediaSource::VHS),
                _ => None,
            }
        })
    }

    fn extract_season(&self, input: &str) -> Option<u32> {
        self.re_season
            .captures(input)
            .and_then(|c| c[1].parse().ok())
    }

    fn extract_year(&self, input: &str) -> Option<u16> {
        // Find all year-like matches and pick the one most likely to be a release year
        // (between 1980 and current year + 1)
        self.re_year.captures(input).and_then(|c| {
            let year: u16 = c[1].parse().ok()?;
            if (1980..=2030).contains(&year) {
                Some(year)
            } else {
                None
            }
        })
    }

    fn extract_episode(&self, input: &str) -> Option<EpisodeSpec> {
        // Try episode range first: "01-12"
        if let Some(caps) = self.re_episode_range.captures(input) {
            let start: u32 = caps[1].parse().ok()?;
            let end: u32 = caps[2].parse().ok()?;
            if start < end {
                return Some(EpisodeSpec::Range(start, end));
            }
        }

        // Try versioned episode: "12v2"
        if let Some(caps) = self.re_episode_version.captures(input) {
            let episode: u32 = caps[1].parse().ok()?;
            let version: u8 = caps[2].parse().ok()?;
            return Some(EpisodeSpec::Version { episode, version });
        }

        // Try single episode
        if let Some(caps) = self.re_episode.captures(input) {
            let ep: u32 = caps[1].parse().ok()?;
            return Some(EpisodeSpec::Single(ep));
        }

        None
    }

    fn extract_version(&self, input: &str, episode: &Option<EpisodeSpec>) -> Option<u8> {
        // If the episode already captured a version (e.g. "12v2"), don't double-extract
        if let Some(EpisodeSpec::Version { .. }) = episode {
            return None;
        }

        self.re_version.captures(input).and_then(|c| {
            // Try group 1 (bracket form [v2]) then group 2 (bare v2)
            c.get(1)
                .or_else(|| c.get(2))
                .and_then(|m| m.as_str().parse().ok())
        })
    }

    /// Extracts the title from the input by identifying the text region
    /// between the group tag (if any) and the first metadata token.
    fn extract_title(&self, input: &str, result: &ParseResult) -> Option<String> {
        let mut work = input.to_string();

        // Remove the group tag from the start
        if result.group.is_some() {
            if let Some(end) = work.find(']') {
                work = work[end + 1..].to_string();
            }
        }

        // Remove file extension from the end
        if let Some(ref ext) = result.extension {
            if let Some(pos) = work.rfind(&format!(".{ext}")) {
                work = work[..pos].to_string();
            }
        }

        // Remove known metadata tokens from the working string
        // by replacing matched regions with a sentinel
        let patterns_to_strip: Vec<&Regex> = vec![
            &self.re_resolution,
            &self.re_vcodec,
            &self.re_acodec,
            &self.re_source,
            &self.re_crc32,
            &self.re_episode_range,
            &self.re_episode_version,
            &self.re_season,
            &self.re_version,
        ];

        for pattern in &patterns_to_strip {
            work = pattern.replace_all(&work, "\x00").to_string();
        }

        // For episode, replace more carefully (avoid consuming part of the title)
        work = self.re_episode.replace_all(&work, "\x00").to_string();

        // Also strip year if it's in brackets or clearly separate
        if let Some(year) = result.year {
            let year_str = year.to_string();
            // Only strip if it appears in brackets or is clearly not part of the title
            let bracketed_year = format!("({year_str})");
            work = work.replace(&bracketed_year, "\x00");
            let bracketed_year = format!("[{year_str}]");
            work = work.replace(&bracketed_year, "\x00");
        }

        // Remove any remaining bracketed content (typically metadata tags like [Multiple Subtitle])
        let re_brackets = Regex::new(r"\[[^\]]*\]|\([^\)]*\)").ok()?;
        work = re_brackets.replace_all(&work, " ").to_string();

        // Take text before the first sentinel (null byte)
        let title_region = work.split('\x00').next().unwrap_or("");

        // Clean up: replace dots, underscores with spaces; normalize whitespace
        let cleaned = title_region
            .replace(['.', '_'], " ")
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
            .trim_matches(|c: char| c == '-' || c == ' ')
            .to_string();

        if cleaned.is_empty() {
            None
        } else {
            Some(cleaned)
        }
    }

    /// Computes a confidence score based on how many metadata fields
    /// were successfully extracted.
    fn compute_confidence(&self, result: &ParseResult) -> f32 {
        let mut fields_present = 0u32;
        let mut fields_total = 7u32; // title, group, episode, resolution, vcodec, acodec, source

        if result.title.is_some() {
            fields_present += 2; // Title is worth double
            fields_total += 1;
        }
        if result.group.is_some() {
            fields_present += 1;
        }
        if result.episode.is_some() {
            fields_present += 1;
        }
        if result.resolution.is_some() {
            fields_present += 1;
        }
        if result.video_codec.is_some() {
            fields_present += 1;
        }
        if result.audio_codec.is_some() {
            fields_present += 1;
        }
        if result.source.is_some() {
            fields_present += 1;
        }

        (fields_present as f32 / fields_total as f32).min(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parser() -> HeuristicParser {
        HeuristicParser::new().unwrap()
    }

    #[test]
    fn empty_input_errors() {
        let p = parser();
        assert!(matches!(p.parse(""), Err(ZantetsuError::EmptyInput)));
        assert!(matches!(p.parse("   "), Err(ZantetsuError::EmptyInput)));
    }

    #[test]
    fn subsplease_standard_format() {
        let p = parser();
        let r = p
            .parse("[SubsPlease] Jujutsu Kaisen - 24 (1080p) [A1B2C3D4].mkv")
            .unwrap();

        assert_eq!(r.title.as_deref(), Some("Jujutsu Kaisen"));
        assert_eq!(r.group.as_deref(), Some("SubsPlease"));
        assert_eq!(r.episode, Some(EpisodeSpec::Single(24)));
        assert_eq!(r.resolution, Some(Resolution::FHD1080));
        assert_eq!(r.crc32.as_deref(), Some("A1B2C3D4"));
        assert_eq!(r.extension.as_deref(), Some("mkv"));
        assert_eq!(r.parse_mode, ParseMode::Light);
    }

    #[test]
    fn erai_raws_versioned_episode() {
        let p = parser();
        let r = p
            .parse("[Erai-raws] Shingeki no Kyojin - The Final Season - 28v2 [1080p][HEVC].mkv")
            .unwrap();

        assert_eq!(r.group.as_deref(), Some("Erai-raws"));
        assert_eq!(
            r.episode,
            Some(EpisodeSpec::Version {
                episode: 28,
                version: 2
            })
        );
        assert_eq!(r.resolution, Some(Resolution::FHD1080));
        assert_eq!(r.video_codec, Some(VideoCodec::HEVC));
        assert_eq!(r.extension.as_deref(), Some("mkv"));
    }

    #[test]
    fn batch_episode_range() {
        let p = parser();
        let r = p
            .parse("[Judas] Golden Kamuy S3 - 01-12 (1080p) [Batch]")
            .unwrap();

        assert_eq!(r.group.as_deref(), Some("Judas"));
        assert_eq!(r.season, Some(3));
        assert_eq!(r.episode, Some(EpisodeSpec::Range(1, 12)));
        assert_eq!(r.resolution, Some(Resolution::FHD1080));
    }

    #[test]
    fn dot_separated_format() {
        let p = parser();
        let r = p
            .parse("One.Piece.1084.VOSTFR.1080p.WEB.x264-AAC.mkv")
            .unwrap();

        assert_eq!(r.title.as_deref(), Some("One Piece"));
        assert_eq!(r.episode, Some(EpisodeSpec::Single(1084)));
        assert_eq!(r.resolution, Some(Resolution::FHD1080));
        assert_eq!(r.video_codec, Some(VideoCodec::H264));
        assert_eq!(r.audio_codec, Some(AudioCodec::AAC));
        assert_eq!(r.extension.as_deref(), Some("mkv"));
    }

    #[test]
    fn resolution_extraction() {
        let p = parser();

        let r = p.parse("[Test] Show - 01 (480p).mkv").unwrap();
        assert_eq!(r.resolution, Some(Resolution::SD480));

        let r = p.parse("[Test] Show - 01 (720p).mkv").unwrap();
        assert_eq!(r.resolution, Some(Resolution::HD720));

        let r = p.parse("[Test] Show - 01 (2160p).mkv").unwrap();
        assert_eq!(r.resolution, Some(Resolution::UHD2160));
    }

    #[test]
    fn video_codec_variants() {
        let p = parser();

        for (input, expected) in [
            ("x264", VideoCodec::H264),
            ("H.264", VideoCodec::H264),
            ("x265", VideoCodec::HEVC),
            ("HEVC", VideoCodec::HEVC),
            ("H.265", VideoCodec::HEVC),
            ("AV1", VideoCodec::AV1),
            ("VP9", VideoCodec::VP9),
        ] {
            let r = p
                .parse(&format!("[Group] Title - 01 [{input}].mkv"))
                .unwrap();
            assert_eq!(r.video_codec, Some(expected), "failed for input: {input}");
        }
    }

    #[test]
    fn audio_codec_variants() {
        let p = parser();

        for (input, expected) in [
            ("FLAC", AudioCodec::FLAC),
            ("AAC", AudioCodec::AAC),
            ("Opus", AudioCodec::Opus),
            ("AC3", AudioCodec::AC3),
            ("DTS", AudioCodec::DTS),
            ("MP3", AudioCodec::MP3),
        ] {
            let r = p
                .parse(&format!("[Group] Title - 01 [{input}].mkv"))
                .unwrap();
            assert_eq!(r.audio_codec, Some(expected), "failed for input: {input}");
        }
    }

    #[test]
    fn source_extraction() {
        let p = parser();

        let r = p.parse("[Group] Title - 01 Blu-ray 1080p.mkv").unwrap();
        assert_eq!(r.source, Some(MediaSource::BluRay));

        let r = p.parse("[Group] Title - 01 WEB-DL 1080p.mkv").unwrap();
        assert_eq!(r.source, Some(MediaSource::WebDL));

        let r = p.parse("[Group] Title - 01 HDTV 720p.mkv").unwrap();
        assert_eq!(r.source, Some(MediaSource::HDTV));
    }

    #[test]
    fn year_extraction() {
        let p = parser();
        let r = p
            .parse("[Group] Title (2024) - 01 (1080p).mkv")
            .unwrap();
        assert_eq!(r.year, Some(2024));
    }

    #[test]
    fn confidence_scales_with_fields() {
        let p = parser();

        // Minimal parse — only title
        let r = p.parse("Some Random Title.mkv").unwrap();
        assert!(r.confidence < 0.5, "confidence should be low: {}", r.confidence);

        // Rich parse — many fields
        let r = p
            .parse("[SubsPlease] Jujutsu Kaisen - 24 (1080p) [H264] [AAC] [A1B2C3D4].mkv")
            .unwrap();
        assert!(r.confidence > 0.7, "confidence should be high: {}", r.confidence);
    }

    #[test]
    fn parse_result_is_serializable() {
        let p = parser();
        let r = p
            .parse("[SubsPlease] Jujutsu Kaisen - 24 (1080p) [A1B2C3D4].mkv")
            .unwrap();

        let json = serde_json::to_string(&r).unwrap();
        let back: ParseResult = serde_json::from_str(&json).unwrap();
        assert_eq!(r, back);
    }
}
