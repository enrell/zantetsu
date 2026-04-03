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
    // Resolution patterns
    re_resolution: Regex,
    re_resolution_dim: Regex,

    // Codec patterns
    re_vcodec: Regex,
    re_acodec: Regex,

    // Source patterns
    re_source: Regex,

    // CRC32 patterns
    re_crc32: Regex,
    re_crc32_no_bracket: Regex,

    // Season and episode patterns
    re_season_episode: Regex,
    re_episode_range: Regex,
    re_episode_version: Regex,
    re_episode: Regex,
    re_explicit_episode: Regex,
    re_dash_episode: Regex,
    re_season: Regex,
    #[allow(dead_code)]
    re_season_long: Regex,

    // Version patterns
    re_version: Regex,

    // Year patterns
    re_year: Regex,

    // File patterns
    re_extension: Regex,
    re_group: Regex,

    // Special episode patterns
    #[allow(dead_code)]
    re_special_episode: Regex,

    // Multi-audio patterns
    #[allow(dead_code)]
    re_dual_audio: Regex,

    // Subtitle patterns
    #[allow(dead_code)]
    re_multi_sub: Regex,
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
            // Resolution patterns
            re_resolution: Regex::new(r"(?i)\b(2160|1080|720|480)[pi]\b")?,
            re_resolution_dim: Regex::new(r"(?i)(\d{3,4})\s*x\s*(\d{3,4})")?,

            // Codec patterns
            re_vcodec: Regex::new(
                r"(?i)\b(x\.?264|x\.?265|h\.?264|h\.?265|hevc|av1|vp9|mpeg4|xvid)\b",
            )?,
            re_acodec: Regex::new(
                r"(?i)\b(flac|aac|opus|ac3|dts(?:-?hd)?|truehd|true\shd|mp3|vorbis|ogg|e-?aac\+?)\b",
            )?,

            // Source patterns
            re_source: Regex::new(
                r"(?i)(?:\b|_)(blu-?ray\s*remux|bdremux|bd-?remux|blu-?ray|bdrip|web-?dl|webrip|web-?rip|web|hdtv|dvd(?:rip)?|laserdisc|ld|vhs|bd)(?:\b|_)",
            )?,

            // CRC32 patterns
            re_crc32: Regex::new(r"\[([0-9A-Fa-f]{8})\]")?,
            re_crc32_no_bracket: Regex::new(r"(?i)(?:^|[\s\-_\.\(\[])((?:[0-9a-f]{8}))")?,

            // Season and episode patterns
            re_season_episode: Regex::new(r"(?i)\bS(\d{1,2})E(\d{1,4})\b")?,
            re_episode_range: Regex::new(
                r"(?i)(?:[\s\-_\.]|(?:^|[\s\-_\.\[\(])ep?\.?\s*)(\d{1,4})\s*[-~]\s*(\d{1,4})\b",
            )?,
            re_episode_version: Regex::new(
                r"(?i)(?:[\s\-_\.]|(?:^|[\s\-_\.\[\(])ep?\.?\s*)(\d{1,4})v(\d)\b",
            )?,
            re_episode: Regex::new(
                r"(?i)(?:[\s\-_\.]|(?:^|[\s\-_\.\[\(])(?:ep?\.?|episode|session)\s*)(\d{1,4})(?:\b|[^0-9v\-~])",
            )?,
            // Explicit episode markers: .E##., EP##, Episode ##, Session ##
            re_explicit_episode: Regex::new(
                r"(?i)(?:[\s\.\-_\[\(])(?:ep?\.?|episode|session)\s*(\d{1,4})\b",
            )?,
            // Standard anime separator: " - ## " with flexible spacing
            re_dash_episode: Regex::new(r"(?:\s+-\s+)(\d{1,4})(?:\b|[^0-9v\-~])")?,

            // Season patterns
            re_season: Regex::new(r"(?i)(?:\bS|season\s*)(\d{1,2})\b")?,
            re_season_long: Regex::new(r"(?i)\bseason\s*(\d{1,2})\b")?,

            // Version patterns
            re_version: Regex::new(r"(?i)\[v(\d)\]|\bv(\d)\b")?,

            // Year patterns
            re_year: Regex::new(r"\b((?:19|20)\d{2})\b")?,

            // File patterns
            re_extension: Regex::new(r"\.(\w{2,4})$")?,
            re_group: Regex::new(r"^\[([^\]]+)\]")?,

            // Special episode patterns (OVA, ONA, Movie, etc.)
            re_special_episode: Regex::new(
                r"(?i)\b(OVA|ONA|OAD|Movie|Film|Special|SP|ED|NCOP|NCED|Preview|Trailer|Extra)\b",
            )?,

            // Multi-audio patterns
            re_dual_audio: Regex::new(
                r"(?i)\b(?:dual[\s\-_]?audio|multi[\s\-_]?audio|multi[\s\-_]?(?:lang|language))\b",
            )?,

            // Subtitle patterns
            re_multi_sub: Regex::new(
                r"(?i)\b(?:multi[\s\-_]?(?:sub|subs|subtitle)|multiple[\s\-_]?subtitle|multi)\b",
            )?,
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

        // Try CRC32 with brackets first, then without
        result.crc32 = self
            .extract_crc32(trimmed)
            .or_else(|| self.extract_crc32_no_bracket(trimmed));

        result.resolution = self.extract_resolution(trimmed);
        result.video_codec = self.extract_video_codec(trimmed);
        result.audio_codec = self.extract_audio_codec(trimmed);
        result.source = self.extract_source(trimmed);
        result.year = self.extract_year(trimmed);

        // Season and episode: try S##E## combined first
        let (se_season, se_episode) = self.extract_season_episode(trimmed);
        result.season = se_season.or_else(|| self.extract_season(trimmed));
        result.episode = se_episode.or_else(|| self.extract_episode(trimmed, &result));
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
        self.re_crc32.captures(input).map(|c| c[1].to_uppercase())
    }

    fn extract_crc32_no_bracket(&self, input: &str) -> Option<String> {
        self.re_crc32_no_bracket.captures(input).and_then(|c| {
            let crc = c.get(2)?.as_str();
            // Only return if it looks like a valid CRC32 (8 hex chars)
            if crc.len() == 8 && crc.chars().all(|ch| ch.is_ascii_hexdigit()) {
                // Make sure it's not part of a number (like 1080p or episode number)
                let prefix = &input[..c.get(1).map(|m| m.start()).unwrap_or(0)];
                if !prefix.ends_with(char::is_numeric) {
                    return Some(crc.to_uppercase());
                }
            }
            None
        })
    }

    fn extract_resolution(&self, input: &str) -> Option<Resolution> {
        // Try standard NNNNp/NNNNi format first
        if let Some(res) = self
            .re_resolution
            .captures(input)
            .and_then(|c| match &c[1] {
                "2160" => Some(Resolution::UHD2160),
                "1080" => Some(Resolution::FHD1080),
                "720" => Some(Resolution::HD720),
                "480" => Some(Resolution::SD480),
                _ => None,
            })
        {
            return Some(res);
        }

        // Try WIDTHxHEIGHT format (e.g. 1920x1080, 1280x720)
        self.re_resolution_dim.captures(input).and_then(|c| {
            let height: u32 = c[2].parse().ok()?;
            match height {
                2160 => Some(Resolution::UHD2160),
                1080 => Some(Resolution::FHD1080),
                720 => Some(Resolution::HD720),
                480 => Some(Resolution::SD480),
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
        // Normalize underscores to spaces for matching (e.g. _Blu-Ray_ patterns)
        let normalized = input.replace('_', " ");
        self.re_source.captures(&normalized).and_then(|c| {
            let source = c[1].to_lowercase().replace([' ', '-'], "");
            match source.as_str() {
                s if s.contains("remux") => Some(MediaSource::BluRayRemux),
                s if s.contains("blu") => Some(MediaSource::BluRay),
                "bdrip" => Some(MediaSource::BluRay),
                "bd" => Some(MediaSource::BluRay),
                "webdl" => Some(MediaSource::WebDL),
                "web" => Some(MediaSource::WebDL),
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
        // Try S## pattern (but not S##E## which is handled by extract_season_episode)
        self.re_season.captures(input).and_then(|c| {
            // Verify it's not part of S##E## — if so, re_season_episode handles it
            let full_match = c.get(0)?;
            let after = &input[full_match.end()..];
            // If immediately followed by E+digits, skip it (handled elsewhere)
            if after.starts_with('E') || after.starts_with('e') {
                let rest = &after[1..];
                if rest.starts_with(|ch: char| ch.is_ascii_digit()) {
                    return None;
                }
            }
            c[1].parse().ok()
        })
    }

    /// Extract combined S##E## season+episode notation.
    fn extract_season_episode(&self, input: &str) -> (Option<u32>, Option<EpisodeSpec>) {
        if let Some(caps) = self.re_season_episode.captures(input) {
            let season: u32 = caps[1].parse().ok().unwrap_or(0);
            let episode: u32 = caps[2].parse().ok().unwrap_or(0);
            return (Some(season), Some(EpisodeSpec::Single(episode)));
        }
        (None, None)
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

    fn extract_episode(&self, input: &str, result: &ParseResult) -> Option<EpisodeSpec> {
        // S##E## is handled by extract_season_episode, skip if present
        if self.re_season_episode.is_match(input) {
            return None;
        }

        // Phase 1: Versioned episodes "12v2" — try all, validate
        for caps in self.re_episode_version.captures_iter(input) {
            let episode: u32 = match caps[1].parse().ok() {
                Some(v) => v,
                None => continue,
            };
            let version: u8 = match caps[2].parse().ok() {
                Some(v) => v,
                None => continue,
            };
            if !self.is_year_or_resolution(episode, result) {
                return Some(EpisodeSpec::Version { episode, version });
            }
        }

        // Phase 2: Episode ranges "01-12" — validate the range is not "Part X-Y"
        for caps in self.re_episode_range.captures_iter(input) {
            let start: u32 = match caps[1].parse().ok() {
                Some(v) => v,
                None => continue,
            };
            let end: u32 = match caps[2].parse().ok() {
                Some(v) => v,
                None => continue,
            };
            if start >= end || self.is_resolution_number(start) {
                continue;
            }
            // Reject if preceded by "Part" or "Season" (e.g. "Part 2 - 25")
            if let Some(m) = caps.get(0) {
                let prefix = input[..m.start()].to_lowercase();
                let prefix_trimmed = prefix.trim_end();
                if prefix_trimmed.ends_with("part") || prefix_trimmed.ends_with("season") {
                    continue;
                }
            }
            return Some(EpisodeSpec::Range(start, end));
        }

        // Phase 3: Explicit episode markers (E##, Ep##, Episode ##, Session ##)
        // These are the strongest signal and override bare numbers
        if let Some(caps) = self.re_explicit_episode.captures(input) {
            let ep: u32 = caps[1].parse().ok()?;
            if !self.is_year_or_resolution(ep, result) {
                return Some(EpisodeSpec::Single(ep));
            }
        }

        // Phase 4: " - ## " separator pattern — find the LAST valid match
        // This covers the standard anime naming convention: [Group] Title - ## (quality)
        let mut last_dash_ep: Option<u32> = None;
        for caps in self.re_dash_episode.captures_iter(input) {
            let ep: u32 = match caps[1].parse().ok() {
                Some(v) => v,
                None => continue,
            };
            if self.is_year_or_resolution(ep, result) {
                continue;
            }
            // Reject Vol numbers
            if let Some(m) = caps.get(0) {
                let prefix = input[..m.start()].to_lowercase();
                let trimmed = prefix.trim_end();
                if trimmed.ends_with("vol.") || trimmed.ends_with("vol") {
                    continue;
                }
            }
            last_dash_ep = Some(ep); // Keep updating — we want the LAST one
        }
        if let Some(ep) = last_dash_ep {
            return Some(EpisodeSpec::Single(ep));
        }

        // Phase 5: Bare number fallback — only if no explicit or dash patterns matched
        // Be conservative: skip numbers that look like version parts (X.Y)
        for caps in self.re_episode.captures_iter(input) {
            let full_match = match caps.get(0) {
                Some(m) => m,
                None => continue,
            };
            let digit_match = match caps.get(1) {
                Some(m) => m,
                None => continue,
            };
            let ep: u32 = match digit_match.as_str().parse().ok() {
                Some(v) => v,
                None => continue,
            };

            if self.is_year_or_resolution(ep, result) {
                continue;
            }

            // Skip version-embedded numbers: digit.digit pattern (e.g. "2.0", "1.1")
            if full_match.start() > 0 {
                let prefix_byte = input.as_bytes()[full_match.start()];
                if prefix_byte == b'.' && full_match.start() >= 2 {
                    let before = input.as_bytes()[full_match.start() - 1];
                    if before.is_ascii_digit() {
                        continue;
                    }
                }
            }

            // Skip numbers followed by ".digit" (decimal: 2.0)
            if digit_match.end() < input.len() {
                let next_byte = input.as_bytes()[digit_match.end()];
                if next_byte == b'.'
                    && digit_match.end() + 1 < input.len()
                    && input.as_bytes()[digit_match.end() + 1].is_ascii_digit()
                {
                    continue;
                }
            }

            // Skip Vol numbers
            if full_match.start() >= 3 {
                let prefix = input[..full_match.start()].to_lowercase();
                if prefix.ends_with("vol")
                    || prefix.trim_end().ends_with("vol.")
                    || prefix.trim_end().ends_with("vol")
                {
                    continue;
                }
            }

            return Some(EpisodeSpec::Single(ep));
        }

        None
    }

    /// Check if a number is a common video resolution height.
    fn is_resolution_number(&self, n: u32) -> bool {
        matches!(n, 480 | 576 | 720 | 1080 | 2160 | 1280 | 1920 | 3840)
    }

    /// Check if a number is likely a year or resolution, not an episode.
    fn is_year_or_resolution(&self, n: u32, result: &ParseResult) -> bool {
        if let Some(year) = result.year
            && n == u32::from(year)
        {
            return true;
        }
        self.is_resolution_number(n)
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
        if result.group.is_some()
            && let Some(end) = work.find(']')
        {
            work = work[end + 1..].to_string();
        }

        // Remove file extension from the end
        if let Some(ref ext) = result.extension
            && let Some(pos) = work.rfind(&format!(".{ext}"))
        {
            work = work[..pos].to_string();
        }

        // Remove known metadata tokens (NOT episode)
        let patterns_to_strip: Vec<&Regex> = vec![
            &self.re_resolution,
            &self.re_resolution_dim,
            &self.re_vcodec,
            &self.re_acodec,
            &self.re_source,
            &self.re_crc32,
            &self.re_season_episode,
            &self.re_episode_range,
            &self.re_episode_version,
            &self.re_season,
            &self.re_version,
        ];

        for pattern in &patterns_to_strip {
            work = pattern.replace_all(&work, "\x00").to_string();
        }

        // For episode: instead of replace_all (which matches title numbers too),
        // find the correct episode position using priority-based matching
        self.sentinel_episode_in_title(&mut work, result);

        // Also strip year if it's in brackets or clearly separate
        if let Some(year) = result.year {
            let year_str = year.to_string();
            let bracketed_year = format!("({year_str})");
            work = work.replace(&bracketed_year, "\x00");
            let bracketed_year = format!("[{year_str}]");
            work = work.replace(&bracketed_year, "\x00");
        }

        // Remove any remaining bracketed content (typically metadata tags)
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

        // Strip common non-title tokens from the end
        let cleaned = strip_trailing_noise(&cleaned);

        if cleaned.is_empty() {
            None
        } else {
            Some(cleaned)
        }
    }

    /// Insert episode sentinel in the title work string at the correct position.
    /// Uses the same priority logic as extract_episode to find the RIGHT number.
    fn sentinel_episode_in_title(&self, work: &mut String, result: &ParseResult) {
        // If S##E## was the episode source, it's already stripped above
        if self.re_season_episode.is_match(work) {
            return;
        }

        // Phase 1: explicit E##/Ep## markers — sentinel these
        if self.re_explicit_episode.is_match(work) {
            *work = self
                .re_explicit_episode
                .replace_all(work, "\x00")
                .to_string();
            return;
        }

        // Phase 2: " - ## " dash separator — find the LAST valid match
        let mut last_dash_pos: Option<(usize, usize)> = None;
        for caps in self.re_dash_episode.captures_iter(work) {
            let m = match caps.get(0) {
                Some(m) => m,
                None => continue,
            };
            let digit = match caps.get(1) {
                Some(d) => d,
                None => continue,
            };
            let ep: u32 = match digit.as_str().parse().ok() {
                Some(v) => v,
                None => continue,
            };
            if self.is_year_or_resolution(ep, result) {
                continue;
            }
            last_dash_pos = Some((m.start(), m.end()));
        }
        if let Some((start, _end)) = last_dash_pos {
            work.insert(start, '\x00');
            return;
        }

        // Phase 3: Bare episode matches — use first valid one
        for caps in self.re_episode.captures_iter(work) {
            let full = match caps.get(0) {
                Some(m) => m,
                None => continue,
            };
            let digit = match caps.get(1) {
                Some(d) => d,
                None => continue,
            };
            let ep: u32 = match digit.as_str().parse().ok() {
                Some(v) => v,
                None => continue,
            };
            if self.is_year_or_resolution(ep, result) {
                continue;
            }
            // Skip version-embedded numbers (digit.digit)
            if full.start() > 0 {
                let prefix_byte = work.as_bytes()[full.start()];
                if prefix_byte == b'.' && full.start() >= 2 {
                    let before = work.as_bytes()[full.start() - 1];
                    if before.is_ascii_digit() {
                        continue;
                    }
                }
            }
            // Skip numbers followed by ".digit"
            if digit.end() < work.len() {
                let next = work.as_bytes()[digit.end()];
                if next == b'.'
                    && digit.end() + 1 < work.len()
                    && work.as_bytes()[digit.end() + 1].is_ascii_digit()
                {
                    continue;
                }
            }
            work.insert(full.start(), '\x00');
            return;
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

/// Strip common non-title tokens from the end of a title string.
fn strip_trailing_noise(title: &str) -> String {
    let noise_tokens = [
        "RAW",
        "VOSTFR",
        "MULTI",
        "Hi10P",
        "10bit",
        "Dual Audio",
        "Multiple Subtitle",
        "Multi-Subs",
        "Main 10",
    ];
    let mut result = title.to_string();
    let mut changed = true;
    while changed {
        changed = false;
        let trimmed = result.trim_end_matches(['-', ' ']);
        if trimmed.len() != result.len() {
            result = trimmed.to_string();
            changed = true;
        }
        for token in &noise_tokens {
            if result.to_lowercase().ends_with(&token.to_lowercase()) {
                result = result[..result.len() - token.len()].to_string();
                changed = true;
            }
        }
    }
    result
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
        let r = p.parse("[Group] Title (2024) - 01 (1080p).mkv").unwrap();
        assert_eq!(r.year, Some(2024));
    }

    #[test]
    fn confidence_scales_with_fields() {
        let p = parser();

        // Minimal parse — only title
        let r = p.parse("Some Random Title.mkv").unwrap();
        assert!(
            r.confidence < 0.5,
            "confidence should be low: {}",
            r.confidence
        );

        // Rich parse — many fields
        let r = p
            .parse("[SubsPlease] Jujutsu Kaisen - 24 (1080p) [H264] [AAC] [A1B2C3D4].mkv")
            .unwrap();
        assert!(
            r.confidence > 0.7,
            "confidence should be high: {}",
            r.confidence
        );
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
