use std::fmt;

use serde::{Deserialize, Serialize};

/// Episode specification supporting complex numbering schemes
/// found in anime torrent/file names.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EpisodeSpec {
    /// Single episode: "01", "12", "1084"
    Single(u32),

    /// Episode range: "01-12", "01~12"
    Range(u32, u32),

    /// Multiple discrete episodes: "01, 03, 05"
    Multi(Vec<u32>),

    /// Versioned episode: "12v2"
    Version {
        /// The episode number.
        episode: u32,
        /// The version number (e.g., v2 = 2).
        version: u8,
    },
}

impl fmt::Display for EpisodeSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Single(ep) => write!(f, "{ep:02}"),
            Self::Range(start, end) => write!(f, "{start:02}-{end:02}"),
            Self::Multi(eps) => {
                let formatted: Vec<String> = eps.iter().map(|e| format!("{e:02}")).collect();
                write!(f, "{}", formatted.join(", "))
            }
            Self::Version { episode, version } => write!(f, "{episode:02}v{version}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn episode_spec_single_display() {
        assert_eq!(EpisodeSpec::Single(1).to_string(), "01");
        assert_eq!(EpisodeSpec::Single(24).to_string(), "24");
        assert_eq!(EpisodeSpec::Single(1084).to_string(), "1084");
    }

    #[test]
    fn episode_spec_range_display() {
        assert_eq!(EpisodeSpec::Range(1, 12).to_string(), "01-12");
        assert_eq!(EpisodeSpec::Range(13, 24).to_string(), "13-24");
    }

    #[test]
    fn episode_spec_multi_display() {
        assert_eq!(EpisodeSpec::Multi(vec![1, 3, 5]).to_string(), "01, 03, 05");
    }

    #[test]
    fn episode_spec_version_display() {
        assert_eq!(
            EpisodeSpec::Version {
                episode: 12,
                version: 2
            }
            .to_string(),
            "12v2"
        );
    }

    #[test]
    fn episode_spec_serialization_roundtrip() {
        let specs = vec![
            EpisodeSpec::Single(42),
            EpisodeSpec::Range(1, 24),
            EpisodeSpec::Multi(vec![1, 5, 10]),
            EpisodeSpec::Version {
                episode: 7,
                version: 3,
            },
        ];

        for spec in &specs {
            let json = serde_json::to_string(spec).unwrap();
            let deserialized: EpisodeSpec = serde_json::from_str(&json).unwrap();
            assert_eq!(*spec, deserialized);
        }
    }
}
