//! # BIO Tags for Named Entity Recognition
//!
//! Defines the tag set for sequence labeling of anime filename components.
//! Uses the BIO (Begin-Inside-Outside) tagging scheme.

use std::fmt;

/// BIO tags for labeling tokens in anime filenames.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BioTag {
    // Title entity
    BeginTitle,
    InsideTitle,
    // Group entity
    BeginGroup,
    InsideGroup,
    // Episode entity
    BeginEpisode,
    InsideEpisode,
    // Season entity
    BeginSeason,
    InsideSeason,
    // Single tags (no BIO variants needed)
    Resolution,
    VCodec,
    ACodec,
    Source,
    Year,
    Crc32,
    Extension,
    Version,
    // Outside (irrelevant token)
    Outside,
}

impl BioTag {
    /// Total number of distinct tags.
    pub const NUM_TAGS: usize = 17;

    /// Get all possible tags in order.
    pub fn all_tags() -> &'static [BioTag] {
        &[
            BioTag::BeginTitle,
            BioTag::InsideTitle,
            BioTag::BeginGroup,
            BioTag::InsideGroup,
            BioTag::BeginEpisode,
            BioTag::InsideEpisode,
            BioTag::BeginSeason,
            BioTag::InsideSeason,
            BioTag::Resolution,
            BioTag::VCodec,
            BioTag::ACodec,
            BioTag::Source,
            BioTag::Year,
            BioTag::Crc32,
            BioTag::Extension,
            BioTag::Version,
            BioTag::Outside,
        ]
    }

    /// Get the tag index for tensor operations.
    pub fn index(&self) -> usize {
        match self {
            BioTag::BeginTitle => 0,
            BioTag::InsideTitle => 1,
            BioTag::BeginGroup => 2,
            BioTag::InsideGroup => 3,
            BioTag::BeginEpisode => 4,
            BioTag::InsideEpisode => 5,
            BioTag::BeginSeason => 6,
            BioTag::InsideSeason => 7,
            BioTag::Resolution => 8,
            BioTag::VCodec => 9,
            BioTag::ACodec => 10,
            BioTag::Source => 11,
            BioTag::Year => 12,
            BioTag::Crc32 => 13,
            BioTag::Extension => 14,
            BioTag::Version => 15,
            BioTag::Outside => 16,
        }
    }

    /// Get tag from index.
    pub fn from_index(idx: usize) -> Option<Self> {
        match idx {
            0 => Some(BioTag::BeginTitle),
            1 => Some(BioTag::InsideTitle),
            2 => Some(BioTag::BeginGroup),
            3 => Some(BioTag::InsideGroup),
            4 => Some(BioTag::BeginEpisode),
            5 => Some(BioTag::InsideEpisode),
            6 => Some(BioTag::BeginSeason),
            7 => Some(BioTag::InsideSeason),
            8 => Some(BioTag::Resolution),
            9 => Some(BioTag::VCodec),
            10 => Some(BioTag::ACodec),
            11 => Some(BioTag::Source),
            12 => Some(BioTag::Year),
            13 => Some(BioTag::Crc32),
            14 => Some(BioTag::Extension),
            15 => Some(BioTag::Version),
            16 => Some(BioTag::Outside),
            _ => None,
        }
    }

    /// Check if this is a "Begin" tag.
    pub fn is_begin(&self) -> bool {
        matches!(
            self,
            BioTag::BeginTitle | BioTag::BeginGroup | BioTag::BeginEpisode | BioTag::BeginSeason
        )
    }

    /// Check if this is an "Inside" tag.
    pub fn is_inside(&self) -> bool {
        matches!(
            self,
            BioTag::InsideTitle
                | BioTag::InsideGroup
                | BioTag::InsideEpisode
                | BioTag::InsideSeason
        )
    }

    /// Get the entity type for this tag.
    pub fn entity_type(&self) -> Option<EntityType> {
        match self {
            BioTag::BeginTitle | BioTag::InsideTitle => Some(EntityType::Title),
            BioTag::BeginGroup | BioTag::InsideGroup => Some(EntityType::Group),
            BioTag::BeginEpisode | BioTag::InsideEpisode => Some(EntityType::Episode),
            BioTag::BeginSeason | BioTag::InsideSeason => Some(EntityType::Season),
            BioTag::Resolution => Some(EntityType::Resolution),
            BioTag::VCodec => Some(EntityType::VCodec),
            BioTag::ACodec => Some(EntityType::ACodec),
            BioTag::Source => Some(EntityType::Source),
            BioTag::Year => Some(EntityType::Year),
            BioTag::Crc32 => Some(EntityType::Crc32),
            BioTag::Extension => Some(EntityType::Extension),
            BioTag::Version => Some(EntityType::Version),
            BioTag::Outside => None,
        }
    }

    /// Check if transitioning from `from` tag to `to` tag is valid.
    pub fn is_valid_transition(from: BioTag, to: BioTag) -> bool {
        // Forbidden transitions (return false)
        match (from, to) {
            // Can't have I-* after O or different B-*
            (BioTag::InsideTitle, BioTag::BeginTitle) => return false,
            (BioTag::InsideGroup, BioTag::BeginGroup) => return false,
            (BioTag::InsideEpisode, BioTag::BeginEpisode) => return false,
            (BioTag::InsideSeason, BioTag::BeginSeason) => return false,
            // Can't transition from one entity's I-* to another entity's I-*
            (BioTag::InsideTitle, BioTag::InsideGroup) => return false,
            (BioTag::InsideTitle, BioTag::InsideEpisode) => return false,
            (BioTag::InsideTitle, BioTag::InsideSeason) => return false,
            (BioTag::InsideGroup, BioTag::InsideTitle) => return false,
            (BioTag::InsideGroup, BioTag::InsideEpisode) => return false,
            (BioTag::InsideGroup, BioTag::InsideSeason) => return false,
            (BioTag::InsideEpisode, BioTag::InsideTitle) => return false,
            (BioTag::InsideEpisode, BioTag::InsideGroup) => return false,
            (BioTag::InsideEpisode, BioTag::InsideSeason) => return false,
            (BioTag::InsideSeason, BioTag::InsideTitle) => return false,
            (BioTag::InsideSeason, BioTag::InsideGroup) => return false,
            (BioTag::InsideSeason, BioTag::InsideEpisode) => return false,
            // Can't have I-* without preceding B-* or I-* of same type
            (BioTag::Outside, BioTag::InsideTitle) => return false,
            (BioTag::Outside, BioTag::InsideGroup) => return false,
            (BioTag::Outside, BioTag::InsideEpisode) => return false,
            (BioTag::Outside, BioTag::InsideSeason) => return false,
            _ => true,
        }
    }
}

impl fmt::Display for BioTag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BioTag::BeginTitle => write!(f, "B-TITLE"),
            BioTag::InsideTitle => write!(f, "I-TITLE"),
            BioTag::BeginGroup => write!(f, "B-GROUP"),
            BioTag::InsideGroup => write!(f, "I-GROUP"),
            BioTag::BeginEpisode => write!(f, "B-EPISODE"),
            BioTag::InsideEpisode => write!(f, "I-EPISODE"),
            BioTag::BeginSeason => write!(f, "B-SEASON"),
            BioTag::InsideSeason => write!(f, "I-SEASON"),
            BioTag::Resolution => write!(f, "RESOLUTION"),
            BioTag::VCodec => write!(f, "VCODEC"),
            BioTag::ACodec => write!(f, "ACODEC"),
            BioTag::Source => write!(f, "SOURCE"),
            BioTag::Year => write!(f, "YEAR"),
            BioTag::Crc32 => write!(f, "CRC32"),
            BioTag::Extension => write!(f, "EXTENSION"),
            BioTag::Version => write!(f, "VERSION"),
            BioTag::Outside => write!(f, "O"),
        }
    }
}

/// Entity types that can be extracted from filenames.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EntityType {
    Title,
    Group,
    Episode,
    Season,
    Resolution,
    VCodec,
    ACodec,
    Source,
    Year,
    Crc32,
    Extension,
    Version,
}

/// An extracted entity with token indices.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entity {
    pub entity_type: EntityType,
    pub start_token: usize,
    pub end_token: usize,
    pub text: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tag_index_roundtrip() {
        for tag in BioTag::all_tags() {
            let idx = tag.index();
            let recovered = BioTag::from_index(idx).unwrap();
            assert_eq!(*tag, recovered);
        }
    }

    #[test]
    fn test_valid_transitions() {
        assert!(BioTag::is_valid_transition(
            BioTag::BeginTitle,
            BioTag::InsideTitle
        ));
        assert!(BioTag::is_valid_transition(
            BioTag::Outside,
            BioTag::BeginTitle
        ));
        assert!(BioTag::is_valid_transition(
            BioTag::BeginEpisode,
            BioTag::Outside
        ));
    }

    #[test]
    fn test_invalid_transitions() {
        assert!(!BioTag::is_valid_transition(
            BioTag::InsideTitle,
            BioTag::BeginTitle
        ));
        assert!(!BioTag::is_valid_transition(
            BioTag::Outside,
            BioTag::InsideTitle
        ));
        assert!(!BioTag::is_valid_transition(
            BioTag::InsideTitle,
            BioTag::InsideGroup
        ));
    }

    #[test]
    fn test_is_begin() {
        assert!(BioTag::BeginTitle.is_begin());
        assert!(BioTag::BeginGroup.is_begin());
        assert!(!BioTag::InsideTitle.is_begin());
        assert!(!BioTag::Outside.is_begin());
    }

    #[test]
    fn test_entity_type() {
        assert_eq!(BioTag::BeginTitle.entity_type(), Some(EntityType::Title));
        assert_eq!(
            BioTag::Resolution.entity_type(),
            Some(EntityType::Resolution)
        );
        assert_eq!(BioTag::Outside.entity_type(), None);
    }
}
