//! # Zantetsu VecDB
//!
//! Canonical title matching for parsed anime names.
//!
//! The current implementation supports two backends:
//! - A local Kitsu SQL dump (`latest.sql` or `latest.sql.gz`)
//! - A remote GraphQL endpoint compatible with the expected anime search schema

pub mod error;
mod matcher;

pub use error::{MatchResult, MatcherError};
pub use matcher::{
    AnimeIds, AnimeTitleMatch, MatchProvider, MatchSource, TitleMatcher,
    default_kitsu_dump_dir,
};
