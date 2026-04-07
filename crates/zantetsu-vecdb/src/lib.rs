//! # Zantetsu VecDB
//!
//! Canonical title matching for parsed anime names.
//!
//! The current implementation supports two backends:
//! - A local Kitsu SQL dump (`latest.sql` or `latest.sql.gz`)
//! - A remote GraphQL endpoint compatible with the expected anime search schema
//!
//! Crates:
//! - [`zantetsu`](https://docs.rs/zantetsu) - unified API surface
//! - [`zantetsu-core`](https://docs.rs/zantetsu-core) - parsing engine
//! - [`zantetsu-vecdb`](https://docs.rs/zantetsu-vecdb) - canonical title matching
//! - [`zantetsu-trainer`](https://docs.rs/zantetsu-trainer) - training workflows
//! - [`zantetsu-ffi`](https://docs.rs/zantetsu-ffi) - Node/Python/C bindings
//!
//! ## Examples
//!
//! Use a local Kitsu dump:
//!
//! ```rust,no_run
//! use zantetsu_vecdb::{MatchSource, TitleMatcher};
//!
//! let matcher = TitleMatcher::new(
//!     MatchSource::kitsu_dump("/home/user/.local/share/zantetsu/kitsu-dumps"),
//! )
//! .unwrap();
//!
//! let best = matcher.match_title("Sousou no Frieren").unwrap();
//! assert!(best.is_some());
//! ```
//!
//! Use a remote GraphQL endpoint:
//!
//! ```rust,no_run
//! use zantetsu_vecdb::{MatchSource, TitleMatcher};
//!
//! let matcher = TitleMatcher::new(
//!     MatchSource::remote_endpoint("https://graphql.anilist.co"),
//! )
//! .unwrap();
//!
//! let best = matcher.match_title("Spy x Family").unwrap();
//! assert!(best.is_some());
//! ```

pub mod error;
mod matcher;

pub use error::{MatchResult, MatcherError};
pub use matcher::{
    AnimeIds, AnimeTitleMatch, MatchProvider, MatchSource, TitleMatcher,
    default_kitsu_dump_dir,
};
