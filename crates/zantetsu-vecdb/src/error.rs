use thiserror::Error;

/// Errors returned by title matching backends.
#[derive(Debug, Error)]
pub enum MatcherError {
    /// The title query is empty or whitespace-only.
    #[error("match query is empty or whitespace-only")]
    EmptyQuery,

    /// A dump path did not resolve to a readable SQL dump.
    #[error("invalid Kitsu dump path: {0}")]
    InvalidDumpPath(String),

    /// The dump did not contain the expected schema or data.
    #[error("invalid Kitsu dump format: {0}")]
    InvalidDump(String),

    /// An HTTP request failed.
    #[error("HTTP request failed: {0}")]
    Http(#[from] reqwest::Error),

    /// A filesystem operation failed.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// The remote GraphQL endpoint returned an error payload.
    #[error("remote GraphQL error: {0}")]
    GraphQl(String),

    /// The remote response was missing expected fields.
    #[error("invalid remote response: {0}")]
    InvalidResponse(String),
}

/// Result type alias for title matching.
pub type MatchResult<T> = std::result::Result<T, MatcherError>;
