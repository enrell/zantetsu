use thiserror::Error;

/// Errors that can occur during Zantetsu core operations.
#[derive(Debug, Error)]
pub enum ZantetsuError {
    /// The input string is empty or contains only whitespace.
    #[error("input is empty or whitespace-only")]
    EmptyInput,

    /// The parser failed to extract any meaningful metadata.
    #[error("failed to extract metadata from input: {input:?}")]
    ParseFailed {
        /// The input that could not be parsed.
        input: String,
    },

    /// A regex pattern failed to compile (should not happen with static patterns).
    #[error("regex compilation error: {0}")]
    RegexError(#[from] regex::Error),

    /// The model weights file could not be loaded.
    #[error("failed to load model: {0}")]
    ModelLoadError(String),

    /// The model inference failed.
    #[error("inference error: {0}")]
    InferenceError(String),

    /// An invalid quality profile was provided.
    #[error("invalid scoring context: {0}")]
    InvalidContext(String),

    /// Neural parser error.
    #[error("neural parser error: {0}")]
    NeuralParser(String),

    /// Candle ML framework error.
    #[error("ML inference error: {0}")]
    CandleError(String),
}

/// Result type alias for Zantetsu operations.
pub type Result<T> = std::result::Result<T, ZantetsuError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display_messages() {
        let err = ZantetsuError::EmptyInput;
        assert_eq!(err.to_string(), "input is empty or whitespace-only");

        let err = ZantetsuError::ParseFailed {
            input: "bad input".into(),
        };
        assert!(err.to_string().contains("bad input"));
    }

    #[test]
    fn error_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ZantetsuError>();
    }
}
