//! # Tokenizer for Neural CRF Parser
//!
//! Splits anime filenames into tokens for sequence labeling.
//! Handles delimiter-based splitting and normalization.

/// A token extracted from a filename with positional information.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Token {
    /// The token text content
    pub text: String,
    /// Start position in the original string
    pub start: usize,
    /// End position in the original string
    pub end: usize,
    /// Token index in the sequence
    pub index: usize,
}

/// Delimiter characters used for tokenization.
const DELIMITERS: &[char] = &['[', ']', '(', ')', '_', '.', '-', ' '];

/// Tokenizer for anime filenames.
#[derive(Debug, Clone, Default)]
pub struct Tokenizer;

impl Tokenizer {
    /// Create a new tokenizer instance.
    pub fn new() -> Self {
        Self
    }

    /// Tokenize a filename into a sequence of tokens.
    ///
    /// # Arguments
    /// * `input` - The filename string to tokenize
    ///
    /// # Returns
    /// A vector of tokens with positional information.
    ///
    /// # Examples
    /// ```
    /// use zantetsu_core::parser::tokenizer::Tokenizer;
    ///
    /// let tokenizer = Tokenizer::new();
    /// let tokens = tokenizer.tokenize("[SubsPlease] Jujutsu Kaisen - 24 (1080p)");
    /// assert!(tokens.len() > 0);
    /// ```
    pub fn tokenize(&self, input: &str) -> Vec<Token> {
        let mut tokens = Vec::new();
        let mut current_start = 0;
        let mut token_index = 0;

        for (idx, c) in input.char_indices() {
            if DELIMITERS.contains(&c) {
                // Emit current token if non-empty
                if idx > current_start {
                    let text = input[current_start..idx].to_string();
                    if !text.is_empty() {
                        tokens.push(Token {
                            text: self.normalize(&text),
                            start: current_start,
                            end: idx,
                            index: token_index,
                        });
                        token_index += 1;
                    }
                }
                current_start = idx + c.len_utf8();
            }
        }

        // Emit final token if non-empty
        if current_start < input.len() {
            let text = input[current_start..].to_string();
            if !text.is_empty() {
                tokens.push(Token {
                    text: self.normalize(&text),
                    start: current_start,
                    end: input.len(),
                    index: token_index,
                });
            }
        }

        tokens
    }

    /// Normalize a token by:
    /// - Converting to lowercase
    /// - Stripping Unicode oddities
    /// - Trimming whitespace
    fn normalize(&self, text: &str) -> String {
        text.to_lowercase()
            .chars()
            .filter(|c| c.is_ascii_alphanumeric() || *c == ' ')
            .collect::<String>()
            .trim()
            .to_string()
    }

    /// Get the original text spans for a range of tokens.
    pub fn get_spans(
        &self,
        tokens: &[Token],
        start_idx: usize,
        end_idx: usize,
    ) -> Option<(usize, usize)> {
        if start_idx >= tokens.len() || end_idx > tokens.len() || start_idx >= end_idx {
            return None;
        }

        let start = tokens[start_idx].start;
        let end = tokens[end_idx - 1].end;
        Some((start, end))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize_basic() {
        let tokenizer = Tokenizer::new();
        let tokens = tokenizer.tokenize("[SubsPlease] Jujutsu Kaisen - 24 (1080p)");

        assert!(!tokens.is_empty());
        assert_eq!(tokens[0].text, "subsplease");
        assert_eq!(tokens[0].start, 1);
        assert_eq!(tokens[0].end, 11);
    }

    #[test]
    fn test_tokenize_with_dots() {
        let tokenizer = Tokenizer::new();
        let tokens = tokenizer.tokenize("One.Piece.1084.VOSTFR.1080p");

        assert!(tokens.len() >= 5);
        assert_eq!(tokens[0].text, "one");
        assert_eq!(tokens[1].text, "piece");
        assert_eq!(tokens[2].text, "1084");
    }

    #[test]
    fn test_tokenize_with_underscores() {
        let tokenizer = Tokenizer::new();
        let tokens = tokenizer.tokenize("[Judas]_Golden_Kamuy_S3_01");

        assert!(tokens.len() >= 5);
        let texts: Vec<_> = tokens.iter().map(|t| t.text.clone()).collect();
        assert!(texts.contains(&"judas".to_string()));
        assert!(texts.contains(&"golden".to_string()));
        assert!(texts.contains(&"kamuy".to_string()));
    }

    #[test]
    fn test_tokenize_empty() {
        let tokenizer = Tokenizer::new();
        let tokens = tokenizer.tokenize("");
        assert!(tokens.is_empty());
    }

    #[test]
    fn test_tokenize_only_delimiters() {
        let tokenizer = Tokenizer::new();
        let tokens = tokenizer.tokenize("[[[]]]()..--__");
        assert!(tokens.is_empty());
    }

    #[test]
    fn test_get_spans() {
        let tokenizer = Tokenizer::new();
        let tokens = tokenizer.tokenize("[SubsPlease] Jujutsu Kaisen - 24");

        let span = tokenizer.get_spans(&tokens, 1, 3);
        assert!(span.is_some());
    }
}
