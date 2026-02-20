//! Data loading for BIO-tagged training data.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// A single training example: sequence of (token, label) pairs.
#[derive(Debug, Clone)]
pub struct TrainingExample {
    pub tokens: Vec<String>,
    pub labels: Vec<usize>,
}

/// BIO labels
pub const BIO_LABELS: &[&str] = &["O", "B-TITLE", "I-TITLE"];

impl TrainingExample {
    pub fn new(tokens: Vec<String>, labels: Vec<usize>) -> Self {
        Self { tokens, labels }
    }
}

/// Load dataset from BIO format file.
pub fn load_bio_dataset<P: AsRef<Path>>(path: P) -> std::io::Result<Vec<TrainingExample>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut examples = Vec::new();
    let mut current_tokens = Vec::new();
    let mut current_labels = Vec::new();

    for line in reader.lines() {
        let line = line?;
        let line = line.trim();

        if line.is_empty() {
            if !current_tokens.is_empty() {
                examples.push(TrainingExample::new(
                    std::mem::take(&mut current_tokens),
                    std::mem::take(&mut current_labels),
                ));
            }
            continue;
        }

        if line.starts_with('#') {
            continue;
        }

        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() == 2 {
            let token = parts[0].to_string();
            let label_str = parts[1];

            let label_idx = match label_str {
                "O" => 0,
                "B-TITLE" => 1,
                "I-TITLE" => 2,
                _ => continue,
            };

            current_tokens.push(token);
            current_labels.push(label_idx);
        }
    }

    // Don't forget the last example
    if !current_tokens.is_empty() {
        examples.push(TrainingExample::new(current_tokens, current_labels));
    }

    Ok(examples)
}

/// Character vocabulary for encoding tokens.
pub struct CharVocab {
    char_to_idx: std::collections::HashMap<char, usize>,
}

impl CharVocab {
    pub fn new() -> Self {
        let mut char_to_idx = std::collections::HashMap::new();

        // Reserve 0 for padding/unknown
        // Add common ASCII characters
        for i in 0..128 {
            char_to_idx.insert(i as u8 as char, i + 1);
        }

        Self { char_to_idx }
    }

    pub fn encode(&self, token: &str) -> Vec<usize> {
        token
            .chars()
            .map(|c| *self.char_to_idx.get(&c).unwrap_or(&0))
            .collect()
    }

    pub fn vocab_size(&self) -> usize {
        129 // padding + ASCII
    }
}

impl Default for CharVocab {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vocab() {
        let vocab = CharVocab::new();
        let encoded = vocab.encode("Hello");
        assert!(!encoded.is_empty());
        assert_eq!(vocab.vocab_size(), 129);
    }
}
