//! # Viterbi Decoding for CRF
//!
//! Implements the Viterbi algorithm for finding the most likely tag sequence
//! given emission scores and transition constraints.

use crate::error::{Result, ZantetsuError};
use crate::parser::bio_tags::BioTag;

/// Viterbi decoder for CRF tag sequences.
#[derive(Debug, Clone)]
pub struct ViterbiDecoder {
    num_tags: usize,
}

/// Path score and backpointer for Viterbi decoding.
#[derive(Debug, Clone, Copy)]
struct PathState {
    score: f32,
    prev_tag: Option<usize>,
}

impl ViterbiDecoder {
    /// Create a new Viterbi decoder.
    ///
    /// # Arguments
    /// * `num_tags` - Number of distinct BIO tags
    pub fn new(num_tags: usize) -> Self {
        Self { num_tags }
    }

    /// Decode the optimal tag sequence using Viterbi algorithm.
    ///
    /// # Arguments
    /// * `emission_scores` - Matrix of shape [seq_len, num_tags] with emission scores
    /// * `transition_matrix` - Matrix of shape [num_tags, num_tags] with transition scores
    ///
    /// # Returns
    /// The optimal tag sequence as indices.
    pub fn decode(
        &self,
        emission_scores: &[Vec<f32>],
        transition_matrix: &[Vec<f32>],
    ) -> Result<Vec<usize>> {
        let seq_len = emission_scores.len();
        if seq_len == 0 {
            return Ok(Vec::new());
        }

        // Validate dimensions
        if emission_scores[0].len() != self.num_tags {
            return Err(ZantetsuError::NeuralParser(format!(
                "Emission score dimension mismatch: expected {}, got {}",
                self.num_tags,
                emission_scores[0].len()
            )));
        }

        // Initialize DP table: [seq_len][num_tags]
        let mut dp: Vec<Vec<PathState>> = vec![
            vec![
                PathState {
                    score: f32::NEG_INFINITY,
                    prev_tag: None
                };
                self.num_tags
            ];
            seq_len
        ];

        // Initialize first position
        for tag in 0..self.num_tags {
            dp[0][tag].score = emission_scores[0][tag];
        }

        // Forward pass
        for pos in 1..seq_len {
            for curr_tag in 0..self.num_tags {
                let curr_bio_tag = BioTag::from_index(curr_tag).ok_or_else(|| {
                    ZantetsuError::NeuralParser(format!("Invalid tag index: {}", curr_tag))
                })?;

                let mut best_score = f32::NEG_INFINITY;
                let mut best_prev = None;

                for prev_tag in 0..self.num_tags {
                    let prev_bio_tag = BioTag::from_index(prev_tag).ok_or_else(|| {
                        ZantetsuError::NeuralParser(format!("Invalid tag index: {}", prev_tag))
                    })?;

                    // Check if transition is valid
                    if !BioTag::is_valid_transition(prev_bio_tag, curr_bio_tag) {
                        continue;
                    }

                    let score = dp[pos - 1][prev_tag].score
                        + transition_matrix[prev_tag][curr_tag]
                        + emission_scores[pos][curr_tag];

                    if score > best_score {
                        best_score = score;
                        best_prev = Some(prev_tag);
                    }
                }

                dp[pos][curr_tag].score = best_score;
                dp[pos][curr_tag].prev_tag = best_prev;
            }
        }

        // Backtrack to find optimal path
        let mut path = Vec::with_capacity(seq_len);

        // Find best final tag
        let mut best_final_tag = 0;
        let mut best_final_score = f32::NEG_INFINITY;
        for tag in 0..self.num_tags {
            if dp[seq_len - 1][tag].score > best_final_score {
                best_final_score = dp[seq_len - 1][tag].score;
                best_final_tag = tag;
            }
        }

        // Backtrack
        path.push(best_final_tag);
        let mut curr_tag = best_final_tag;

        for pos in (1..seq_len).rev() {
            curr_tag = dp[pos][curr_tag].prev_tag.unwrap_or(0);
            path.push(curr_tag);
        }

        path.reverse();
        Ok(path)
    }

    /// Decode with hard constraints (forbidden transitions get -inf score).
    ///
    /// This is an optimized version that pre-computes valid transitions.
    pub fn decode_constrained(
        &self,
        emission_scores: &[Vec<f32>],
        transition_matrix: &[Vec<f32>],
    ) -> Result<Vec<usize>> {
        // Build constraint mask
        let mut valid_transitions: Vec<Vec<bool>> = vec![vec![false; self.num_tags]; self.num_tags];

        for prev_idx in 0..self.num_tags {
            if let Some(prev_tag) = BioTag::from_index(prev_idx) {
                for curr_idx in 0..self.num_tags {
                    if let Some(curr_tag) = BioTag::from_index(curr_idx) {
                        valid_transitions[prev_idx][curr_idx] =
                            BioTag::is_valid_transition(prev_tag, curr_tag);
                    }
                }
            }
        }

        let seq_len = emission_scores.len();
        if seq_len == 0 {
            return Ok(Vec::new());
        }

        // DP table
        let mut dp: Vec<Vec<f32>> = vec![vec![f32::NEG_INFINITY; self.num_tags]; seq_len];
        let mut backptr: Vec<Vec<Option<usize>>> = vec![vec![None; self.num_tags]; seq_len];

        // Initialize
        for tag in 0..self.num_tags {
            dp[0][tag] = emission_scores[0][tag];
        }

        // Forward pass with constraints
        for pos in 1..seq_len {
            for curr_tag in 0..self.num_tags {
                let mut best_score = f32::NEG_INFINITY;
                let mut best_prev = None;

                for prev_tag in 0..self.num_tags {
                    if !valid_transitions[prev_tag][curr_tag] {
                        continue;
                    }

                    let score = dp[pos - 1][prev_tag]
                        + transition_matrix[prev_tag][curr_tag]
                        + emission_scores[pos][curr_tag];

                    if score > best_score {
                        best_score = score;
                        best_prev = Some(prev_tag);
                    }
                }

                dp[pos][curr_tag] = best_score;
                backptr[pos][curr_tag] = best_prev;
            }
        }

        // Backtrack
        let mut best_final_tag = 0;
        let mut best_final_score = f32::NEG_INFINITY;
        for tag in 0..self.num_tags {
            if dp[seq_len - 1][tag] > best_final_score {
                best_final_score = dp[seq_len - 1][tag];
                best_final_tag = tag;
            }
        }

        let mut path = vec![best_final_tag];
        let mut curr_tag = best_final_tag;

        for pos in (1..seq_len).rev() {
            curr_tag = backptr[pos][curr_tag].unwrap_or(0);
            path.push(curr_tag);
        }

        path.reverse();
        Ok(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_simple_transition_matrix(num_tags: usize) -> Vec<Vec<f32>> {
        // Simple transition matrix with small positive scores for valid transitions
        let mut matrix = vec![vec![0.0f32; num_tags]; num_tags];

        // Set some reasonable transition scores
        for i in 0..num_tags {
            for j in 0..num_tags {
                if BioTag::is_valid_transition(
                    BioTag::from_index(i).unwrap(),
                    BioTag::from_index(j).unwrap(),
                ) {
                    matrix[i][j] = 0.1;
                } else {
                    matrix[i][j] = -1000.0; // Strong penalty for invalid
                }
            }
        }

        matrix
    }

    #[test]
    fn test_viterbi_simple() {
        let decoder = ViterbiDecoder::new(BioTag::NUM_TAGS);
        let transition = create_simple_transition_matrix(BioTag::NUM_TAGS);

        // Simple emission scores: favor tag 0 at position 0, tag 1 at position 1
        let emissions = vec![
            vec![
                1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            vec![
                0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
        ];

        let result = decoder.decode(&emissions, &transition).unwrap();
        assert_eq!(result.len(), 2);
        // With our simple scores, should pick tag 0 then tag 2 (B-Group)
        assert_eq!(result[0], 0);
    }

    #[test]
    fn test_viterbi_empty() {
        let decoder = ViterbiDecoder::new(BioTag::NUM_TAGS);
        let transition = create_simple_transition_matrix(BioTag::NUM_TAGS);
        let emissions: Vec<Vec<f32>> = vec![];

        let result = decoder.decode(&emissions, &transition).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_decode_constrained() {
        let decoder = ViterbiDecoder::new(BioTag::NUM_TAGS);
        let transition = create_simple_transition_matrix(BioTag::NUM_TAGS);

        let emissions = vec![vec![1.0; BioTag::NUM_TAGS], vec![1.0; BioTag::NUM_TAGS]];

        let result = decoder.decode_constrained(&emissions, &transition).unwrap();
        assert_eq!(result.len(), 2);
    }
}
