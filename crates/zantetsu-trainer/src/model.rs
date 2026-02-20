//! CRF Model for sequence labeling.
//! Improved feature-based CRF with better tokenization.

pub const NUM_LABELS: usize = 3;

#[derive(Clone)]
pub struct CrfModel {
    pub transition: Vec<f32>,
    pub emission_weights: Vec<f32>, // Per-label bias
}

impl CrfModel {
    pub fn new() -> Self {
        let transition = vec![0.0f32; NUM_LABELS * NUM_LABELS];
        let emission_weights = vec![0.0f32; NUM_LABELS];

        Self {
            transition,
            emission_weights,
        }
    }

    fn extract_features(
        &self,
        token: &str,
        prev_token: Option<&str>,
        next_token: Option<&str>,
    ) -> Vec<f32> {
        let mut features = Vec::new();

        let lower = token.to_lowercase();
        let len = token.len();

        // Basic features
        features.push(
            if token
                .chars()
                .all(|c| !c.is_alphabetic() || c.is_uppercase())
            {
                1.0
            } else {
                0.0
            },
        ); // is_all_caps
        features.push(if token.starts_with('[') || token.starts_with('(') {
            1.0
        } else {
            0.0
        }); // has_bracket_start
        features.push(if token.ends_with(']') || token.ends_with(')') {
            1.0
        } else {
            0.0
        }); // has_bracket_end
        features.push(
            if lower.contains("e0")
                || lower.contains("s0")
                || lower.chars().all(|c| c.is_ascii_digit())
            {
                1.0
            } else {
                0.0
            },
        ); // is_episode
        features.push(
            if lower.contains("720p")
                || lower.contains("1080p")
                || lower.contains("480p")
                || lower == "bd"
                || lower == "web"
            {
                1.0
            } else {
                0.0
            },
        ); // is_quality
        features.push(if token.chars().any(|c| c.is_ascii_digit()) {
            1.0
        } else {
            0.0
        }); // has_digit
        features.push(if token.len() > 3 { 1.0 } else { 0.0 }); // long_token

        // Context features
        features.push(if let Some(p) = prev_token {
            if p.starts_with('[') || p.starts_with('(') {
                1.0
            } else {
                0.0
            }
        } else {
            0.0
        });
        features.push(if let Some(n) = next_token {
            if n.starts_with('[') || n.starts_with('(') {
                1.0
            } else {
                0.0
            }
        } else {
            0.0
        });

        features
    }

    fn compute_emission(
        &self,
        token: &str,
        prev_token: Option<&str>,
        next_token: Option<&str>,
        label: usize,
    ) -> f32 {
        let features = self.extract_features(token, prev_token, next_token);

        let bias = self.emission_weights[label];

        // Score based on features and label
        let mut score = bias;

        match label {
            0 => {
                // O
                score += features[2] * 2.0; // has brackets -> O
                score += features[3] * 2.0; // episode -> O
                score += features[4] * 2.0; // quality -> O
                score -= features[0] * 1.0; // all_caps -> not O
            }
            1 => {
                // B-TITLE
                score += features[0] * 2.0; // all_caps -> B-TITLE
                score -= features[2] * 2.0; // has brackets -> not B-TITLE
                score -= features[3] * 2.0; // episode -> not B-TITLE
                score -= features[4] * 2.0; // quality -> not B-TITLE
                score += features[5] * 0.5; // has digit (part of title)
            }
            2 => {
                // I-TITLE
                score += features[0] * 1.5; // all_caps -> I-TITLE
                score -= features[2] * 2.0; // has brackets -> not I-TITLE
                score -= features[3] * 2.0; // episode -> not I-TITLE
                score -= features[4] * 2.0; // quality -> not I-TITLE
            }
            _ => {}
        }

        score
    }

    pub fn forward(&self, tokens: &[String]) -> (Vec<Vec<f32>>, Vec<f32>) {
        let seq_len = tokens.len();
        let mut emissions = Vec::new();

        for (i, token) in tokens.iter().enumerate() {
            let prev = if i > 0 {
                Some(tokens[i - 1].as_str())
            } else {
                None
            };
            let next = if i < seq_len - 1 {
                Some(tokens[i + 1].as_str())
            } else {
                None
            };

            let mut scores = Vec::new();
            for label in 0..NUM_LABELS {
                scores.push(self.compute_emission(token, prev, next, label));
            }
            emissions.push(scores);
        }

        (emissions, self.transition.clone())
    }

    pub fn predict(&self, tokens: &[String]) -> Vec<usize> {
        let (emissions, transitions) = self.forward(tokens);

        let emissions_flat: Vec<f32> = emissions.iter().flatten().cloned().collect();
        viterbi_decode(&emissions_flat, &transitions, NUM_LABELS)
    }

    pub fn train_step(&mut self, tokens: &[String], true_labels: &[usize], _lr: f32) {
        // Simple perceptron-style update
        let preds = self.predict(tokens);

        for (i, (&pred, &true_label)) in preds.iter().zip(true_labels.iter()).enumerate() {
            if pred != true_label {
                // Update emission weights
                for label in 0..NUM_LABELS {
                    if label == true_label {
                        self.emission_weights[label] += 0.1;
                    } else if label == pred {
                        self.emission_weights[label] -= 0.1;
                    }
                }
            }
        }

        // Update transition matrix based on correct sequences
        for i in 1..true_labels.len().min(preds.len()) {
            let from = preds[i - 1];
            let to = preds[i];
            let correct_from = true_labels[i - 1];
            let correct_to = true_labels[i];

            if from != correct_from || to != correct_to {
                // Penalize wrong transitions
                self.transition[to * NUM_LABELS + from] -= 0.01;
                // Reward correct transitions
                self.transition[correct_to * NUM_LABELS + correct_from] += 0.01;
            }
        }

        // Constrain transitions
        // B-TITLE can only be followed by I-TITLE or O (not directly O after B without I)
        self.transition[0 * NUM_LABELS + 1] = self.transition[0 * NUM_LABELS + 1].min(-1.0);
        // B -> O is bad
    }

    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let json = serde_json::json!({
            "transition": self.transition,
            "emission_weights": self.emission_weights,
            "num_labels": NUM_LABELS,
        });
        std::fs::write(path, serde_json::to_string_pretty(&json).unwrap())?;
        Ok(())
    }

    pub fn load(path: &str) -> std::io::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let json: serde_json::Value = serde_json::from_str(&content).unwrap();

        let transition: Vec<f32> = json["transition"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        let emission_weights: Vec<f32> = json["emission_weights"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();

        Ok(Self {
            transition,
            emission_weights,
        })
    }
}

impl Default for CrfModel {
    fn default() -> Self {
        Self::new()
    }
}

pub fn viterbi_decode(emissions: &[f32], transitions: &[f32], num_labels: usize) -> Vec<usize> {
    if emissions.is_empty() || num_labels == 0 {
        return vec![];
    }

    let seq_len = emissions.len() / num_labels;
    if seq_len == 0 {
        return vec![];
    }

    let mut viterbi = vec![vec![f32::MIN / 1e10; num_labels]; seq_len];
    let mut backpointers = vec![vec![0usize; num_labels]; seq_len.saturating_sub(1).max(1)];

    // Initialize
    for j in 0..num_labels {
        if j < emissions.len() {
            viterbi[0][j] = emissions[j];
        }
    }

    // Forward pass
    for t in 1..seq_len {
        for j in 0..num_labels {
            let mut best_score = f32::MIN / 1e10;
            let mut best_prev = 0;

            for i in 0..num_labels {
                let score = viterbi[t - 1][i] + transitions[j * num_labels + i];
                if score > best_score {
                    best_score = score;
                    best_prev = i;
                }
            }

            let emission_idx = t * num_labels + j;
            if emission_idx < emissions.len() {
                viterbi[t][j] = best_score + emissions[emission_idx];
            }
            if t < backpointers.len() {
                backpointers[t][j] = best_prev;
            }
        }
    }

    // Backtrack
    let mut path = vec![0usize; seq_len];
    if seq_len > 0 {
        let last_row = &viterbi[seq_len - 1];
        path[seq_len - 1] = last_row
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        for t in (0..seq_len - 1).rev() {
            if t + 1 < backpointers.len() {
                path[t] = backpointers[t + 1][path[t + 1]];
            }
        }
    }

    path
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_viterbi() {
        let emissions = vec![0.1, 0.8, 0.1, 0.8, 0.1, 0.1];
        let transitions = vec![0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0];

        let path = viterbi_decode(&emissions, &transitions, 3);
        assert!(!path.is_empty());
    }
}
