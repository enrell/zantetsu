//! Training loop for the CRF model.

use crate::data::{load_bio_dataset, CharVocab};
use crate::model::{viterbi_decode, CrfModel, NUM_LABELS};

pub struct Trainer {
    model: CrfModel,
    vocab: CharVocab,
}

impl Trainer {
    pub fn new() -> Self {
        let vocab = CharVocab::new();
        let model = CrfModel::new();

        Self { model, vocab }
    }

    pub fn train_on_file<P: AsRef<std::path::Path>>(
        &mut self,
        path: P,
        epochs: usize,
    ) -> anyhow::Result<()> {
        let examples = load_bio_dataset(path)?;
        println!("Loaded {} training examples", examples.len());

        let lr = 0.1f32;

        for epoch in 0..epochs {
            let mut correct = 0usize;
            let mut total = 0usize;

            // Shuffle
            let mut indices: Vec<usize> = (0..examples.len()).collect();
            for i in (1..indices.len()).rev() {
                let j = (epoch * 17 + i * 13) % (i + 1);
                indices.swap(i, j);
            }

            for (step, &idx) in indices.iter().enumerate() {
                let example = &examples[idx];
                if example.tokens.is_empty() {
                    continue;
                }

                // Training step
                self.model.train_step(&example.tokens, &example.labels, lr);

                // Evaluate
                let preds = self.model.predict(&example.tokens);

                for (i, &pred) in preds.iter().enumerate() {
                    if i < example.labels.len() {
                        if pred == example.labels[i] {
                            correct += 1;
                        }
                        total += 1;
                    }
                }

                if (step + 1) % 5000 == 0 {
                    let acc = if total > 0 {
                        correct as f32 / total as f32
                    } else {
                        0.0
                    };
                    println!(
                        "Epoch {}/{}, Step {}/{}, Accuracy: {:.2}%",
                        epoch + 1,
                        epochs,
                        step + 1,
                        examples.len(),
                        acc * 100.0
                    );
                }
            }

            let acc = if total > 0 {
                correct as f32 / total as f32
            } else {
                0.0
            };
            println!(
                "Epoch {}/{} complete - Accuracy: {:.2}%",
                epoch + 1,
                epochs,
                acc * 100.0
            );
        }

        Ok(())
    }

    pub fn save_model<P: AsRef<std::path::Path>>(&self, path: P) -> anyhow::Result<()> {
        self.model.save(path.as_ref().to_str().unwrap())?;
        println!("Model saved to {:?}", path.as_ref());
        Ok(())
    }
}

impl Default for Trainer {
    fn default() -> Self {
        Self::new()
    }
}

pub fn run_training() -> anyhow::Result<()> {
    let mut trainer = Trainer::new();

    let data_path = "data/training/bio_train_50k.txt";

    if !std::path::Path::new(data_path).exists() {
        anyhow::bail!("Training data not found: {}", data_path);
    }

    println!("Starting improved CRF training...");
    trainer.train_on_file(data_path, 3)?;

    std::fs::create_dir_all("models")?;
    trainer.save_model("models/crf_model_v2.json")?;

    Ok(())
}
