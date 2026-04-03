//! Character-Level CNN for anime filename NER
//!
//! Lightweight CNN + BiLSTM model for character-level sequence labeling.
//! Designed for fast inference without external ML dependencies.

use std::collections::HashMap;

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder};

const NUM_TAGS: usize = 17;
const MAX_LEN: usize = 256;
const CHAR_VOCAB_SIZE: usize = 97;
const CHAR_DIM: usize = 128;
const CNN_FILTERS: usize = 64;
const HIDDEN_DIM: usize = 128;

#[derive(Debug, Clone)]
pub struct CharVocab {
    char_to_idx: HashMap<char, usize>,
}

impl CharVocab {
    pub fn new() -> Self {
        let mut char_to_idx = HashMap::new();
        char_to_idx.insert('<', 1);
        for i in 32..127 {
            char_to_idx.insert(char::from_u32(i as u32).unwrap(), (i - 32 + 1) as usize);
        }
        char_to_idx.insert('<', 95);
        char_to_idx.insert('>', 96);
        Self { char_to_idx }
    }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        text.chars()
            .take(MAX_LEN - 2)
            .map(|c| *self.char_to_idx.get(&c).unwrap_or(&96))
            .collect()
    }
}

#[derive(candle_nn::Module)]
pub struct CharacterCNN {
    embedding: candle_nn::Embedding,
    conv3: candle_nn::Conv1d,
    conv5: candle_nn::Conv1d,
    conv7: candle_nn::Conv1d,
    highway_input: Linear,
    highway_gate: Linear,
    lstm: candle_nn::LSTM,
    classifier: Linear,
}

impl CharacterCNN {
    pub fn load(vb: VarBuilder) -> Result<Self> {
        let embedding = candle_nn::embedding(CHAR_VOCAB_SIZE, CHAR_DIM, vb.pp("embedding"))?;
        let conv3 = candle_nn::conv1d(CHAR_DIM, CNN_FILTERS, 3, vb.pp("conv3"))?;
        let conv5 = candle_nn::conv1d(CHAR_DIM, CNN_FILTERS, 5, vb.pp("conv5"))?;
        let conv7 = candle_nn::conv1d(CHAR_DIM, CNN_FILTERS, 7, vb.pp("conv7"))?;
        let highway_input =
            candle_nn::linear(CNN_FILTERS * 3, CNN_FILTERS * 3, vb.pp("highway_input"))?;
        let highway_gate =
            candle_nn::linear(CNN_FILTERS * 3, CNN_FILTERS * 3, vb.pp("highway_gate"))?;
        let lstm = candle_nn::lstm(CNN_FILTERS * 3, HIDDEN_DIM, 2, vb.pp("lstm"))?;
        let classifier = candle_nn::linear(HIDDEN_DIM * 2, NUM_TAGS, vb.pp("classifier"))?;

        Ok(Self {
            embedding,
            conv3,
            conv5,
            conv7,
            highway_input,
            highway_gate,
            lstm,
            classifier,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let embedded = self.embedding.forward(x)?;
        let embedded = embedded.transpose(1, 2)?;

        let h3 = candle_nn::activation::relu(&self.conv3.forward(&embedded)?)?;
        let h5 = candle_nn::activation::relu(&self.conv5.forward(&embedded)?)?;
        let h7 = candle_nn::activation::relu(&self.conv7.forward(&embedded)?)?;

        let concatenated = Tensor::cat(&[&h3, &h5, &h7], 1)?;
        let concatenated = concatenated.transpose(1, 2)?;

        let highway_t = candle_nn::activation::sigmoid(&self.highway_gate.forward(&concatenated)?)?;
        let highway_h = candle_nn::activation::relu(&self.highway_input.forward(&concatenated)?)?;
        let highway_out =
            (&highway_t * &highway_h)? + ((&highway_t.ones_like()? - &highway_t)? * &concatenated)?;

        let (_, (hidden, _)) = self.lstm.forward(&highway_out, None)?;
        let lstm_out = hidden.transpose(0, 1)?.flatten_from(1)?;

        let emissions = self.classifier.forward(&lstm_out)?;
        emissions.reshape((x.dim(0)?, x.dim(1)?, NUM_TAGS))
    }
}

#[derive(candle_nn::Module)]
pub struct CrfLayer {
    transitions: Linear,
    start_transitions: Linear,
    end_transitions: Linear,
}

impl CrfLayer {
    pub fn load(vb: VarBuilder) -> Result<Self> {
        let transitions = candle_nn::linear(NUM_TAGS, NUM_TAGS, vb.pp("crf_transitions"))?;
        let start_transitions = candle_nn::linear(NUM_TAGS, 1, vb.pp("crf_start"))?;
        let end_transitions = candle_nn::linear(NUM_TAGS, 1, vb.pp("crf_end"))?;
        Ok(Self {
            transitions,
            start_transitions,
            end_transitions,
        })
    }

    pub fn decode(&self, emissions: &Tensor) -> Result<Vec<Vec<usize>>> {
        let batch_size = emissions.dim(0)?;
        let seq_len = emissions.dim(1)?;

        let mut results = Vec::with_capacity(batch_size);

        for b in 0..batch_size {
            let emission_slice = emissions.get(b)?;
            let path = self.viterbi_decode_slice(&emission_slice, seq_len)?;
            results.push(path);
        }

        Ok(results)
    }

    fn viterbi_decode_slice(&self, emissions: &Tensor, seq_len: usize) -> Result<Vec<usize>> {
        let start_scores = self.start_transitions.forward(emissions)?;
        let mut score = start_scores.squeeze(1)?;

        for i in 1..seq_len {
            let prev_score = score.unsqueeze(1)?;
            let trans_scores = self.transitions.forward(emissions)?;
            let next_score = (prev_score + &trans_scores)?.max(&emissions.get(i)?.unsqueeze(1))?;
            score = next_score.squeeze(1)?;
        }

        let end_scores = self.end_transitions.forward(emissions)?;
        score = (&score + &end_scores.squeeze(1))?;

        let (_, best_last_tag) = score.argmax(0, DType::U32, false)?;

        let mut best_path = vec![best_last_tag.to_scalar::<u32>()? as usize];

        let mut current_score = score;
        for i in (0..seq_len - 1).rev() {
            let prev_scores = current_score.unsqueeze(1)?;
            let trans_scores_t = self.transitions.forward(emissions)?.t()?;
            let scores = (prev_scores + &trans_scores_t.unsqueeze(0))?;
            let (_, prev_tag) = scores.argmax(0, DType::U32, false)?;
            let tag_idx = prev_tag.to_scalar::<u32>()? as usize;
            best_path.insert(0, tag_idx);
            current_score = emissions.get(i)?;
        }

        Ok(best_path)
    }
}

pub struct CharCnnParser {
    vocab: CharVocab,
    cnn: CharacterCNN,
    crf: CrfLayer,
    device: Device,
}

impl CharCnnParser {
    pub fn new() -> Result<Self> {
        let device = Device::Cpu;
        let vocab = CharVocab::new();
        Ok(Self {
            vocab,
            cnn: CharacterCNN::load,
            crf: CrfLayer::load,
            device,
        })
    }

    pub fn load_model(model_path: &str) -> Result<Self> {
        let device = Device::Cpu;
        let vocab = CharVocab::new();

        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[model_path], DType::F32, &device)? };

        let cnn = CharacterCNN::load(vb.clone())?;
        let crf = CrfLayer::load(vb)?;

        Ok(Self {
            vocab,
            cnn,
            crf,
            device,
        })
    }

    pub fn parse(&self, input: &str) -> Result<Vec<(usize, usize, &'static str)>> {
        let char_indices = self.vocab.encode(input);

        let mut padded = char_indices.clone();
        padded.resize(MAX_LEN, 0);

        let input_tensor = Tensor::new(padded, &self.device)?.unsqueeze(0)?;

        let emissions = self.cnn.forward(&input_tensor)?;

        let tags = self.crf.decode(&emissions)?;

        let mut entities = Vec::new();
        let mut current_entity: Option<(usize, usize, &'static str)> = None;

        for (i, &tag) in tags[0].iter().enumerate() {
            if i >= char_indices.len() {
                break;
            }

            let tag_name: &'static str = match tag {
                0 => "O",
                1 => "TITLE",
                2 => "TITLE",
                3 => "GROUP",
                4 => "GROUP",
                5 => "EPISODE",
                6 => "EPISODE",
                7 => "SEASON",
                8 => "SEASON",
                9 => "RESOLUTION",
                10 => "VCODEC",
                11 => "ACODEC",
                12 => "SOURCE",
                13 => "YEAR",
                14 => "CRC32",
                15 => "EXTENSION",
                16 => "VERSION",
                _ => "O",
            };

            if tag == 1 || tag == 3 || tag == 5 || tag == 7 {
                if let Some(entity) = current_entity.take() {
                    entities.push(entity);
                }
                current_entity = Some((i, i + 1, tag_name));
            } else if (tag == 2 || tag == 4 || tag == 6 || tag == 8) && current_entity.is_some() {
                if let Some((_, end, _)) = current_entity.as_mut() {
                    *end = i + 1;
                }
            } else if tag != 0 && tag != 2 && tag != 4 && tag != 6 && tag != 8 {
                if let Some(entity) = current_entity.take() {
                    entities.push(entity);
                }
                entities.push((i, i + 1, tag_name));
            } else if tag == 0 {
                if let Some(entity) = current_entity.take() {
                    entities.push(entity);
                }
            }
        }

        if let Some(entity) = current_entity {
            entities.push(entity);
        }

        Ok(entities)
    }
}

impl Default for CharCnnParser {
    fn default() -> Self {
        Self::new().expect("Failed to create CharCnnParser")
    }
}
