//! # Neural CRF Parser
//!
//! ML-based parser using a DistilBERT + CRF architecture for sequence labeling.
//! Uses candle for inference without external dependencies.

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::distilbert::Config as BertConfig;
use tokenizers::Tokenizer as HfTokenizer;

use crate::crf::model::CrfModel;
use crate::error::{Result, ZantetsuError};
use crate::parser::bio_tags::{BioTag, Entity, EntityType};
use crate::parser::viterbi::ViterbiDecoder;
use crate::types::{AudioCodec, EpisodeSpec, MediaSource, ParseResult, Resolution, VideoCodec};

/// Neural CRF Parser for anime filenames.
pub struct NeuralParser {
    hf_tokenizer: Option<HfTokenizer>,
    model: Option<CrfModel>,
    viterbi: ViterbiDecoder,
    device: Device,
}

impl NeuralParser {
    /// Create a new neural parser with lazy model initialization.
    pub fn new() -> Result<Self> {
        let device = Device::Cpu;

        Ok(Self {
            hf_tokenizer: None,
            model: None,
            viterbi: ViterbiDecoder::new(BioTag::NUM_TAGS),
            device,
        })
    }

    /// Initialize model with default paths (for production).
    /// If weights are missing, the parser will fail cleanly to trigger fallback.
    pub fn init_model(&mut self) -> Result<()> {
        // Try to load pre-trained weights from safetensors.
        // In this implementation, we look in the relative "models/ner_model" directory typically defined during the Python SFT step.
        let model_path = "models/ner_model/model.safetensors";
        let tokenizer_path = "models/ner_model/tokenizer.json";
        
        let tokenizer_file = std::path::Path::new(tokenizer_path);
        if tokenizer_file.exists() {
            let hf_tokenizer = HfTokenizer::from_file(tokenizer_file)
                .map_err(|e| ZantetsuError::NeuralParser(e.to_string()))?;
            self.hf_tokenizer = Some(hf_tokenizer);
        } else {
            return Err(ZantetsuError::NeuralParser(format!("Tokenizer not found at {}", tokenizer_path)));
        }

        let safetensors_path = std::path::Path::new(model_path);
        if !safetensors_path.exists() {
            return Err(ZantetsuError::NeuralParser(format!("Model not found at {}", model_path)));
        }

        // Load config from json
        let config_path = std::path::Path::new("models/ner_model/config.json");
        let config_str = std::fs::read_to_string(config_path)
            .map_err(|e| ZantetsuError::NeuralParser(format!("Failed to read config: {}", e)))?;
        let config: BertConfig = serde_json::from_str(&config_str)
            .map_err(|e| ZantetsuError::NeuralParser(format!("Failed to parse config: {}", e)))?;
        
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[safetensors_path], DType::F32, &self.device) }
            .map_err(|e| ZantetsuError::CandleError(e.to_string()))?;
            
        let model = CrfModel::load(vb, config)
            .map_err(|e| ZantetsuError::CandleError(e.to_string()))?;
            
        self.model = Some(model);
        Ok(())
    }

    /// Parse a filename using the neural CRF model.
    pub fn parse(&self, input: &str) -> Result<ParseResult> {
        if input.trim().is_empty() {
            return Err(ZantetsuError::EmptyInput);
        }

        let tokenizer = self.hf_tokenizer.as_ref().ok_or_else(|| {
            ZantetsuError::NeuralParser("Tokenizer is not initialized".into())
        })?;

        let model = self.model.as_ref().ok_or_else(|| {
            ZantetsuError::NeuralParser("Model is not initialized".into())
        })?;

        // 1. Tokenize using Hugging Face tokenizer to get wordpieces
        let encoding = tokenizer.encode(input, true).map_err(|e| {
            ZantetsuError::NeuralParser(format!("Tokenize error: {}", e))
        })?;

        let tokens = encoding.get_ids();
        if tokens.is_empty() {
            return Err(ZantetsuError::ParseFailed { input: input.to_string() });
        }

        // 2. Prepare tensors
        let input_ids = Tensor::new(tokens, &self.device)
            .map_err(|e| ZantetsuError::CandleError(e.to_string()))?
            .unsqueeze(0) // add batch dimension
            .map_err(|e| ZantetsuError::CandleError(e.to_string()))?;
            
        let attention_mask = Tensor::ones_like(&input_ids)
            .map_err(|e| ZantetsuError::CandleError(e.to_string()))?;

        // 3. Compute emission scores
        let emissions = model.forward(&input_ids, &attention_mask)
            .map_err(|e| ZantetsuError::CandleError(e.to_string()))?;
            
        // Shape of emissions should be [1, seq_len, num_tags]
        let emissions_vec = emissions.squeeze(0).map_err(|_| ZantetsuError::NeuralParser("Emission dimension mismatch".into()))?;
        let seq_len = tokens.len();
        
        // Convert to Vec<Vec<f32>> for Viterbi decoding
        let emissions_flat: Vec<f32> = emissions_vec.flatten_all()
            .map_err(|e| ZantetsuError::CandleError(e.to_string()))?
            .to_vec1()
            .map_err(|e| ZantetsuError::CandleError(e.to_string()))?;
            
        let mut scores = Vec::with_capacity(seq_len);
        let num_tags = BioTag::NUM_TAGS;
        for i in 0..seq_len {
            let start = i * num_tags;
            let end = start + num_tags;
            scores.push(emissions_flat[start..end].to_vec());
        }

        // Get transition matrix as 2D vec
        let transition_flat: Vec<f32> = model.transitions.flatten_all()
            .map_err(|e| ZantetsuError::CandleError(e.to_string()))?
            .to_vec1()
            .map_err(|e| ZantetsuError::CandleError(e.to_string()))?;
            
        let mut transition_matrix = vec![vec![0.0f32; num_tags]; num_tags];
        for i in 0..num_tags {
            for j in 0..num_tags {
                transition_matrix[i][j] = transition_flat[i * num_tags + j];
                // Apply strict constraints explicitly just in case the model didn't freeze them properly
                if !BioTag::is_valid_transition(BioTag::from_index(i).unwrap(), BioTag::from_index(j).unwrap()) {
                    transition_matrix[i][j] = -10000.0;
                }
            }
        }

        // 4. Decode optimal tag sequence using Viterbi
        let tag_indices = self.viterbi.decode_constrained(&scores, &transition_matrix)?;
        
        // 5. Build parsed entities from wordpiece segments mapping back to offsets
        let offsets = encoding.get_offsets();
        let entities = self.assemble_entities(input, offsets, &tag_indices)?;

        // 6. Build ParseResult
        let result = self.build_parse_result(input, &entities)?;

        Ok(result)
    }

    /// Assemble entities cleanly from HF subword tags and original string offset map.
    fn assemble_entities(&self, input: &str, offsets: &[(usize, usize)], tag_indices: &[usize]) -> Result<Vec<Entity>> {
        let mut entities = Vec::new();
        let mut i = 0;
        let tags: Vec<BioTag> = tag_indices.iter().map(|&id| BioTag::from_index(id).unwrap()).collect();

        while i < tags.len() {
            let tag = tags[i];

            if tag.is_begin() || tag.entity_type().is_some() {
                let entity_type = tag.entity_type().unwrap();
                let start_idx = i;
                
                // Track byte offsets on the original string
                let start_offset = offsets[start_idx].0;
                let mut end_offset = offsets[start_idx].1;
                
                i += 1;

                // Collect continuing wordpieces (Inside tags or same type for non-BIO elements)
                while i < tags.len() {
                    let next_tag = tags[i];
                    if next_tag.is_inside() && next_tag.entity_type() == Some(entity_type) {
                        end_offset = offsets[i].1;
                        i += 1;
                    } else if next_tag == tag && !tag.is_begin() && !tag.is_inside() {
                        // For non-BIO tags like Resolution, contiguous subwords share the same tag.
                        end_offset = offsets[i].1;
                        i += 1;
                    } else {
                        break;
                    }
                }

                // If offsets are 0, it means it's a special token (like [CLS] or [SEP]), skip it.
                if start_offset == 0 && end_offset == 0 {
                    continue;
                }

                let text = input[start_offset..end_offset].trim().to_string();

                if !text.is_empty() {
                    entities.push(Entity {
                        entity_type,
                        start_token: start_idx,
                        end_token: i,
                        text,
                    });
                }
            } else {
                i += 1;
            }
        }

        Ok(entities)
    }

    /// Build ParseResult from extracted entities using exact original snippets.
    fn build_parse_result(&self, input: &str, entities: &[Entity]) -> Result<ParseResult> {
        let mut title = None;
        let mut group = None;
        let mut episode = None;
        let mut season = None;
        let mut resolution = None;
        let mut video_codec = None;
        let mut audio_codec = None;
        let mut source = None;
        let mut year = None;
        let mut crc32 = None;
        let mut extension = None;
        let mut version = None;

        for entity in entities {
            match entity.entity_type {
                EntityType::Title => title = Some(entity.text.clone()),
                EntityType::Group => group = Some(entity.text.clone()),
                EntityType::Episode => {
                    if let Ok(num) = entity.text.parse::<u32>() {
                        episode = Some(EpisodeSpec::Single(num));
                    }
                }
                EntityType::Season => {
                    if let Ok(num) = entity.text.parse::<u32>() {
                        season = Some(num);
                    }
                }
                EntityType::Resolution => resolution = self.parse_resolution(&entity.text),
                EntityType::VCodec => video_codec = self.parse_video_codec(&entity.text),
                EntityType::ACodec => audio_codec = self.parse_audio_codec(&entity.text),
                EntityType::Source => source = self.parse_source(&entity.text),
                EntityType::Year => year = entity.text.parse::<u16>().ok(),
                EntityType::Crc32 => crc32 = Some(entity.text.clone()),
                EntityType::Extension => extension = Some(entity.text.clone()),
                EntityType::Version => {
                    version = entity.text.chars().find(|c| c.is_ascii_digit()).and_then(|c| c.to_digit(10)).map(|v| v as u8);
                }
            }
        }

        // A basic confidence heuristic based on non-empty extractions
        let extracted_count = [
            title.is_some(), group.is_some(), episode.is_some(),
            season.is_some(), resolution.is_some(), video_codec.is_some(),
            audio_codec.is_some(), source.is_some(), year.is_some(),
            crc32.is_some(), extension.is_some()
        ].iter().filter(|&&x| x).count();
        
        let confidence = (extracted_count as f32 / 11.0).clamp(0.0, 1.0);

        Ok(ParseResult {
            input: input.to_string(),
            title,
            group,
            episode,
            season,
            resolution,
            video_codec,
            audio_codec,
            source,
            year,
            crc32,
            extension,
            version,
            confidence,
            parse_mode: crate::types::ParseMode::Full,
        })
    }

    fn parse_resolution(&self, text: &str) -> Option<Resolution> {
        let t = text.to_lowercase();
        if t.contains("2160") { Some(Resolution::UHD2160) }
        else if t.contains("1080") { Some(Resolution::FHD1080) }
        else if t.contains("720") { Some(Resolution::HD720) }
        else if t.contains("480") { Some(Resolution::SD480) }
        else { None }
    }

    fn parse_video_codec(&self, text: &str) -> Option<VideoCodec> {
        let t = text.to_lowercase();
        if t.contains("av1") { Some(VideoCodec::AV1) }
        else if t.contains("265") || t.contains("hevc") { Some(VideoCodec::HEVC) }
        else if t.contains("264") || t.contains("h264") || t.contains("h.264") { Some(VideoCodec::H264) }
        else if t.contains("vp9") { Some(VideoCodec::VP9) }
        else if t.contains("mpeg4") || t.contains("mp4") || t.contains("xvid") { Some(VideoCodec::MPEG4) }
        else { None }
    }

    fn parse_audio_codec(&self, text: &str) -> Option<AudioCodec> {
        let t = text.to_lowercase();
        if t.contains("flac") { Some(AudioCodec::FLAC) }
        else if t.contains("truehd") { Some(AudioCodec::TrueHD) }
        else if t.contains("dts") { Some(AudioCodec::DTS) }
        else if t.contains("opus") { Some(AudioCodec::Opus) }
        else if t.contains("aac") { Some(AudioCodec::AAC) }
        else if t.contains("ac3") || t.contains("dolby") { Some(AudioCodec::AC3) }
        else if t.contains("vorbis") || t.contains("ogg") { Some(AudioCodec::Vorbis) }
        else if t.contains("mp3") { Some(AudioCodec::MP3) }
        else { None }
    }

    fn parse_source(&self, text: &str) -> Option<MediaSource> {
        let t = text.to_lowercase();
        if t.contains("remux") { Some(MediaSource::BluRayRemux) }
        else if t.contains("webdl") || t.contains("web-dl") || t.contains("webrip") { Some(MediaSource::WebDL) }
        else if t.contains("bluray") || t.contains("blu-ray") || t.contains("bd") { Some(MediaSource::BluRay) }
        else if t.contains("hdtv") { Some(MediaSource::HDTV) }
        else if t.contains("dvd") { Some(MediaSource::DVD) }
        else if t.contains("vhs") { Some(MediaSource::VHS) }
        else { None }
    }
}

impl Default for NeuralParser {
    fn default() -> Self {
        Self::new().expect("Failed to create NeuralParser")
    }
}
