use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Linear, VarBuilder, Module};
use candle_transformers::models::distilbert::{Config, DistilBertModel};
use crate::parser::bio_tags::BioTag;

/// A Transformer-CRF sequence classification model (DistilBERT + Linear + Transitions)
pub struct CrfModel {
    pub distilbert: DistilBertModel,
    pub emission: Linear,
    pub transitions: Tensor,
}

impl CrfModel {
    /// Load the model from safetensors
    pub fn load(vb: VarBuilder, config: Config) -> Result<Self> {
        let distilbert = DistilBertModel::load(vb.pp("distilbert"), &config)?;
        
        let hidden_size = 768; // Default distilbert dimension
        let num_labels = BioTag::NUM_TAGS;
        
        // The linear layer for emission scores maps from hidden_size to num_labels
        // In Hugging Face sequence classification, this is usually named `classifier`
        let emission = candle_nn::linear(hidden_size, num_labels, vb.pp("classifier"))?;
        
        // Depending on whether CRF transitions were learned in python, we might load them here
        // If not, we can initialize a dummy transition matrix for the Viterbi decoder to use
        let dev = vb.device();
        let transitions = match vb.pp("crf_transitions").get((num_labels, num_labels), "weight") {
            Ok(t) => t,
            Err(_) => Tensor::zeros((num_labels, num_labels), DType::F32, dev)?,
        };

        Ok(Self {
            distilbert,
            emission,
            transitions,
        })
    }

    /// Forward pass producing emission scores
    /// `input_ids`: [batch_size, seq_len]
    /// `attention_mask`: [batch_size, seq_len]
    pub fn forward(&self, input_ids: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        // DistilBert forward returns either hidden states or multiple things depending on config
        let hidden_states = self.distilbert.forward(input_ids, attention_mask)?;
        
        // Pass the output of the transformer through the emission linear layer
        let emissions = self.emission.forward(&hidden_states)?;
        
        Ok(emissions)
    }
}
