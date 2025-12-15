use candle::{DType, Device, Result, Tensor};
use candle_core as candle;
use candle_core::IndexOp;
use candle_nn::{Linear, Module, VarBuilder};
use candle_transformers::models::bert::{BertModel, Config};
use std::path::Path;

struct BertForSequenceClassificationImpl {
    bert: BertModel,
    classifier: Linear,
}

impl BertForSequenceClassificationImpl {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let bert = if vb.contains_tensor("bert.embeddings.word_embeddings.weight") {
            BertModel::load(vb.pp("bert"), config)?
        } else if vb.contains_tensor("roberta.embeddings.word_embeddings.weight") {
            BertModel::load(vb.pp("roberta"), config)?
        } else {
            BertModel::load(vb.clone(), config)?
        };

        let hidden_size = config.hidden_size;
        let classifier = candle_nn::linear(hidden_size, 1, vb.pp("classifier"))?;

        Ok(Self { bert, classifier })
    }

    fn forward(
        &self,
        input_ids: &Tensor,
        token_type_ids: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let output = self
            .bert
            .forward(input_ids, token_type_ids, attention_mask)?;
        let cls_token = output.i((.., 0, ..))?;
        self.classifier.forward(&cls_token)
    }
}

#[derive(Clone)]
pub struct BertClassifier(std::sync::Arc<BertForSequenceClassificationImpl>);

impl BertClassifier {
    pub fn load<P: AsRef<Path>>(model_dir: P, device: &Device) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let config_path = model_dir.join("config.json");
        let weights_path = model_dir.join("model.safetensors");

        let config_content = std::fs::read_to_string(config_path)?;
        let config: Config = serde_json::from_str(&config_content)
            .map_err(|e| candle::Error::Msg(format!("Failed to parse config: {}", e)))?;

        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, device)? };

        let model = BertForSequenceClassificationImpl::load(vb, &config)?;

        Ok(Self(std::sync::Arc::new(model)))
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        token_type_ids: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        self.0.forward(input_ids, token_type_ids, attention_mask)
    }
}
