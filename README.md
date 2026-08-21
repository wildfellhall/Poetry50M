# Poetry50M
A small language model trained from scratch on the DanFosing/public-domain-poetry dataset on HuggingFace. 

## Built With: 
Dataset: DanFosing/public-domain-poetry on HuggingFace
Libraries: Pytorch

Config: 
=== MODEL CONFIGURATION ===
{
  "transformers_version": "5.15.0",
  "architectures": null,
  "output_hidden_states": false,
  "return_dict": true,
  "dtype": null,
  "chunk_size_feed_forward": 0,
  "is_encoder_decoder": false,
  "id2label": {
    "0": "LABEL_0",
    "1": "LABEL_1"
  },
  "label2id": {
    "LABEL_0": 0,
    "LABEL_1": 1
  },
  "problem_type": null,
  "vocab_size": 16384,
  "n_positions": 1024,
  "n_embd": 512,
  "n_layer": 12,
  "n_head": 8,
  "n_inner": null,
  "activation_function": "gelu_new",
  "resid_pdrop": 0.1,
  "embd_pdrop": 0.1,
  "attn_pdrop": 0.1,
  "layer_norm_epsilon": 1e-05,
  "initializer_range": 0.02,
  "summary_type": "cls_index",
  "summary_use_proj": true,
  "summary_activation": null,
  "summary_proj_to_labels": true,
  "summary_first_dropout": 0.1,
  "scale_attn_weights": true,
  "use_cache": true,
  "bos_token_id": 1,
  "eos_token_id": 5,
  "pad_token_id": 0,
  "scale_attn_by_inverse_layer_idx": false,
  "reorder_and_upcast_attn": false,
  "add_cross_attention": false,
  "tie_word_embeddings": true,
  "_name_or_path": "",
  "n_ctx": 1024,
  "model_type": "gpt2",
  "output_attentions": false
}

=== PARAMETER BREAKDOWN BY MODULE ===
 - transformer    :   46,742,528 params
---------------------------------------
 TOTAL TRAINABLE PARAMETERS: 46,742,528
 HARD LIMIT STATUS: PASS

AI Assistance: Gemini 3.1 Flash, GPT 5.6 Sol for discussion, planning, and training script refinement
## Hardware: 
Using Google Colab Pro, I trained this model on an A100 GPU. 
