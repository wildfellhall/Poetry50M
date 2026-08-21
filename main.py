"""
Main training script - iteration 1
"""

# Preprocessing, Dataset Formatting - Cell 1: 
!pip install -q transformers datasets tokenizers accelerate torch

import os
import torch
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast, GPT2Config, GPT2LMHeadModel, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# 1. Load the Dataset
dataset = load_dataset("DanFosing/public-domain-poetry")

# 2. Format with Structural Tokens (Preserves stanza & line breaks)
def format_poem(example):
    title = example.get('title', 'Untitled') or 'Untitled'
    author = example.get('author', 'Unknown') or 'Unknown'
    body = example.get('poem', example.get('text', example.get('content', '')))
    
    formatted_text = (
        f"<|startofpoem|>\n"
        f"<|title|> {title.strip()}\n"
        f"<|author|> {author.strip()}\n"
        f"<|body|>\n"
        f"{body.strip()}\n"
        f"<|endofpoem|>"
    )
    return {"text": formatted_text}

formatted_dataset = dataset.map(format_poem)

# Export raw text to train custom BPE Tokenizer
with open("corpus.txt", "w", encoding="utf-8") as f:
    for text in formatted_dataset['train']['text']:
        f.write(text + "\n")

# Tokenizer - Cell 2: 
# Train BPE Tokenizer with small vocab to save embedding parameters

special_tokens = [
    "<|pad|>",
    "<|startofpoem|>",
    "<|title|>",
    "<|author|>",
    "<|body|>",
    "<|endofpoem|>"
]

bpe_tokenizer = ByteLevelBPETokenizer()
bpe_tokenizer.train(
    files=["corpus.txt"],
    vocab_size=16384,  # Small vocab keeps embedding params < 8.4M
    min_frequency=2,
    special_tokens=special_tokens
)

# Save and wrap as a Hugging Face Fast Tokenizer
os.makedirs("poetry_tokenizer", exist_ok=True)
bpe_tokenizer.save_model("poetry_tokenizer")

tokenizer = PreTrainedTokenizerFast(
    tokenizer_file="poetry_tokenizer/vocab.json",
    vocab_file="poetry_tokenizer/vocab.json",
    merges_file="poetry_tokenizer/merges.txt",
    bos_token="<|startofpoem|>",
    eos_token="<|endofpoem|>",
    pad_token="<|pad|>",
    unk_token="<|pad|>"
)


