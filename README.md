# GPT-2 124M

A GPT-2-style language model implemented in PyTorch and pretrained from scratch on FineWeb-Edu, exploring language modeling and pretraining efficiency.

## Model
- **Architecture:** GPT-2 (124M parameters)
- **Dataset:** FineWeb-Edu
- **Tokenizer:** GPT-2 BPE
- **Objective:** Causal language modeling

## Training
Baseline POC: **500 steps**, loss **10.988 → 5.5684**.

## Inference & Evaluation
Autoregressive generation with greedy decoding. HellaSwag evaluation implemented.

## Research Focus
Reducing pretraining compute and training time through profiling, experimentation, and optimization.