# Modern VLM Landscape vs. nanoVLM

This note captures the models we discussed—CLIP, BLIP, Flamingo—and highlights how nanoVLM fits alongside more recent vision-language systems.

---

## CLIP (2021)

- **Architecture:** Dual towers. A ViT/ResNet vision encoder and a Transformer text encoder each produce embeddings that are projected into a shared latent space. There is **no fusion transformer**; the modalities only meet through cosine similarity.
- **Training:** Contrastive pretraining (InfoNCE) on hundreds of millions of web-scale image–text pairs. The loss encourages matching image/text pairs and pushes apart mismatched pairs.
- **Strengths:** Excellent zero-shot classification, retrieval, and representation learning. Lightweight inference because the text encoder can be prompted with labels.
- **Limitations:** Not generative—no decoder—and no joint reasoning inside the model; downstream tasks need extra heads or prompting tricks.

---

## BLIP Family (2022–2023)

- **Architecture:** ViT visual backbone → Q-Former (a stack of cross-attention layers with learnable query tokens) → language module.
  - The **Q-Former** compresses the dense patch grid into a handful of query embeddings that summarize the image.
  - BLIP-2 replaces the language module with a frozen LLaMA/OPT and keeps the Q-Former + projection trainable.
- **Training:** Multi-stage. Contrastive objectives and image–text matching, followed by supervised captioning. Instruction-tuned variants add conversational finetuning.
- **Strengths:** Bidirectional understanding (ITM) and captioning in a single model. Q-Former gives more control over how many image tokens reach the decoder.
- **Limitations:** More moving parts than placeholder fusion; cross-attention stack adds latency; still limited to single-image or small multi-image setups.

---

## Flamingo (2022)

- **Architecture:** Frozen ViT-G/14 vision tower, Perceiver Resampler to produce a fixed set of latent visual tokens, and a large frozen language model (e.g. Chinchilla-80B) with **gated cross-attention (GCA) blocks** inserted between LM layers.
- **Training:** Massive curated multimodal corpora (billions of samples). Uses a mix of masked language modeling, contrastive terms, and instruction tuning to support few-shot dialogue.
- **Strengths:** Handles multi-image sequences, video frames, and long-context multimodal conversation. Near state-of-the-art—but extremely compute heavy.
- **Limitations:** Architecture and training pipeline are complex; replication requires large-scale infrastructure.

---

## Other 2023–2025 VLMs

- **Kosmos-2.5:** Multigranular region tokens, OCR supervision, multilingual coverage.
- **Qwen-VL / Qwen-VL-Chat:** SigLIP or Florence vision + Q-Former bridge + Qwen LLM, 20k context, bilingual training.
- **LLaVA-Next / LLaVA-1.6:** CLIP-L vision encoder with simple projector into Vicuna; heavy instruction tuning with GPT-4V data.
- **MiniGPT-4 / InstructBLIP:** BLIP-2 style bridging but smaller (Vicuna 7B).
- **Fuyu-8B:** Flattened image patches fed directly into the decoder without heavy preprocessing.
- **SmolVLM / SmolVLM-Instruct:** Pixel-shuffle bridge similar to nanoVLM but scaled to multi-image, longer context.

---

## nanoVLM in Context

- **Architecture:** SigLIP2-B/16 Vision Transformer + pixel-shuffle modality projector + SmolLM2 decoder. Visual tokens replace `<|image|>` placeholders directly inside the language model input (`models/vision_language_model.py`).
- **Training:** Finetunes on instruction-style datasets via `train.py`. Relies on pretrained backbones, only training the projector (and optionally fine-tuning the backbones). Integrates with `lmms-eval` for automated benchmarking.
- **Strengths:**
  - Minimal PyTorch code (~750 lines) for learners and rapid experiments.
  - Uses mainstream fusion strategy (placeholder replacement) without additional cross-attention modules.
  - Easy to swap backbones or tweak pixel-shuffle granularity through `VLMConfig`.
- **Trade-offs:**
  - Single-image focus and no dual-encoder contrastive training.
  - Smaller model (≈450 M params) than Flamingo/Qwen/Kosmos, so answer quality lags SOTA.
  - No dedicated components for OCR, multi-image dialogue, or video.

---

## Summary Table

| Model | Fusion Strategy | Decoder | Training Signal | Scale |
| --- | --- | --- | --- | --- |
| CLIP | None (dual encoders) | N/A | Contrastive | 100s M params / 400 M samples |
| BLIP / BLIP-2 | Q-Former cross-attention | BERT / LLaMA | Contrastive + captioning + ITM | 300 M–B+ |
| Flamingo | Gated cross-attention blocks | Chinchilla LM | Instruction + contrastive | 80 B |
| nanoVLM | Placeholder replacement + projector | SmolLM2 | Autoregressive QA/caption | 450 M |

---

### Takeaways

1. **Fusion mechanism defines complexity.** CLIP avoids fusion entirely; BLIP and Flamingo add cross-attention modules; nanoVLM sticks to the simplest approach—map and insert.
2. **Training scale drives capability.** CLIP and Flamingo leverage billions of samples; nanoVLM trades raw performance for approachability.
3. **Projector design is a key lever.** Pixel shuffle keeps nanoVLM fast and small, while BLIP’s Q-Former or Flamingo’s Perceiver add expressive power at compute cost.

Use this note when you need to explain where nanoVLM sits in the current multimodal ecosystem or when considering which architectural upgrades to prototype next.

