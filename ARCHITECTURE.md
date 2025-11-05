# nanoVLM Architecture Overview

This guide focuses on the vision components of **nanoVLM**—how raw pixels are transformed into language-ready embeddings—and how those embeddings are merged with text tokens inside the model. The language decoder mirrors modern LLaMA-style transformers and is therefore only referenced where it interacts with the vision stack.

---

## Data Flow at a Glance

```
image(s) ──▶ ViT encoder ──▶ patch grid
                         └─▶ pixel shuffle ──▶ projected visual tokens
prompt text ──▶ tokenizer ─────────────────────────────────────────────┐
                                                                      ▼
                                       replace <|image|> placeholders with visual tokens
                                                                ▼
                                                     language decoder (SmolLM2)
                                                                ▼
                                                   next-token logits / loss
```

---

## Vision Encoder (`models/vision_transformer.py`)

### Backbone
- nanoVLM reimplements a SigLIP-style Vision Transformer (ViT) in ~150 lines of PyTorch.
- `ViTPatchEmbeddings` splits an RGB image into non-overlapping patches via a strided 2D convolution. Patches are flattened and combined with learnable positional embeddings. Using a patch size of 16 on a 512×512 image yields 1024 tokens per image.
- Each `ViTBlock` applies LayerNorm → Multi-Head Self Attention → residual, followed by another LayerNorm → feed-forward MLP → residual. Dropout defaults to zero but can be enabled in `VLMConfig`.

### Pretrained Weights
- When you instantiate the model with `load_backbone=True`, `ViT.from_pretrained` pulls weights from the Hugging Face repo declared in `VLMConfig.vit_model_type` (default `google/siglip2-base-patch16-512`).
- The loader maps HF tensor names to this lightweight implementation, concatenating the separate Q/K/V projections from SigLIP into the combined `qkv_proj` used here.

### Output
- The encoder returns a `(num_patches, vit_hidden_dim)` sequence per image. If `vit_cls_flag` is `True`, a class token is prepended and pooled; otherwise the entire grid is passed downstream.
- No pooling or projection happens inside the ViT—downstream modules decide how to consume the spatial grid.

---

## Modality Projection (`models/modality_projector.py`)

### Why a Projection Layer?
The ViT emits a long grid of patch embeddings in the SigLIP hidden dimension (e.g. 1024 tokens × 1152 channels). Feeding that directly into the language model would explode the sequence length. The modality projector condenses vision tokens and aligns them with the decoder width.

### Pixel Shuffle Downsampling
- `pixel_shuffle` reshapes the square patch grid into a coarse grid by grouping `mp_pixel_shuffle_factor`×`mp_pixel_shuffle_factor` neighborhoods (default factor 4).
- For a 32×32 patch grid, the shuffle yields an 8×8 grid (64 visual tokens) while increasing the channel dimension by `factor²`.
- The op is inspired by SmolVLM/SmolLM and preserves locality: each new token summarises a small image region.

### Linear Projection
- After reshaping, a single `nn.Linear` maps from the expanded vision width to `lm_hidden_dim` (the decoder width).
- The projector is the only vision component trained from scratch; all decoder/encoder loads can remain partially frozen during fine-tuning if desired.

---

## Token/Image Fusion (`models/vision_language_model.py`)

### Placeholder Tokens
- Training samples insert `<|image|>` (and, when needed, grid-specific tokens like `<row_3_col_5>`) into the text prompt. These placeholders mark where visual context should appear.
- The tokenizer defined by `VLMConfig` includes these extra tokens in its vocabulary.

### Runtime Fusion
1. `VisionLanguageModel.forward` embeds the input text with `LanguageModel.token_embedding`.
2. It pushes images through the ViT and modality projector, producing a batch of visual tokens shaped `(num_visual_tokens, lm_hidden_dim)`.
3. `_replace_img_tokens_with_embd` builds a boolean mask over the input IDs where `<|image|>` appears and substitutes the corresponding visual embeddings in-place. The rest of the textual embeddings remain untouched.
4. The fused sequence is fed to the language decoder exactly as if it were a pure text prompt—the decoder is agnostic to which positions came from images.

### Multiple Images
- Inputs may contain multiple images per example; the data pipeline precomputes all necessary `<row_i_col_j>` tokens to mark splits.
- `_process_images` flattens nested image lists, concatenates the projected tokens, and drops the optional global token if the tokenizer lacks it. This keeps batching simple: all image tokens enter the decoder as a single contiguous block.

### Generation Path
- During inference (`generate`), nanoVLM performs a multimodal prefill pass with the fused prompt, caching keys/values.
- Autoregressive decoding then mirrors standard language models—no special handling for vision tokens is required after the initial replacement step.

---

## Configuration Levers (`models/config.py`)

| Field | Effect |
| --- | --- |
| `vit_model_type`, `vit_img_size`, `vit_patch_size` | Swap SigLIP checkpoints or change patch granularity. Larger images increase the pre-projection token count. |
| `mp_pixel_shuffle_factor`, `mp_image_token_length` | Control how aggressively the pixel shuffle downsamples the grid. Higher factors shrink the visual sequence length but coarsen spatial detail. |
| `lm_hidden_dim`, `lm_model_type` | Must match the language decoder width; projector output adapts automatically. |
| `vlm_extra_tokens` | Configure the placeholder vocabulary inserted by the dataset processor. |

Tweaking these values lets you experiment with latency vs. fidelity, or substitute other vision backbones / decoder widths while keeping the fusion logic unchanged.

---

## Key Takeaways

- The ViT is lightweight but fully compatible with pretrained SigLIP, giving you strong visual features without external dependencies.
- Pixel shuffle + linear projection is the bridge between vision and language, creating a compact set of visual tokens aligned with the decoder’s embedding space.
- Placeholder tokens make fusion explicit: you always know where image context sits in the prompt, and swapping embeddings in-place keeps the language model architecture untouched.

Armed with these pieces, you can modify the vision side of nanoVLM—swap backbones, change token layouts, or investigate new fusion schemes—without rewriting the rest of the training stack.

