# EmoNet Emotions via Empathic‑Insight‑Voice‑Plus

VoiceNet tells you **how** speech is delivered (57 perceptual voice/prosody dimensions). This guide
covers the complementary half of the stack: **which emotion is expressed**, using LAION's
**[Empathic‑Insight‑Voice‑Plus](https://huggingface.co/laion/Empathic-Insight-Voice-Plus)** model
over the **[EmoNet emotion taxonomy](../taxonomy/emonet_taxonomy.md)**.

> **The full "expressive speech" stack**
> - **VoiceNet** → *how it is said*: 57 voice/prosody dimensions (+ genuineness, vocal‑burst‑blend).
>   See [`inference.md`](inference.md).
> - **EmoNet / Empathic‑Insight‑Voice** → *which emotion is expressed*: 40 emotions + Valence +
>   Arousal + audio‑quality scores.
>
> Run both on the same clip to get a complete perceptual description of a voice performance.

---

## 1. What Empathic‑Insight‑Voice‑Plus predicts

For each clip, the model emits a score per **expert head**:

- **40 EmoNet emotions** — the full [EmoNet taxonomy](../taxonomy/emonet_taxonomy.md) (Amusement,
  Elation, Affection, Fear, Anger, Sadness, Contempt, … through Jealousy & Envy).
- **Valence** and **Arousal** — two continuous affect dimensions.
  → **40 emotions + Valence + Arousal = 42 emotion/affect outputs.**
- **Additional voice attributes** — e.g. Age, Gender, High‑Pitched vs. Low‑Pitched, Monotone vs.
  Expressive, Soft vs. Harsh, Warm vs. Cold, Submissive vs. Dominant, Confident vs. Hesitant,
  Serious vs. Humorous, Vulnerable vs. Emotionally Detached, Authenticity, Recording Quality,
  Background Noise.
- **4 audio‑quality scores** — **Overall Quality**, **Speech Quality**, **Background Quality**
  (distilled from Microsoft **DNSMOS**), and **Content Enjoyment** (distilled from Meta
  **AudioBox**). These are MOS‑like continuous scores.
- **A transcription** — the encoder is a Whisper variant, so a caption comes for free in the same
  pass.

The "Plus" in the name is the 4 quality experts on top of the base Empathic‑Insight‑Voice‑Small
emotion/attribute suite; together the pipeline expects **59 expert heads** per clip (55
emotion/attribute + 4 quality).

---

## 2. Architecture — one Whisper encode → many MLP experts

The design mirrors VoiceNet's "embed once, many cheap heads" philosophy, but on a Whisper encoder:

```
audio (mono, 16 kHz, 30 s) ──► BUD-E-Whisper encoder ──► encoder hidden states [1500 × 768]
                                    │
        ┌───────────────────────────┼───────────────────────────────┬─────────────────────┐
        ▼                           ▼                               ▼                     ▼
  Whisper decoder            emotion/attribute experts        quality experts        (same encode)
  → transcription            FullEmbeddingMLP                 PooledEmbeddingMLP
                             input = flattened [1500×768]     input = pooled [3072]
                             (~73.7M params each)             mean+min+max+std (~200K each)
```

- **Encoder:** **[laion/BUD-E‑Whisper](https://huggingface.co/laion/BUD-E-Whisper)** (a fine‑tuned
  Whisper‑Small). The encoder is run **once per clip**; its output is reused by the decoder (for the
  caption) **and** by every expert head — the encoder is never run twice.
- **Emotion / attribute experts (`FullEmbeddingMLP`):** consume the **full** encoder sequence,
  flattened to `1500 × 768`, projected to 64, then `hidden = [64, 32, 16]` → 1 output. These are
  large (~73.7M params each) because they read the whole sequence.
- **Quality experts (`PooledEmbeddingMLP`):** consume **pooled** features — `mean ⊕ min ⊕ max ⊕ std`
  over the sequence dimension → a **3072‑d** vector — then the same MLP shape. These are tiny
  (~200K params each).
- Each expert outputs one continuous score.

This is a distillation architecture: the emotion/attribute experts were trained on large‑scale
voice‑acting + in‑the‑wild speech, and the 4 quality experts distill DNSMOS / AudioBox.

---

## 3. Efficient batched inference

Two ready‑made run modes live in the LAION **emotion‑annotations** repo
(<https://github.com/LAION-AI/emotion-annotations>): a **standalone folder script** and a
**server/client** service. Both implement the same key optimization: **run the encoder once and
share its output across the decoder and all experts.**

### The core efficiency trick (from `annotate_audio.py` / `server.py`)

```python
with torch.no_grad():
    inputs = processor(waveforms, sampling_rate=16000, return_tensors="pt",
                       padding="max_length", truncation=True).to(device)

    # 1. Run the encoder ONCE for the whole batch
    encoder_outputs = whisper.get_encoder()(inputs.input_features, return_dict=True)

    # 2. Reuse it for transcription (decode)
    ids = whisper.generate(encoder_outputs=encoder_outputs)
    captions = processor.batch_decode(ids, skip_special_tokens=True)

    # 3. Reuse the SAME hidden states for every expert
    emb = encoder_outputs.last_hidden_state          # [B, 1500, 768]
    pooled = pool_embedding(emb.float())             # [B, 3072] mean+min+max+std, computed once

    for i in range(emb.size(0)):
        preds = {d: mlp(emb[i:i+1]).item()    for d, mlp in emotion_experts.items()}
        preds.update({d: mlp(pooled[i:i+1]).item() for d, mlp in quality_experts.items()})
```

Additional throughput optimizations used by the scripts:

- **FP16** (half precision) for encoder and experts on CUDA.
- **Flash Attention 2** for the Whisper encoder, with an automatic fallback to `sdpa` if unavailable.
- **`torch.compile`** on the MLP experts (`reduce-overhead`).
- **Multi‑GPU DDP:** `annotate_audio.py` shards the file list across all visible GPUs, one worker
  process per GPU, dynamic per‑GPU batch sizing (~3 GB/batch target).
- **Resume‑safe:** clips whose `.json` already contains all 59 expert scores are skipped.

### (a) Batch a folder (offline, multi‑GPU)

```bash
pip install torch transformers huggingface-hub librosa soundfile tqdm
python annotate_audio.py /path/to/audio_folder          # writes one .json next to each clip
# python annotate_audio.py /path/to/audio_folder --no-caption-sanitize   # keep raw captions
```

Each clip gets a JSON with the transcription plus all emotion/attribute/quality scores:

```json
{
  "source_audio_file": "clip.wav",
  "caption": "This is the transcribed text.",
  "emotions": {
    "Amusement": 0.04, "Interest": 2.82, "Contentment": 1.78,
    "Valence": 1.95, "Arousal": 3.01, "Age": 3.01, "Gender": 0.7,
    "score_overall_quality": 2.54, "score_speech_quality": 1.89,
    "score_background_quality": 3.22, "score_content_enjoyment": 4.10
  }
}
```

> **Caption note:** BUD‑E‑Whisper occasionally prepends one or two spurious capital letters
> (`"AThis is a sentence"` → `"This is a sentence"`). The scripts sanitize this by default; disable
> with `--no-caption-sanitize`.

### (b) Server / client (real‑time)

A FastAPI server pre‑loads all models once and uses **dynamic batching** (default `batch_size=16`,
`max_wait_time=0.05 s`) to coalesce concurrent requests into full GPU batches:

```bash
uvicorn server:app --host 0.0.0.0 --port 8022      # loads encoder + all experts once
python client.py --demo                            # single-file demo
python client.py --file audio.wav                  # one file
python client.py --folder ./audio --concurrency 32 # many files
python client.py --benchmark                       # 128 concurrent throughput test
```

The client sorts and prints the top emotions and attributes per clip.

---

## 4. Reading emotion outputs

Each emotion/attribute score is a continuous number; **rank the emotion experts and take the
top‑k** to get "which emotions are expressed" (the reference client shows the top‑5 emotions and
top‑5 attributes). **Valence** and **Arousal** give the overall affective position; the **quality**
experts are MOS‑like (roughly 1–5, higher is better) and are useful for **filtering** a dataset.
Full emotion definitions and synonym clusters are in
[`../taxonomy/emonet_taxonomy.md`](../taxonomy/emonet_taxonomy.md).

---

## 5. Combining VoiceNet + EmoNet

Run both models on the same clip and you get a complete perceptual profile:

| Question | Model | Output |
|---|---|---|
| *How is it delivered?* | **VoiceNet** | 57 voice/prosody dims (0–6) + genuineness + vocal‑burst‑blend |
| *Which emotion is expressed?* | **Empathic‑Insight‑Voice‑Plus (EmoNet)** | 40 emotions + Valence + Arousal + quality + caption |

Together they support expressive‑TTS conditioning, dataset curation and filtering, voice‑acting
analysis, and emotion research. See the emotion pipeline and scripts at
**<https://github.com/LAION-AI/emotion-annotations>**, and the VoiceNet inference recipe in
[`inference.md`](inference.md).

---

## Links

- Model: **[laion/Empathic‑Insight‑Voice‑Plus](https://huggingface.co/laion/Empathic-Insight-Voice-Plus)**
- Encoder: **[laion/BUD-E‑Whisper](https://huggingface.co/laion/BUD-E-Whisper)**
- Emotion pipeline & scripts: **[LAION-AI/emotion-annotations](https://github.com/LAION-AI/emotion-annotations)**
- EmoNet taxonomy: **[`../taxonomy/emonet_taxonomy.md`](../taxonomy/emonet_taxonomy.md)**
- Research line: *EMONET‑VOICE: A Fine‑Grained, Expert‑Verified Benchmark for Speech Emotion Detection*
