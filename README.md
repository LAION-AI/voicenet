# VoiceNet

**Perceptual voice &amp; speech dimension models, data, and demos.**

> ⚠️ **Work in progress.** This is an early landing repository for the VoiceNet
> project. More models, checkpoints, evaluation code and documentation will be
> added here over time.

VoiceNet is a family of models that listen to a short speech clip and estimate
**57 human-interpretable perceptual voice &amp; speech dimensions** — qualities such
as voice age, arousal, valence, warmth, brightness, tempo, breathiness,
resonance placement, and speaking style — each described on an ordinal
**0–6 rubric**. Alongside the 57 dimensions, VoiceNet also predicts a
**Genuineness** score (0–6, how authentic vs. performed a delivery sounds) and a
**Vocal-Burst Blend** score (0–10, how naturally a laugh / sob / gasp / sigh
blends into the surrounding speech).

Every prediction is produced from a **single audio embedding** per clip (from the
VoiceCLAP-commercial encoder), which small model "heads" read to score each
dimension. Each dimension is served by two heads: a **regression** head (a smooth
continuous estimate) and a **classification** head (a hard 0–6 bucket with a
plain-language level description).

## 🔊 Live demo

**→ [https://laion-ai.github.io/voicenet/](https://laion-ai.github.io/voicenet/)**

An interactive showcase of VoiceNet predictions on 100 diverse, multilingual
[Emolia](https://huggingface.co/datasets/laion/emolia-voicenet-gemini-annotations)
speech clips. Each sample has an audio player, a radar profile of all 57
regression dimensions, the genuineness and vocal-burst-blend gauges, and a full
per-dimension table comparing the regression and classification predictions.

## Models (coming soon)

The current demo uses the **commercial** VoiceNet dimension-predictor heads,
trained on Gemini round-1+2 annotations. An updated round-1+2+3 version is in
training and the demo page will be regenerated when it lands.

- Model repo: **[laion/voicenet-dimension-predictors-commercial](https://huggingface.co/laion/voicenet-dimension-predictors-commercial)**

More heads, encoders, and evaluation artifacts will be linked here as they are
released.

## Data

- Annotations dataset: **[laion/emolia-voicenet-gemini-annotations](https://huggingface.co/datasets/laion/emolia-voicenet-gemini-annotations)**

## Taxonomy

The full rubric of all 57 dimensions and their 0–6 level definitions is in
**[`voicenet_taxonomy.md`](voicenet_taxonomy.md)**, based on the VoiceNet-Ext
framework (Schumann et al., 2025 — [arXiv:2505.20033](https://arxiv.org/abs/2505.20033)).

---

*More models, links, and documentation will be added. Stay tuned.*
