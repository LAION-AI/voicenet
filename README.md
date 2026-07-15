# VoiceNet

**Predicting *how* speech is delivered — 57 perceptual voice & prosody dimensions from a single audio embedding.**

VoiceNet is a family of models that listen to a short speech clip and estimate **57
human‑interpretable perceptual voice/prosody dimensions** — qualities such as voice age, arousal,
valence, warmth, brightness, tempo, breathiness, resonance placement, and speaking style — each on
an ordinal **0–6 rubric** (`EXPL`, content appropriateness, is 0–2). Alongside the 57 dimensions,
VoiceNet also predicts a **Genuineness** score (0–6, how authentic vs. performed a delivery sounds)
and a **Vocal‑Burst Blend** score (0–10, how naturally a laugh / sob / gasp / sigh blends into the
surrounding speech).

Every prediction is produced from **one 768‑d audio embedding per clip** (from the
**VoiceCLAP‑commercial** encoder). Small model "heads" then read that single vector to score each
dimension — so a clip is encoded **once** and all heads run essentially for free. Each dimension is
served by **two heads**: a **regression** head (a smooth continuous estimate) and a
**classification** head (a hard 0–6 bucket with a plain‑language level description).

> **VoiceNet answers *how* something is said. Its companion, EmoNet, answers *which emotion* is
> expressed.** Run both for a complete perceptual description of a voice performance — see the
> [EmoNet / Empathic‑Insight guide](guides/emonet_empathic_insight.md).

> 📄 **Paper.** The VoiceNet taxonomy and project come from the **forthcoming VoiceNet paper**,
> which is **not yet published**. This repository is the practical companion: taxonomy, trained
> models, data, and run recipes. (The extended dimension rubric builds on the VoiceNet‑Ext
> framework, [arXiv:2505.20033](https://arxiv.org/abs/2505.20033).)

---

## 🔊 Live demo

**→ [https://projects.laion.ai/voicenet/](https://projects.laion.ai/voicenet/)**

An interactive showcase of VoiceNet predictions on 100 diverse, multilingual
[Emolia](https://huggingface.co/datasets/laion/emolia-voicenet-gemini-annotations) speech clips.
Each sample has an audio player, a radar profile of all 57 regression dimensions, the genuineness
and vocal‑burst‑blend gauges, and a full per‑dimension table comparing the regression and
classification predictions.

### 🆕 v3 grid with procedural captions

**→ [https://projects.laion.ai/voicenet/grid-v3/](https://projects.laion.ai/voicenet/grid-v3/)**

Predictions from the **current best‑per‑dimension v3 models**, re‑run on the same 100 clips. Each
card leads with a **procedural caption** — the 5 most distinctive dimensions for that voice,
computed as the largest z‑score deviations from a **1000‑clip Emolia average** and phrased in plain
English (e.g. *"Extremely full and cinematic."*, *"Very cartoonish and exaggerated."*). No taxonomy
knowledge needed to read a voice at a glance.

---

## 📚 Documentation — table of contents

| Document | What's inside |
|---|---|
| **[taxonomy/voicenet_taxonomy.md](taxonomy/voicenet_taxonomy.md)** | The full **57‑dimension** rubric — every dimension's code, name, group, and 0–6 level definitions. |
| **[taxonomy/emonet_taxonomy.md](taxonomy/emonet_taxonomy.md)** | The **40 EmoNet emotions** and their synonym clusters (the target space of the emotion model, which also predicts Valence + Arousal). |
| **[guides/inference.md](guides/inference.md)** | **Run the models.** Load the model repo, run `predict.py`, get all **57 dims + genuineness + vocal‑burst‑blend** from one embedding; load audio from **WebDataset** shards or individual **mp3/wav**; the **measured batch‑throughput optimization guide** (batch sizing, DataLoader workers, multi‑GPU scaling, why I/O is the bottleneck). |
| **[guides/training.md](guides/training.md)** | **How the models were trained** — Gemini labels, dual‑CLAP candidate selection, head architecture, balancing, metrics, and what's easy vs. hard. |
| **[guides/emonet_empathic_insight.md](guides/emonet_empathic_insight.md)** | **Also get the emotion.** Obtain the **40 EmoNet emotions + Valence/Arousal + audio‑quality scores** via **Empathic‑Insight‑Voice‑Plus** — architecture summary + the efficient batched‑inference recipe. |

---

## 🧠 Models

**[laion/voicenet-dimension-predictors-commercial](https://huggingface.co/laion/voicenet-dimension-predictors-commercial)**
— self‑contained model repo: bundled VoiceCLAP‑commercial embedder + `predict.py` + `dimensions.py`
+ **57 regression** and **57 classification** best‑per‑dimension heads (all `variants/cap3x|cap4x|cap5x`
also shipped), plus the **genuineness** (0–6) and **vocal‑burst‑blend** (0–10) heads that run on the
same 768‑d embedding.

```python
from predict import VoiceNetPredictor
p = VoiceNetPredictor()            # bundled embedder + all 114 dimension heads
out = p.predict("audio.wav")       # ONE encode -> 57 dims
print(out["AROU"])                 # {'name':'Arousal','reg_score':..,'cls_bucket':..,'confidence':..}
```

One 80 GB GPU encodes at **~54 clips/s** (≈ 1600× real‑time); the 116 heads on a 512‑clip batch run
in ~6.2 ms (~82,000 clips/s), so the encoder is the only cost. Full recipe and measured numbers in
[guides/inference.md](guides/inference.md). Best‑per‑dimension mean metrics on the frozen validation
split: **reg‑MAE 0.744, Pearson r 0.791, within‑1 0.870, classification accuracy 0.617**.

## 📊 Data

**[laion/emolia-voicenet-gemini-annotations](https://huggingface.co/datasets/laion/emolia-voicenet-gemini-annotations)**
— **468,180** dimension‑level annotations over **236,613** multilingual Emolia clips, scored 0–6 by
**Gemini‑3.5‑flash** (non‑thinking, temperature 0) across three rounds (~$265 total). Ships the
annotations, audio provenance manifest, per‑dimension balance statistics, and the taxonomy. See
[guides/training.md](guides/training.md).

---

## The two‑layer stack

| Question | Project | Output |
|---|---|---|
| **How is it delivered?** | **VoiceNet** (this repo) | 57 voice/prosody dimensions (0–6) + genuineness + vocal‑burst‑blend |
| **Which emotion is expressed?** | **EmoNet / [Empathic‑Insight‑Voice‑Plus](https://huggingface.co/laion/Empathic-Insight-Voice-Plus)** | 40 emotions + Valence + Arousal + audio‑quality scores |

The emotion side is built and documented in **[LAION-AI/emotion-annotations](https://github.com/LAION-AI/emotion-annotations)**.

---

## Repository layout

```
voicenet/
├── README.md                              ← you are here
├── voicenet_taxonomy.md                   ← 57-dimension rubric (root copy)
├── taxonomy/
│   ├── voicenet_taxonomy.md               ← 57 voice/prosody dimensions × 0–6 rubrics
│   └── emonet_taxonomy.md                 ← 40 EmoNet emotions + synonym clusters
├── guides/
│   ├── inference.md                       ← run the models, throughput optimization
│   ├── training.md                        ← how the models were trained
│   └── emonet_empathic_insight.md         ← emotion (EmoNet) prediction recipe
└── docs/                                  ← the live GitHub Pages demo (do not edit)
```

## License & credits

Models and annotations by **LAION** (CC‑BY‑4.0). Source speech from the **Emolia** corpus. The
57‑dimension rubric builds on the **VoiceNet‑Ext** framework (Schumann et al., 2025,
[arXiv:2505.20033](https://arxiv.org/abs/2505.20033)); the full VoiceNet paper is forthcoming.
