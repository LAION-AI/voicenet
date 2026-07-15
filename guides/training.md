# VoiceNet Training Guide

How the **VoiceNet dimension predictors** were built: the labels, the candidate selection, the
embedding, the head architecture, the balancing, and what turned out easy vs. hard. This documents
the released model
[laion/voicenet-dimension-predictors-commercial](https://huggingface.co/laion/voicenet-dimension-predictors-commercial)
and its label set
[laion/emolia-voicenet-gemini-annotations](https://huggingface.co/datasets/laion/emolia-voicenet-gemini-annotations).

---

## 1. Overview

VoiceNet predicts **57 perceptual voice/prosody dimensions** (0–6 ordinal; `EXPL` 0–2) from a
single frozen audio embedding. The design philosophy is deliberately lightweight:

```
frozen VoiceCLAP-commercial embedding (768-d, L2-normalized, 30 s window)
        └──► one small MLP head per dimension  (Linear→GELU→Dropout→Linear)
```

The encoder is **never fine‑tuned**. All learning happens in the tiny per‑dimension heads
(~20–50k params each). This keeps training cheap, keeps every dimension independent, and means a
clip is embedded **once** and re‑used by every head at inference time.

---

## 2. Labels — Gemini‑3.5‑flash, three rounds

- **Annotator:** **Gemini‑3.5‑flash, non‑thinking** (`thinkingBudget = 0`), **temperature 0**.
  Each judgement is a single‑token integer, so labels are direct perceptual reads, not
  chain‑of‑thought.
- **Task:** the model hears one clip and emits one integer for the **one** dimension the clip was
  selected for, following that dimension's verbatim 0–6 level rubric from the
  [VoiceNet taxonomy](../taxonomy/voicenet_taxonomy.md).
- **Scale:** 0–6 integer for all dimensions **except `EXPL`** (content appropriateness), which is
  0–2.
- **Volume:** **468,180 annotations** across the 57 dimensions, over **three annotation rounds**.
- **Cost:** **~$265 total** (gemini‑3.5‑flash batch API, 50% batch discount).

The three rounds progressively fixed data starvation:

1. **Round 1** — the initial pass over CLAP‑bucketed candidates across all 57 dimensions.
2. **Round 2** — topped up the most under‑represented `(dimension × bucket)` cells.
3. **Round 3** — a final targeted top‑up of the **worst‑performing / most data‑starved** dimensions:
   the hardest dimensions' **starved score levels** were refilled using **agreement‑only**
   candidates.

Two dimensions were **intentionally excluded** from the round‑3 top‑up: **`R_MIXD`** (Mixed
Resonance) is **CLAP‑blind** — the embedders cannot resolve it (Pearson ~0.4), so no reliable
candidates exist — and **`ROUG`** (Roughness) because its candidate pool was already exhausted.

---

## 3. Candidate selection — dual VoiceCLAP zero‑shot bucketing

Sampling the corpus prior would give almost no examples at the extreme score levels. Instead,
candidates for each dimension were chosen by **VoiceCLAP zero‑shot bucketing**: score each clip by
CLAP similarity against the dimension's level descriptions, then spread candidates across the 0–6
range.

Two CLAP models were used together:

- **VoiceCLAP‑commercial** (768‑d)
- **VoiceCLAP large‑v2** (3584‑d)

The reliable candidates are the ones where the **two models agree** (the commercial bucket == the
large bucket). This **agreement rule** filters out clips the embedders are unsure about. Round 3
used **agreement‑only** candidates to top up starved levels.

This selection strategy is powerful for macro‑acoustic axes but has a built‑in limitation: for
**fine resonance** axes, CLAP zero‑shot bucketing **collapses** — Gemini then rates ~90–99% of those
candidates in a single level, so those buckets carry little balanced signal (see §6). This is a
**CLAP separability + data** limit, not an annotation‑budget limit.

---

## 4. Embedding

Each unique clip is encoded **once** with **VoiceCLAP‑commercial** into a **768‑d, L2‑normalized**
vector over a **30‑second window** (mono, 16 kHz). That single vector is the input to every head,
for both training and inference.

---

## 5. Head architecture

Both head types share the same shape:

```
Linear(768, H) → GELU → Dropout(p) → Linear(H, out)
```

| head | out | H | dropout | loss |
|---|---|---|---|---|
| **Regression** | 1 | 64 | 0.33 | Huber (δ = 1.5) |
| **Classification** | k (levels) | 48 | 0.33 | Cross‑entropy |

- Inputs are **standardized per dimension** using the train mean/std (stored in each checkpoint).
- **Adam, 50 epochs**, **best‑val checkpoint** kept (lowest val MAE for regression; highest val
  accuracy for classification).
- ~**20–50k parameters per head**.

The `H`/`p` choices come from a small architecture exploration (`H ∈ {32,48,64}`, `p ∈
{0.2,0.33,0.5}`) on three probe dimensions spanning difficulty (`AROU` easy, `AGEV` medium,
`R_NASL` hard/collapsed).

---

## 6. Balancing — cap each bucket, pick the best variant per dimension

Even after CLAP bucketing, score distributions are skewed. Training balances them:

- Bucket the training clips by Gemini score.
- Set `max_per_bucket = N × S`, where **S = the count of the second‑smallest usable bucket**, and
  randomly subsample larger buckets to that cap. Near‑empty buckets (< 4 clips) are skipped.
- Three cap variants were explored: **cap3x / cap4x / cap5x**.

A **frozen, identical validation split** was held out **before** capping and kept **the same across
all rounds and all variants**, so every metric is **directly comparable** across the whole project.

**Best‑variant‑per‑dimension selection:** for each dimension independently, keep the variant that
did best on the frozen val set — the **regression** head with the **lowest val MAE** and the
**classification** head with the **highest val accuracy** (chosen separately). Those become the
top‑level `regression/` + `classification/` defaults; all `cap3x/cap4x/cap5x` checkpoints are also
shipped under `variants/` for reproducibility.

Variant means (over all 57 dims): cap3x MAE 0.767 / cap4x 0.762 / cap5x 0.756; mixing the
best‑per‑dimension picks beats any single variant.

---

## 7. Results — what's easy, what's hard

Best‑variant‑per‑dimension means on the frozen validation split:

| metric | value |
|---|--:|
| regression MAE | **0.744** |
| regression Pearson r | **0.791** |
| regression within‑±1 | **0.870** |
| classification accuracy | **0.617** |

Adding round‑3 data improved the defaults on the identical frozen split: mean reg MAE 0.768 → 0.744
and mean cls accuracy 0.593 → 0.617, with the biggest MAE gains exactly on the hard resonance axes it
targeted (`R_MIXD`, `R_NASL`, `R_THRT`, `COGL`, `S_CART`).

- **Strong** (predicts well): macro‑acoustic and demographic axes — brightness, volatility,
  arousal, formality, chest resonance, background noise, tempo, gender, pitch range, smoothness.
- **Weak** (near‑constant): **fine resonance** (`R_MIXD` mixed, `R_NASL` nasal, `R_MASK` mask,
  `METL` metallic), **shift/flux** axes (`ARSH` arousal‑shift, `VALS` valence‑shift, `VFLX`
  velocity‑flux), and **rare style** axes (`S_ASMR`, `S_CART`, `S_TECH`, `S_RANT`).

The weakness is **not** an annotation‑budget problem — it is a limit of **CLAP separability + the
resulting collapsed labels**. Where the two CLAP embedders cannot separate a dimension, Gemini
labels pile into one bucket and there is little balanced signal to learn.

The **Genuineness** (0–6) and **Vocal‑Burst Blend** (0–10) heads share the same MLP structure and
run on the same 768‑d embedding.

---

## 8. Reproduce / extend

- **Labels + provenance:** [laion/emolia-voicenet-gemini-annotations](https://huggingface.co/datasets/laion/emolia-voicenet-gemini-annotations)
  (annotations, raw model text, audio manifest, per‑dimension balance stats, taxonomy).
- **Trained heads + embedder:** [laion/voicenet-dimension-predictors-commercial](https://huggingface.co/laion/voicenet-dimension-predictors-commercial)
  (57+57 best‑per‑dim heads, all `variants/`, bundled embedder, `metrics.parquet`).
- **Run them:** see [`inference.md`](inference.md).
