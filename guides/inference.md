# VoiceNet Inference Guide

How to run the **VoiceNet dimension predictors** to get, from a single audio clip:

- all **57 perceptual voice/prosody dimension scores** (0–6 ordinal; `EXPL` is 0–2), each as a
  **regression** score *and* a **classification** bucket + confidence,
- a **Genuineness** score (0–6, how authentic vs. performed the delivery sounds), and
- a **Vocal‑Burst Blend** score (0–10, how naturally a laugh/sob/gasp/sigh blends into speech).

Everything is computed from **one 768‑d VoiceCLAP‑commercial embedding per clip**. You encode the
audio **once**; all 116 heads then read that same vector. This is what makes VoiceNet cheap to run
at scale.

- Model repo: **[laion/voicenet-dimension-predictors-commercial](https://huggingface.co/laion/voicenet-dimension-predictors-commercial)**
- Taxonomy: **[`../taxonomy/voicenet_taxonomy.md`](../taxonomy/voicenet_taxonomy.md)** (all 57 dims × 0–6 rubrics)
- Live demo: **<https://projects.laion.ai/voicenet/>**

---

## 1. What you get and how it is computed

```
audio ──► VoiceCLAP-commercial encoder ──► 768-d L2-normalized embedding (ONE per clip)
                                              │
                 ┌────────────────────────────┼───────────────────────────────┐
                 ▼                            ▼                                ▼
        57 regression heads          57 classification heads         genuineness (0–6)
        (continuous 0–k score)       (k-way bucket + softmax conf)   + vocal-burst-blend (0–10)
```

Every head is a tiny MLP: `Linear(768, H) → GELU → Dropout(p) → Linear(H, out)`
(regression `out=1`, classification `out=k`). Inputs are standardized per‑dimension with the
head's stored train mean/std. The 57 regression + 57 classification heads are the
**best‑variant‑per‑dimension** picks (the model repo also ships all `variants/cap3x|cap4x|cap5x`
checkpoints). The **genuineness** and **vocal‑burst‑blend** heads share the same MLP shape and run
on the **same 768‑d embedding**.

> **One encode → all heads.** The 768‑d embedding is the only expensive part. Running all 116
> heads on a 512‑clip batch of embeddings takes ~**6.2 ms** (~82,000 clips/s) — negligible next to
> the encode. Never re‑encode a clip per dimension.

---

## 2. Install & load the model

The model repo is **self‑contained**: it bundles the full VoiceCLAP‑commercial embedder under
`voiceclap_commercial/`, plus `predict.py`, `dimensions.py`, the `regression/` and
`classification/` heads, and the `variants/`. No separate encoder download is required.

```bash
pip install torch torchaudio transformers huggingface_hub numpy
# grab the self-contained model repo (embedder + all heads + predict.py)
huggingface-cli download laion/voicenet-dimension-predictors-commercial \
    --local-dir voicenet-commercial
cd voicenet-commercial
```

### Single file (shell)

```bash
python predict.py path/to/audio.wav
```

Prints one line per dimension: code, name, regression score, classification bucket, softmax
confidence, and the plain‑language level label.

### Single file (Python)

```python
from predict import VoiceNetPredictor

p = VoiceNetPredictor()                 # loads bundled embedder + best-per-dim heads (114)
out = p.predict("audio.wav")            # ONE encode -> 57 dims

print(out["AROU"])
# {'name': 'Arousal', 'reg_score': 3.1, 'cls_bucket': 3, 'confidence': 0.62,
#  'bucket_label': 'The audio captures the standard, baseline physiological ...', 'n_levels': 7}
```

To force one balancing variant instead of the best‑per‑dim defaults:

```python
p4 = VoiceNetPredictor(variant="cap4x")     # or from shell: python predict.py audio.wav --variant cap4x
```

### Adding the genuineness and vocal‑burst‑blend heads

These two extra heads consume the **same** 768‑d embedding. Load them once and apply them to the
embedding you already computed:

```python
import torch, torch.nn as nn

class MLPHead(nn.Module):
    def __init__(self, D, H, p, out):
        super().__init__()
        self.f1, self.act, self.dp, self.f2 = nn.Linear(D, H), nn.GELU(), nn.Dropout(p), nn.Linear(H, out)
    def forward(self, x):
        return self.f2(self.dp(self.act(self.f1(x))))

def load_extra_head(path, dev):
    ck = torch.load(path, map_location=dev, weights_only=False)
    a = ck["arch"]; net = MLPHead(a["D"], a["H"], a["p"], a["out"]).to(dev).eval()
    net.load_state_dict(ck["state_dict"])
    ck["net"], ck["mu"], ck["sd"] = net, torch.as_tensor(ck["mu"], device=dev), torch.as_tensor(ck["sd"], device=dev)
    return ck

genu  = load_extra_head("genu_commercial_best.pt",  p.dev)   # 0–6 genuineness
blend = load_extra_head("blend_head_commercial.pt", p.dev)   # 0–10 vocal-burst-blend

emb = p.embed("audio.wav")                                   # the SAME 768-d vector
for name, ck in [("genuineness", genu), ("vocal_burst_blend", blend)]:
    x = ((emb - ck["mu"]) / ck["sd"]).unsqueeze(0)
    print(name, round(float(ck["net"](x).squeeze().cpu()), 3))
```

---

## 3. Loading audio

VoiceCLAP‑commercial expects **mono, 16 kHz, a 30‑second window** (480,000 samples; shorter clips
are zero‑padded, longer clips truncated). `predict.py` already does this via `torchaudio`.

### (a) Individual mp3 / wav / flac files

```python
import torchaudio, torch
TARGET = 480000  # 30 s @ 16 kHz

def load_clip(path):
    w, sr = torchaudio.load(path)                 # decodes wav/mp3/flac/ogg/m4a
    if w.dim() == 2: w = w.mean(0)                # to mono
    if sr != 16000: w = torchaudio.functional.resample(w, sr, 16000)
    T = w.shape[0]
    return w[:TARGET] if T >= TARGET else torch.nn.functional.pad(w, (0, TARGET - T))
```

Feed a **batch** of these waveforms (stacked to `[B, 480000]`) to `embedder.encode_waveform(...)`
to get `[B, 768]` in one call — see §4.

### (b) WebDataset shards (recommended at scale)

Store audio as **WebDataset** tar shards and stream them sequentially. This turns millions of tiny
random reads into a handful of large sequential streams — the single most important thing for
throughput (see §5).

```python
import io, torch, torchaudio, webdataset as wds

TARGET = 480000
def decode_audio(sample):
    for ext in ("flac", "wav", "mp3", "ogg"):
        if ext in sample:
            w, sr = torchaudio.load(io.BytesIO(sample[ext]))
            if w.dim() == 2: w = w.mean(0)
            if sr != 16000: w = torchaudio.functional.resample(w, sr, 16000)
            T = w.shape[0]
            w = w[:TARGET] if T >= TARGET else torch.nn.functional.pad(w, (0, TARGET - T))
            return {"__key__": sample["__key__"], "wav": w}
    return None

ds = (wds.WebDataset("shards/train-{000000..000999}.tar", shardshuffle=False)
        .map(decode_audio).select(lambda s: s is not None)
        .to_tuple("__key__", "wav").batched(32))

loader = torch.utils.data.DataLoader(
    ds, batch_size=None, num_workers=12, pin_memory=True, prefetch_factor=4)

for keys, wavs in loader:
    emb = embedder.encode_waveform(wavs.to(device))    # [B, 768]
    # ... run all heads on emb (see §4) ...
```

The companion annotation dataset
[laion/emolia-voicenet-gemini-annotations](https://huggingface.co/datasets/laion/emolia-voicenet-gemini-annotations)
stores its audio exactly this way (flac inside HF WebDataset tars, avg ~360 KB/clip).

---

## 4. Batched inference: embed once, run every head

```python
import torch, glob, os, numpy as np, torch.nn as nn
from transformers import AutoModel

device = "cuda"
embedder = AutoModel.from_pretrained("voicenet-commercial/voiceclap_commercial",
                                     trust_remote_code=True).to(device).eval()

# load all heads once (reuse predict.py's MLPHead / _load_head, or the snippet in §2)
from predict import _load_head
reg = {os.path.basename(p)[:-3]: _load_head(p, device) for p in glob.glob("voicenet-commercial/regression/*.pt")}
cls = {os.path.basename(p)[:-3]: _load_head(p, device) for p in glob.glob("voicenet-commercial/classification/*.pt")}

@torch.no_grad()
def score_batch(wavs):                              # wavs: [B, 480000] on device
    emb = embedder.encode_waveform(wavs)            # [B, 768]  <-- the ONLY heavy step
    res = {}
    for dim, r in reg.items():
        x = (emb - r["mu"]) / r["sd"]
        res[dim] = r["net"](x).squeeze(-1).cpu().numpy()          # [B] regression
    for dim, c in cls.items():
        x = (emb - c["mu"]) / c["sd"]
        prob = torch.softmax(c["net"](x), -1)
        res[dim + "_bucket"] = prob.argmax(-1).cpu().numpy()      # [B] class bucket
        res[dim + "_conf"]   = prob.max(-1).values.cpu().numpy()  # [B] confidence
    return res
```

Because the heads are ~20–50k params each and vectorize over the batch, the 116‑head sweep is
essentially free; keep batches ≥ 16 so the **encoder** stays saturated.

---

## 5. Throughput optimization guide (measured)

All figures below are **measured** on VoiceCLAP‑commercial encoding **30‑second clips on one
80 GB GPU** (FP32/FP16 encode; heads in the same precision).

### Batch size — the encoder is compute‑bound

| batch size | throughput (clips/s) | peak VRAM |
|--:|--:|--:|
| 1  | 37 | — |
| 8  | 52 | — |
| 16 | 53 | 1.3 GB |
| 32 | 54 | 2.2 GB |
| 64 | 54 | 3.9 GB |
| 128 | 54 | 7.3 GB |

Throughput **saturates at ~54 clips/s for batch ≥ 16** (≈ **1600× real‑time** for 30 s clips).
Bigger batches buy almost nothing in speed but cost linear VRAM, so **batch 16–32 is the sweet
spot**: full throughput at low VRAM. The workload is **compute‑bound**, not memory‑bound.

### The heads are negligible

All 116 heads (57 regression + 57 classification + genuineness + blend) on a **512‑clip** batch of
embeddings run in **~6.2 ms → ~82,000 clips/s**. The 768‑d embedding is computed **once** and feeds
all 116 heads, so the head cost never matters — plan around the encoder.

### Per‑GPU and multi‑GPU

- **Per GPU (fed):** ~54 clips/s ≈ **~194,000 clips/hour**.
- **6 GPUs (data‑parallel):** ~324 clips/s ≈ **~1.16M clips/hour**, near‑linear — the heads add
  nothing and each GPU runs an independent shard stream.

### The real bottleneck is audio I/O, not the model

At scale the GPU starves on **audio decode + resample + network**, not compute. In large embedding
runs, streaming flac from **HF WebDataset tars over the network** ran only **~5–15 clips/s per
worker** (~7 MB/s per stream, network‑bound); local NVMe decode is far faster. flac clips average
~360 KB. To keep a ~54 clips/s GPU fed:

- Use `DataLoader(num_workers=8–16, pin_memory=True, prefetch_factor=…)` so decoding overlaps compute.
- Read from **WebDataset shards** (sequential reads, many parallel shard streams) rather than
  millions of tiny random files.
- Prefer **local NVMe** over network streaming when possible; otherwise open many shard streams in
  parallel to hide per‑stream latency.
- Keep batches at **16–32**: enough to saturate the encoder, small enough that a worker pool can
  refill them in time.

**Rule of thumb:** provision I/O so each GPU receives **≥ 54 decoded, resampled clips/s**. Below
that you are I/O‑bound; above it you are wasting bandwidth.

---

## 6. Interpreting the outputs

- **Regression score** — smooth estimate on the dimension's 0–k scale (k=6 for most; `EXPL` is 0–2;
  `BKGN` is 0–4). Best for ranking/filtering.
- **Classification bucket + confidence** — a hard 0–k level and its softmax probability; the
  `bucket_label` is the verbatim level description from the taxonomy.
- **Genuineness** 0–6, **Vocal‑Burst Blend** 0–10.

Predictions are **perceptual estimates** distilled from a single flash‑LLM annotator
(Gemini‑3.5‑flash), not ground truth. Coarse perceptual axes (arousal, valence, gender, formality,
tempo, brightness, background noise) predict strongly; fine resonance/timbre axes (`R_MIXD`,
`R_NASL`, `R_MASK`, `METL`) and shift/flux axes (`ARSH`, `VALS`, `VFLX`) are collapsed in the labels
and predict near‑constant. Best‑per‑dimension mean metrics on the frozen validation split:
**reg‑MAE 0.744, Pearson r 0.791, within‑1 0.870, classification accuracy 0.617**. See the model
repo for the full per‑dimension table, and [`training.md`](training.md) for how the heads were
trained.

---

## 7. Also want the emotion (EmoNet) scores?

VoiceNet tells you **how** speech is delivered. To also get **which emotion** is expressed — the
**40 EmoNet emotions + Valence + Arousal + audio‑quality scores** — run the complementary
**Empathic‑Insight‑Voice‑Plus** model. See
[`emonet_empathic_insight.md`](emonet_empathic_insight.md).
