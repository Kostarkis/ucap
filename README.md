# UCAP — Utility Carve-out of Anonymisation Protocol

Depth-gated anonymization for egocentric RGB video: redact bystanders' privacy,
**preserve the manipulation signal** (the operator's hands and the objects they
act on), and never resolve a privacy/utility conflict silently.

Code release for the diploma project *"Reconciling Privacy and Utility in
Egocentric RGB Video: A Depth-Gated Anonymization Pipeline for Embodied-AI
Training Data"* (Bartosz Kostarczyk, Kozminski University, 2026).

## How it works

For every frame, UCAP:

1. **Detects** privacy concepts with **SAM 3** (promptable segmentation), run
   *recall-first* at a low score threshold — a missed face leaks permanently,
   while a false positive is merely a region the next stage can recover.
2. **Estimates a near-field mask** from depth — **UniDepth v2** (monocular,
   metric, self-calibrating) or **OpenCV SGBM** (classical stereo) — and
   thresholds it at arm's reach (1.1 m). This is the interaction zone to preserve.
3. **Redacts detections MINUS the near field** (irreversible solid fill by
   default), so background PII disappears while the interaction zone stays sharp.
4. **Flags conflicts**: any privacy region overlapping the near field by ≥ 0.2
   is logged and the frame is routed to **human review** — a 3-way decision
   (drop the blur / keep the blur / mark the frame unusable).

The redacted video the pipeline writes is the **maximum-utility visualisation**,
not the shareable deliverable: flagged frames must pass review first. Privacy is
therefore judged by **silent** failures only.

**Metrics** (all lower-better): **M1** face-leak (a present face neither redacted
nor flagged — silent privacy failure), **M2** conflict/human-review rate (share
of frames flagged, in % — the review workload, a cost rather than damage),
**M3** lost-training-value (interaction content redacted without a flag — silent
utility failure). Implemented in `metrics.py`.

## Headline result

Hand-annotated, frame-by-frame, on six egocentric clips (in-domain wide-angle
manipulation, in-the-wild 4K, and deliberate stressors), with SAM 3 prompted for
the same two categories as the EgoBlur baseline ("face", "license plate"):

| | EgoBlur (specialist baseline) | UCAP |
|---|---|---|
| Silent face leaks, five matched clips | **57** | **0** |
| Silent face leaks, in-domain clip | 44 (source partly pre-blurred) | **0** |
| Failure visibility | none (no flagging mechanism) | every conflict flagged for review |

UCAP's residual cost is over-blur driven by monocular depth at the 1.1 m
boundary (never a face) and the M2 review load (31.7–75.8 % of frames,
clip-dependent). Full numbers: `results/SUMMARY.md` and the per-clip JSON logs.

## Repository layout

```
pipeline.py             the shared engine: detect -> protect -> resolve -> redact
metrics.py              M1 / M2 / M3 + IoU, AP/mAP, recall@precision, FPS, VRAM
temporal.py             per-track temporal recall (the complement of M1)
video_io.py             MP4 read/write + stereo-pair iterator
backends.py             detector adapters (EgoBlur baseline, etc.)
bench_harness.py        per-model benchmark runner
run_benchmark_colab.py  benchmark entry point for Colab
env_check.py            environment / GPU sanity check
requirements.txt        dependencies (CPU core + Colab inference stack)
tests/                  synthetic-data unit tests — no GPU, no real footage
notebooks/
  pipeline_unidepth_colab.ipynb   end-to-end UCAP, SAM 3 + UniDepth v2 (monocular)
  pipeline_stereo_colab.ipynb     end-to-end UCAP, SAM 3 + OpenCV SGBM (stereo pair)
  egoblur_eval_colab.ipynb        EgoBlur-only pass over a clip (baseline comparison)
  egoblur_parallel_colab.ipynb    parallelised EgoBlur anonymizer (baseline at speed)
results/
  SUMMARY.md                      per-clip results backing the paper's tables
  <clip>/review_log_dac_face_plate.json   UCAP machine log (config, stats, flagged events)
  <clip>/egoblur_eval.json                EgoBlur baseline stats for the same clip
```

*Naming note:* the `dac` in the log filenames is historical — it refers to Depth
Any Camera, an earlier depth-backend candidate that was excluded (it pins an old
PyTorch and needs supplied intrinsics). The shipped monocular backend is
**UniDepth v2**; `_face_plate` marks the released two-prompt configuration.

## Quick start

CPU-only (no models, no footage — verifies the engine and the metrics):

```bash
pip install numpy opencv-python-headless
python env_check.py
python tests/run_all.py        # full synthetic test suite, ~seconds
```

To anonymize a clip, open `notebooks/pipeline_unidepth_colab.ipynb` (monocular)
or `notebooks/pipeline_stereo_colab.ipynb` (stereo pair) in Google Colab on a
single T4/L4 GPU; each notebook installs its own dependencies, downloads the
models, and walks through configuration, processing, the conflict-review video,
and the metrics. Note: the SAM 3 checkpoint (`facebook/sam3`) is gated on
Hugging Face — request access and log in with a HF token; EgoBlur weights come
from the official Meta release.

## Released configuration

| Parameter | Value |
|---|---|
| SAM 3 prompts | `"face"`, `"license plate"` (EgoBlur-matched; extendable by prompt, e.g. `"document"`) |
| Detection score threshold | 0.6 (recall-first) |
| Near-zone threshold | 1.1 m metric (UniDepth v2) or disparity quantile (SGBM) |
| Conflict overlap threshold | 0.2 |
| Redaction | irreversible solid fill (mask-aware Gaussian blur configurable) |
| Runtime | single NVIDIA L4/T4, batch size 1, < 5.4 GB peak VRAM |

## Annotation protocol (evaluation labels)

All evaluation labels are manual, by a single annotator, frame by frame on the
rendered outputs, all six clips in full (no sampling). Per clip: **silent face
leaks** (a face neither redacted nor flagged), **over-blur of interaction
content** (hands or manipulated objects redacted), and a split of flagged frames
into those where dropping the blur would expose a face versus those exposing
only hands/objects. Flagged-and-preserved content is a review decision, never
counted as a leak.

## Privacy rules for this repository

No identifying data lives here: tests use synthetic frames, the JSON logs carry
numbers only (no pixels), and `.gitignore` blocks all image and video formats.
Do **not** commit frames, clips, or model dumps containing faces, documents, or
licence plates.

## Third-party models and licences

SAM 3 (Meta), UniDepth v2 (Piccinelli et al.), and EgoBlur (Meta / Project Aria)
are **not** redistributed here; the notebooks download them at runtime from
their official sources, and they remain under their respective licences. The
code in this repository is released under the MIT License (see `LICENSE`).

## Citation

See `CITATION.cff`, or cite: Kostarczyk, B. (2026). *Reconciling Privacy and
Utility in Egocentric RGB Video: A Depth-Gated Anonymization Pipeline for
Embodied-AI Training Data.* Diploma project, Kozminski University, Warsaw.
