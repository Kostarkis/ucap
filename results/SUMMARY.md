# Results summary — per-clip evaluation

Single source for the numbers in §3. All clips processed by **UCAP** (Utility
Carve-out of Anonymisation Protocol) — `notebooks/pipeline_unidepth_colab.ipynb`: SAM 3 @ score
0.6 + UniDepth v2 metric depth, near = 1.1 m, carve policy, conflict overlap 0.2
— on a single GPU (NVIDIA L4 / T4-class). EgoBlur baseline
(`notebooks/egoblur_eval_colab.ipynb`) manually annotated on POV1–3, myIphone1, myIphone2,
and Ropedia.

**Provenance:** every UCAP number below comes from the **matched-prompt runs**
(`<clip>/review_log_dac_face_plate.json`, prompts = "face", "license plate" —
the same runs that were hand-annotated). Earlier 3-prompt-run numbers
(M2 0.455/0.432/0.612/0.449/0.533, POV1 62 flagged, Ropedia 2649 flagged) are
superseded.

> Caveat: manual ground-truth annotation is now done for **UCAP on all six clips**
> (POV1–3, myIphone1, myIphone2, Ropedia) and for the **EgoBlur baseline on POV1–3,
> myIphone1, myIphone2, and Ropedia**. Across **all six clips UCAP had M1 = 0
> silent face leaks** — including the in-domain Ropedia clip. The residual cost is
> over-blur of interaction content (0 on POV1/POV3; 36, 17 on the iPhone takes;
> 178 on POV2; 166 on Ropedia), driven by monocular-depth behaviour near the 1.1 m
> threshold and recall-first false positives, never by a face leak. The
> carved/preserved video is the **maximum-utility visualization**; substantial
> carves are the **conflict frames** counted by M2 and routed to human review.

## Method, per clip

| Clip | Source / lens | Res | Frames | pipe fps | M2 | priv_frac (mean) | near_frac (mean) | depth_valid | Headline |
|---|---|---|---|---|---|---|---|---|---|
| **rhopediaFull** | Ropedia `stereo_right`, **wide-angle, in-domain** | 512² | 5822 | 3.47 | **0.442** (2575) | 0.007 | 0.44 | 1.0 | **In-domain success**: bystander faces (far) auto-redacted, hands+objects (near) preserved (frame 2330). M2 driven by large near zone + small detections. |
| **myIphone1** | iPhone 4K, indoor + bystander | 3840² | 678 | 1.46 | 0.400 (271) | 0.014 | 0.10 | 1.0 | **Demonstrator**: far face auto-redacted, near face preserved+flagged. |
| **myIphone2** | iPhone 4K (2nd take) | 3840² | 676 | 1.48 | 0.317 (214) | 0.012 | 0.14 | 1.0 | **Reproduces** myIphone1; fewer flags, but more of them expose a face (more dwell at arm's reach). |
| **POV1** | stock POV, confrontation | 1920×1080 | 138 | 2.11 | 0.420 (58) | 0.020 | 0.28 | 1.0 | High-conflict: subject at arm's reach. |
| **POV3** | stock POV, person approaching | 1920×1080 | 120 | 1.86 | 0.758 (91) | 0.018 | 0.18 | 1.0 | High-conflict (one 91-frame episode); EgoBlur baseline available. |
| **POV2** | stock GoPro, open water | 1920×1080 | 287 | 2.00 | 0.519 (149) | 0.75 (med **0.98**) | 0.40 | 1.0 | **Limit case**: recall-first false positives on water + monocular depth unstable on low-texture/transition → whole-frame blur, unusable. |

depth_valid = 1.0 on every clip: UniDepth produced valid depth throughout. All
clips are wide/normal FoV — **no true fisheye clip was processed**.

## EgoBlur baseline — manual annotation

Frame-by-frame manual review of EgoBlur output, counting its two silent failure
modes: **face leaks** (a face is present but left unblurred → M1-analog) and
**over-blurs** (non-face content blurred → lost utility / M3-analog). EgoBlur
flags neither — every miss and every false positive is silent.

| Clip | Frames | Face present | Silent leaks (face not covered) | Over-blurs (non-face) |
|---|---|---|---|---|
| **POV1** | 138 | yes | **12** (8.7% of frames) | 1 (0.7%) |
| **POV2** | 287 | none | 0 | **4** (1.4%) |
| **POV3** | 120 | yes | **7** (5.8% of frames) | 2 (1.7%) |
| **myIphone1** | 678 | yes | **17** (2.5% of frames) | **29** (4.3% of frames) |
| **myIphone2** | 676 | yes | **21** (3.1% of frames) | **13** (1.9% of frames) |
| **rhopediaFull** | 5822 | yes (source pre-blurred) | **44** † | not counted |

† **Ropedia caveat:** the Ropedia source had already *attempted* to anonymize the
footage, so a number of faces were already blurred by the publisher. EgoBlur's 44
silent leaks are therefore counted over the faces the source left visible, and
this figure is **not directly comparable** to the other clips (where no source
pre-blurring exists). It is recorded as an in-domain illustration of EgoBlur's
under-redaction, not as a head-to-head number.

Every clip shows EgoBlur both under-redacting and over-redacting. POV2 is the
cleanest leak-side illustration: with **no faces in the clip at all**, EgoBlur
still fired 4 false-positive blurs. myIphone1 is the cleanest over-blur
illustration: **29 false positives** on 4K footage — the highest of any clip.
EgoBlur leaked a face on every face-bearing clip (12, 7, 17, 21, and 44 frames),
**none of them flagged**. (Rates are over all frames; a proper per-face-frame leak
rate needs the manual face-present count per clip — see gaps.)

## EgoBlur baseline vs method — POV3

| | Detection coverage | Manual leaks | Output behaviour | Speed |
|---|---|---|---|---|
| **EgoBlur** | face detected on **102/120 (85%)** | **7/120 (5.8%)** silent leaks + 2 over-blurs, none flagged | blurs every detection | 0.39 fps |
| **Method** | **120/120 (100%)** | **0** silent leaks (annotated); conflicts flagged (M2) | far faces auto-blurred, near preserved+flagged | ~2 fps |

Honest comparison axis = **silent failures vs reviewed conflicts**: EgoBlur
under- and over-redacts simultaneously with no signal; the method routes every
substantial conflict to human review (M2) rather than failing silently.

## Method — manual annotation

Frame-by-frame manual review of the method's output (`anon_dac` + `conflicts`).

### POV1 — clean case

| Metric | POV1 | Note |
|---|---|---|
| **M1** silent face leaks (face out of interaction, missed) | **0** | no privacy-relevant face left unblurred without a flag |
| **M3** over-blur of hands / objects of interaction (lost utility) | **0** | no useful near content destroyed |
| **M2** conflicts | pipeline flagged **58** (0.420); **56** genuine carve conflicts on review | — |
| &nbsp;&nbsp;↳ exposed a **face** when blur dropped | **24 / 56** | genuine human decision → blur the face (sacrifice that utility) |
| &nbsp;&nbsp;↳ exposed only **hands / objects** (no face) | **32 / 56** | could have been auto-preserved **without** human review |

Headline: on POV1 the method had **0 silent leaks and 0 utility-destroying
over-blurs** — every privacy-relevant exposure was caught and flagged, never
silent (contrast EgoBlur POV1: 12 silent leaks + 1 over-blur). The cost is
conservatism: **32 of 56 conflicts (57%) were over-cautious** — the near content
was a hand/object, not a face, so a human reviewer was asked to confirm a
non-issue.

Two future improvements this exposes:
1. **Less eager face detection** would drop many of the 32 needless flags and
   save reviewer time — but it directly raises silent-leak risk (the failure the
   method exists to avoid), so it's the risky lever.
2. **Classify near objects** into *hands/objects of interaction* (auto-preserve)
   vs *face* (auto-blur), treating interaction objects as a labelled subset of
   the near zone. Then most conflicts resolve automatically: the 32 non-face
   carves auto-preserve, the 24 face cases auto-blur — human review only for the
   genuinely ambiguous remainder.

### POV2 — limit case (aquatic / swimming footage)

POV2 has **no faces** (so M1 = 0 trivially); the failure here is utility loss on
the object of interaction (a surfboard). 287 frames, 149 conflicts flagged
(M2 = 0.519):

| Outcome for the surfboard | POV2 | Note |
|---|---|---|
| carve worked correctly (surfboard preserved) | **107** | good |
| **M3** unjustly blurred (lost utility) | **178** | UniDepth never registered it as near (fully or partially) |
| &nbsp;&nbsp;↳ partial conflicts | ~42 | partially carved / partially blurred (149 flagged − 107 preserved) |
| &nbsp;&nbsp;↳ blurred but **never flagged** | ~25 | **silent** over-blur — no review would catch it (annotated estimate) |

Root cause: **aquatic / swimming footage is out-of-domain for monocular metric
depth** — reflective, low-texture water gives no stable scale, so UniDepth
under-detects the near surfboard and it gets blurred (likely compounded by SAM 3
over-eager segmentation). This bounds the method's applicability: it needs depth
the backbone can actually estimate. Unlike POV1, here ~25 over-blurs were *silent*
(unflagged), which is the dangerous mode — the limit case is honest about that.

### POV3 — sustained near-face case

120 frames, 91 conflicts flagged (M2 = 0.758) — essentially one continuous
"person approaching" episode, so the near zone is dominated by a face that
should be blurred rather than by interaction objects:

| Metric | POV3 | Note |
|---|---|---|
| **M1** silent face leaks | **0** | all 80 face exposures were flagged, none silent |
| **M3** false-positive over-blur (e.g. license plates) | **0** | none observed |
| **M2** conflicts | **91** | — |
| &nbsp;&nbsp;↳ carved face revealed identity → decision: **blur** | **80 / 91** | face was **not** an object of interaction |
| &nbsp;&nbsp;↳ no intervention needed | **11 / 91** | over-cautious flag |

Headline: POV3 **inverts POV1's conflict profile**. Here **80/91 (88%)**
conflicts were genuine (a real face exposed by the carve), vs POV1's 24/56 (43%).
Because POV3 is a sustained approach, the near zone is mostly a face that should
be blurred — so the reviewer blurs all 80, and the max-utility carve is the
*visualization*, the reviewed output is the deliverable. Crucially the method
flagged every one — **0 silent leaks** — so POV1 (low genuine-conflict, lots of
over-caution) and POV3 (high genuine-conflict) bracket the spectrum, and on
neither does a face leak silently.

### myIphone1, myIphone2 — in-the-wild 4K demonstrators

Both 4K iPhone takes were UCAP-annotated. SAM 3 detected on 672/678 and
612/676 frames (the gap frames have no face/plate in view — confirmed by 0
silent leaks), and depth was clean throughout.

| Metric | myIphone1 | myIphone2 | Note |
|---|---|---|---|
| frames | 678 | 676 | |
| **M1** silent face leaks | **0** | **0** | every face exposure was redacted or flagged |
| **M3** over-blur of interaction content | **36** | **17** | lost utility; causes below |
| **M2** flagged frames | 271 (0.400) | 214 (0.317) | |
| &nbsp;&nbsp;↳ flagged frames exposing a **face** | **126** | **140** | genuine conflicts; reviewer decision = blur |

Headline: **0 silent leaks on both takes** (consistent with POV1/POV3). The
residual cost is over-blur of interaction content, and its causes are
**diagnostic of the depth stage, not the design**:
1. A **paper sheet behind a waving hand** was segmented by SAM 3 as a "licence
   plate" (recall-first false positive).
2. **Fast hand motion** produced uneven, jittery depth on the moving hand, so the
   near mask flickered and blur spilled onto the hand on some frames.
3. A **TV remote** held at the frame edge, with only its tip visible and no hand
   in shot, was placed too far by UniDepth and let through to blur.

Causes 2–3 are **monocular-depth failures** (jitter on fast motion, mis-scaled
small/edge objects), exactly the regime where stereo or a better depth backbone
would help — these clips are the clearest argument that the depth stage, not the
carve logic, is the limiting factor (§4.3).

### rhopediaFull — in-domain (wide-angle manipulation)

The in-domain hero clip, now UCAP-annotated. 5822 frames, M2 = 0.442 (2575
flagged). This is the strongest RQ1 result: **0 silent face leaks in-domain** —
bystanders are far and cleanly redacted, so no face passed unredacted-and-unflagged.

| Metric | rhopediaFull | Note |
|---|---|---|
| **M1** silent face leaks | **0** | the headline in-domain privacy result |
| **M3** over-blur of object of interaction | **166** | coffee-maker display blurred; cause below |
| flagged frames exposing a **face** | **2** | genuine conflicts (face carved into the near zone by depth); reviewer decision = blur. Not silent (flagged), so not a leak |

Over-blur cause: the operator interacts with a coffee maker whose **display sits
right at the 1.1 m near threshold**. On some frames SAM 3 reads the display as a
"licence plate" (recall-first false positive), and because the display hovers at
the depth boundary the near mask does **not** carve it out, so it is blurred. This
is the same depth-at-threshold mechanism as the POV2 surfboard and the iPhone
remote: a **monocular-depth boundary effect compounded by a detector false
positive**, not a carve-logic flaw (§4.3). The 2 carved-face frames are the
in-domain analogue of the POV3 conflicts: a near face the depth gate preserved but
the pipeline **flagged** for review (decision: blur), so M1 stays 0.

For RQ1, the head-to-head with EgoBlur on this clip is **UCAP 0 vs EgoBlur 44**
silent face leaks — but EgoBlur's 44 is measured over a source that had **itself
attempted anonymisation** (some faces already publisher-blurred), so it is
recorded as an in-domain illustration, not a clean matched number.

## Gaps before submission
- **Face-present frame counts** for the face-bearing clips → convert the silent-leak counts into a proper per-face-frame M1 rate (the one remaining "counts, not rates" caveat).
- A **true fisheye** clip through both — needed if the fisheye claim is kept; else move to future work.
- Done: UCAP annotation on **all six clips** (POV1–3 + myIphone1 + myIphone2 + Ropedia — M1 = 0 silent face leaks on every one); EgoBlur baseline on POV1–3 + myIphone1 + myIphone2 + Ropedia.
