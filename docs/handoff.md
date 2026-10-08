# Handoff: status, decisions, research, next steps

Last updated 2026-10-08 by Will's local session: the measured RunPod run is done, the pod is
terminated, and a first demo video exists. The first version was built by a cloud session.

## Status at a glance

| Piece | State |
|---|---|
| Python pipeline (`src/polehealth`) | Built. 91 tests pass, ruff clean. Run end to end against the real model on an A100. |
| Data sources | All six fetch and work: PD-Defect lean, crossarm, vegetation; Kaggle wrc50 and wrc45 (no login needed); OSM Toowoomba poles (extract committed). |
| Register | 1,191 Toowoomba poles, deterministic (seed 7). Rebuild with `polehealth build-register`. |
| Demo page (`demo/`) | Live inspection (hero view, Explain / Real speed / Ramp pacing, `?autoplay=1&mode=ramp`), Map, Flip, Engineer queue and Accuracy views, plus "Try a photo" when served by `polehealth serve`. Checked on real data in Chromium at 1920x1080. |
| Local server (`polehealth serve`) | Built and tested with the mock backend. Not yet used against the real model. |
| GPU runbook (`scripts/gpu/`) | Proven on RunPod, 2026-10-08: setup took about 10 minutes, model ready 90 s after start. |
| Real model output | Recorded run `runs/20261008T120934Z-http` (A100-SXM4-80GB, RunPod Secure Cloud, USD 1.59/hr): lean 82.9% (n 474), crossarm 68.1% (n 257), vegetation 70.2% (n 457), record health 69.0% (n 500); median model time 547 ms per lean photo pass, mean 1,096 ms per demo pole (photo read + decision); USD 0.48 per 1,000 poles. `demo/data/` holds 400 real decisions (130 to the engineer queue). Both are gitignored or uncommitted, see next steps. |
| Spend | One pod, 42 minutes at USD 1.59/hr, about USD 1.11 (AUD 1.70) plus disk. Terminated; no pods left. |
| Demo video | `runs/20261008T120934Z-http/pole-health-demo.mp4` (78 s, 1920x1080), recorded headless from the real data with a scripted Chromium screencast. Not committed. |
| README | Drafted. Results section still empty. Not yet brand-reviewed. |

## Decisions already made (do not re-ask)

- **Brand:** Beyond Data (company), not Will's personal byline. Positioning hooks: "fix one broken
  operational decision"; open weights mean "your data never leaves your tenant".
- **Model:** Maincode's Matilda Jev (`Maincode/matilda-jev-v1`), because it takes images and JSON in
  one request, is open (Apache-2.0) and Australian. Not TypeSafe's Jev (text only, no image input).
- **Use case:** utility pole asset health. v1 = photo to findings and health score. v2 = photo +
  findings + asset record + plain-English policy to replace / maintain / defer / reinspect.
- **GPU:** RunPod, on-demand A100 80GB or H100 80GB (or RTX PRO 6000 96GB), per-second billing.
  BinaryLane was ruled out: it does not offer GPUs at all (their support FAQ says so, by choice).
- **Budget:** under AUD 20 total, including RunPod's minimum credit top-up (about USD 10; confirm).
- **Repo:** `will-beyond-data/jev-utilities-pole-health`, private until Will approves going public.
- **UI:** no MOCK banner (Will's call). Mock data is for development only and must never be filmed.
- **Unusable photos are excluded (Will, 2026-10-08):** PD-Defect's lean `Rejected` folder (blurred,
  blocked, pole too far away) is not used in the benchmark or demo. `cannot_assess` stays as an answer
  option so a bad uploaded photo routes to an engineer.
- **Video:** opens on the Live inspection view: real photo, an abstract processing sweep with a
  millisecond timer, findings bars, record card, decision stamp with measured times, mini-map dot.
  "Ramp" mode: three poles slowed down with the real ms shown, then real speed.

## Next steps, in order

1. Done 2026-10-08: local setup, first Space answers (the free ZeroGPU quota ran out after 8 calls),
   measured RunPod run, demo data, first demo video. Notes from that run:
   - `torch==2.14.0` has no cu128 build, so `setup.sh` now defaults to cu126.
   - The record question now uses the wrc50 inspection scale (5 = minor defects); with the shared
     photo scale ("as new" at the top) every answer landed one level low.
   - To re-run: same runbook. The weights download took about 2 minutes on RunPod.
2. **Will reviews the video** and the demo on real data: `uv run python -m http.server 8000 -d demo`,
   then open `http://127.0.0.1:8000/?autoplay=1&mode=ramp`.
3. **Fill the README results** from `runs/<dir>/eval_summary.json` only, commit `demo/data/` (keep
   the photo subset small; at max edge 960 each photo is about 130 KB), then brand-review README
   and the demo copy with `beyond-data-brand-voice`.
4. **Before going public:** resolve the Kaggle licence question (below), get Will's approval, flip
   the repo to public.
5. **Content:** carousel and LinkedIn brief go through Will's content-engine repo (Beyond Data
   brand), drafts only. Will posts.

## Open questions and risks

- **NVIDIA is untested by Maincode.** The runtime needs `flash-linear-attention` (Triton based,
  should work on CUDA) and pins transformers 5.17.0 and torch 2.14.0. The config file mentions a
  96 GB RTX PRO 6000, which suggests NVIDIA was at least considered.
- **The prompts have never met the real model.** Step 3 exists for this.
- **Kaggle licence is not stated** on the utilityanalytics datasets. Raw workbooks are never
  committed and `data/register.json` is gitignored, but published `demo/data/poles.json` will carry
  derived records. Options: ask the uploader, or accept the risk for a non-commercial demo. Will's call.
- **Photos are Indian concrete poles; Toowoomba is mostly timber.** Stated everywhere. Lean,
  crossarm and vegetation look the same on any pole; rot and woodpecker damage are not covered by
  labelled photos.
- **Maincode's own claims** (Decision Index ahead of TypeSafe Jev on 27 of 43 benchmarks; median
  56.8 ms on MI355X) are self-reported and awaiting independent leaderboard verification. Quote
  them as Maincode's claims only.

## Research notes (so nobody repeats it)

**Matilda Jev** (`huggingface.co/Maincode/matilda-jev-v1`, created 2026-09-30, updated 2026-10-07):
26.1B params, fine-tuned from Qwen/Qwen3.8-27B, BF16, about 49 to 54 GB of weights, Apache-2.0.
Decision model with a 255-option readout and a stored calibration temperature; no text generation.
Official runtime is in the model repo under `runtime/maincode_jev_serve` and serves
`POST /v1/systemone` (request and response shapes are in `docs/spec.md`). Images: up to 4 per
request, base64 data URLs (png, jpeg, webp), 8 MB and 16 MP limits. Question types: choice (1 to
255 options), noul (yes/no), score (2 to 10 ordered levels). Tested by Maincode on AMD MI355X with
Python 3.12, transformers 5.17.0, torch 2.14.0. Context limits auto-size to GPU memory: on a RunPod
A100-SXM4-80GB the server reported `max_context_tokens` 8192 (2026-10-08).

**Matilda Jev Space** (`hugging-apps/matilda-jev`, Gradio on ZeroGPU): endpoint
`/decide_request` takes the exact `/v1/systemone` JSON as a string and returns the same response
plus `latency_s`. Needs an HF token for quota.

**TypeSafe's Jev** (the original, launched 2026-09-15 by TypeSafe AI, San Francisco): text only, no
image input per its docs. About USD 0.042 per million input tokens, output free. The waitlist was
dropped around 2026-09-20 for direct signup at console.typesafe.ai, then new signups were paused on
2026-09-22. Also listed on OpenRouter as `typesafe/jev-latest`. Only useful here for a
"Matilda vs TypeSafe" text-only comparison, which is out of scope for v1.

**Other "Jev" variants seen:** Autoloops' Jevified Gemma-4-31B (hosted, claims under 200 ms with
images) and the `jevify` open-source package; autotrust JEV-27B-VL. Not used.

**Datasets considered but not used (yet):**
- OHL-UK (University of Strathclyde): 4,570 Google Street View images of UK wooden poles with a lean
  angle per pole, CC BY 4.0 on the listing, 3.41 GB zip. Street View origin makes redistribution
  murky: use for evaluation only, via a download script.
- Wikimedia Commons: Utility poles in Queensland (about 45, includes Brisbane), Stobie poles (70),
  Damaged utility poles (107), Leaning utility poles (47), Woodpecker damage (20). Free licences,
  per-file attribution. Good for Australian hero shots; too few to score accuracy.
- Insulators: MPID (CC BY 4.0), EPRI IDID (CC BY, access via IEEE DataPort competition), CPLID
  (free IEEE login). A possible extra defect type.
- Mapillary: street imagery with detected utility poles worldwide, free API token needed. Could add
  thousands of Australian poles without labels.
- No public labelled dataset of timber pole rot exists (searched Roboflow, Kaggle, HF, GitHub,
  papers). Roboflow Universe blocks automated search from the cloud sandbox; browse it manually.

**Infra notes:** `overpass-api.de` reset connections from the cloud sandbox; `overpass.private.coffee`
works but intermittently returns HTML 500 (the code retries across mirrors). GitHub repo creation
failed from the cloud session (the Claude GitHub App lacked permission), so Will created the repo.

## Known demo quirks (from the frontend build)

- Per-pole time shown is v1 + v2 model time summed ("per pole: photo read + decision"). Change
  `poleMs()` in `demo/app.js` if you only want one pass.
- In Real speed and late Ramp the output cards update in place, so the photo can briefly sit next to
  the previous pole's answers. Deliberate, reads as a blur.
- On static hosting the browser logs a 404 for `api/health`; that probe is how "Try a photo" hides itself.
- `demo/sample/` is mock data with drawn pole illustrations, generated by `demo/sample/make_sample.py`.
  The page falls back to it when `demo/data/` is missing (console warning only, no banner). Never film it.
- `window.__demo` exposes state for capture scripts.
