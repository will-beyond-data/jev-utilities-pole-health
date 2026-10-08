# Handoff: status, decisions, research, next steps

Last updated 2026-10-08 by the cloud session that built the first version. Will is picking this up in
a local Claude Code session on his own machine, where his tokens live.

## Status at a glance

| Piece | State |
|---|---|
| Python pipeline (`src/polehealth`) | Built. 91 tests pass, ruff clean. Verified end to end with the mock backend only. |
| Data sources | All six fetch and work: PD-Defect lean, crossarm, vegetation; Kaggle wrc50 and wrc45 (no login needed); OSM Toowoomba poles (extract committed). |
| Register | 1,191 Toowoomba poles, deterministic (seed 7). Rebuild with `polehealth build-register`. |
| Demo page (`demo/`) | Map, pole panel, flip, engineer queue, accuracy views built and screenshot-checked on sample data. "Live inspection" hero view and "Try a photo": see the note at the end of this file for whether they landed. |
| Local server (`polehealth serve`) | Built and tested with the mock backend. |
| GPU runbook (`scripts/gpu/`) | Written for RunPod. `setup.sh` is syntax-checked only; it has never run on a GPU. |
| Real model output | **None yet.** No call to Matilda Jev has succeeded from this project. The first one is the next step. |
| README | Drafted. Results section deliberately empty until a real run exists. Not yet brand-reviewed. |

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
- **Video:** opens on the Live inspection view: real photo, an abstract processing sweep with a
  millisecond timer, findings bars, record card, decision stamp with measured times, mini-map dot.
  "Ramp" mode: three poles slowed down with the real ms shown, then real speed.

## Next steps, in order

1. **Local setup.** `uv sync --extra space`, `uv run pytest -q`, `uv run polehealth fetch`,
   `uv run polehealth build-register`.
2. **First real answers, free, via the Hugging Face Space.** With `HF_TOKEN` set:
   `uv run polehealth ask --backend space --image <a PD-Defect Leaned photo from data/raw>` then
   `uv run polehealth eval --backend space --task all --limit 10`. The anonymous Space quota was
   exhausted from the cloud container, so a token is required. Free accounts get limited ZeroGPU
   minutes per day.
3. **Check the questions before paying for GPU time.** Look at the Space answers. If accuracy on
   lean, crossarm or vegetation is poor, iterate on the wording in `src/polehealth/questions.py`
   (the only place prompts live) against the Space, which is free. Ignore Space latency.
4. **Measured run on RunPod** (follow `scripts/gpu/RUNBOOK.md`): rent the pod, `bash
   scripts/gpu/setup.sh` (downloads about 54 GB, starts the official Maincode server, runs a smoke
   test), clone this repo on the pod, run `polehealth eval --backend http --task all` and
   `polehealth demo-data --backend http --inspect 300` on the pod itself, then `polehealth cost`.
   Copy `runs/` and `demo/data/` back, then **stop and terminate the pod**. Expected compute is a
   few AUD; the top-up is the bigger cost.
   - If `setup.sh` fails at model load, that is the NVIDIA risk (Maincode only validated on AMD
     MI355X). Stop the pod before debugging. Fallback: run the eval through the Space and present
     accuracy only, with no speed claims.
5. **Film the demo** while you have real data: `uv run polehealth serve`, open
   `http://127.0.0.1:8000/?autoplay=1&mode=ramp`, screen-record at 1920x1080. For the live "Try a
   photo" moment, keep the pod up and tunnel the server (RUNBOOK has the SSH tunnel steps).
6. **Fill the README results** from `runs/<dir>/eval_summary.json` only, commit `demo/data/` (keep
   the photo subset small; at max edge 960 each photo is about 130 KB), then brand-review README
   and the demo copy with `beyond-data-brand-voice`.
7. **Before going public:** resolve the Kaggle licence question (below), get Will's approval, flip
   the repo to public.
8. **Content:** carousel and LinkedIn brief go through Will's content-engine repo (Beyond Data
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
Python 3.12, transformers 5.17.0, torch 2.14.0. Context limits auto-size to GPU memory (80 GB gets
32k tokens).

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

## Note on the demo's Live inspection view

The Live inspection view, the "Try a photo" drop zone and the health-level display fix were
requested in a second frontend pass that was still running when this handoff was first pushed. If
`demo/index.html` has no "Live inspection" tab, that pass did not land: rebuild it from the brief in
`docs/spec.md` ("Frontend") plus the beat list under "Video" above.
