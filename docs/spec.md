# Build spec: jev-utilities-pole-health

Internal build doc. Public-facing copy lives in README.md.

## What this is

An open-source demo, published by Beyond Data, showing a decision model (Maincode's
Matilda Jev, open weights, Apache-2.0) triaging the health of electricity distribution
poles and recommending what to do with each one.

- **v1, photo to health.** A pole photo goes in. Matilda Jev returns typed answers:
  lean (straight / leaning / cannot_assess), crossarm condition, vegetation risk, and an
  overall health score on a 5-level scale.
- **v2, photo plus record to action.** The same model gets the photo, the v1 answers and
  the pole's asset record (age, inspection history, years since maintenance, criticality)
  and returns REPLACE / MAINTAIN / DEFER / REINSPECT with probabilities. Low-confidence
  answers are routed to an engineer review queue.

The story for the video: one Australian open model, one forward pass per decision, tens of
milliseconds, real photos, real pole locations, real inspection histories, honest seams.

This is a portfolio demo, not a production system. Honesty rules:

1. Every number shown in the demo comes from a recorded run, with the backend, hardware
   and date recorded alongside it. Never invent a latency or an accuracy figure.
2. Mock decisions exist only so the UI can be built without a GPU. Any data file produced
   by the mock backend carries `"backend": "mock"` and the UI shows a loud MOCK banner.
3. The pairing of photos, locations and records is synthetic (no public dataset links a
   pole photo to that pole's history). Say so in the UI footer and the README.

## Matilda Jev API (verified from the model repo runtime, 2026-10-08)

Self-hosted server: `POST {base}/v1/systemone`, optional `Authorization: Bearer <MJ_API_KEY>`.

```json
{
  "model": "matilda-jev-v1",
  "state": "<string | JSON object | JSON array>",
  "questions": {
    "<key>": {"type": "choice", "instructions": "...", "criteria": {"opt_a": "description or null", "opt_b": null}},
    "<key>": {"type": "noul", "instructions": "...", "criteria": {"true": "...", "false": "..."}},
    "<key>": {"type": "score", "instructions": "...", "criteria": ["lowest", "...", "highest"]}
  },
  "images": ["data:image/jpeg;base64,..."]
}
```

- `model` must equal the served name (default `matilda-jev-v1`) or `maincode-jev-latest`.
- `images`: max 4, base64 data URLs (png/jpeg/webp), max 8 MB decoded, max 16 MP. We
  resize to max edge 1024 and JPEG quality 85 before sending.
- `score` criteria: 2 to 10 levels, lowest first. `choice`: 1 to 255 options.
- Response:

```json
{"model": "matilda-jev-v1",
 "answers": {
   "<choice key>": {"type": "choice", "choice": "opt_a", "probabilities": {"opt_a": 0.93, "opt_b": 0.07}, "confidence": 0.9},
   "<noul key>":   {"type": "noul", "noul": 0.81},
   "<score key>":  {"type": "score", "score": 2.7, "probabilities": {"0": 0.1, "1": 0.2, "2": 0.4, "3": 0.2, "4": 0.1}, "legend": {"0": "critical", "...": "..."}, "confidence": 0.5}},
 "usage": {"input_tokens": 812, "output_tokens": 0}}
```

  Note: `score` is the probability-weighted mean of level indexes (0-based). Treat
  `confidence` as optional (compute max probability if absent).
- Health: `GET {base}/health` returns `{"status": "ready", "device": ..., ...}`.
- Errors: 413 too long, 422 invalid, 503 loading, 529 busy (retry with backoff).

Hugging Face Space backend (free ZeroGPU, needs an `HF_TOKEN` for quota):
`gradio_client.Client("hugging-apps/matilda-jev", token=HF_TOKEN).predict(json.dumps(body), api_name="/decide_request")`
returns the same response shape plus `latency_s` (in-Space compute time). Wall time via the
Space includes queueing and GPU attach, so Space latencies are never shown as "the model's
speed"; the recorded run file stores both `latency_ms` (client wall) and `model_ms` (from
`latency_s` when present, else the `server-timing` header for HTTP).

Reference runtime source (do not vendor, read only): `../mj/runtime/maincode_jev_serve/`.

## Data sources (see DATA_SOURCES.md for attribution)

| Key | Source | Licence | Committed to repo? |
|---|---|---|---|
| `pd_lean` | EPDCL/pd-defect `pole_lean_assessment/classification/{test,val}` (Leaned / Straight; Rejected is excluded as unusable photos) | CC-BY-4.0 | No. `polehealth demo-data` writes resized copies (max edge 960) to the gitignored `demo/data/` |
| `pd_crossarm` | EPDCL/pd-defect `crossarm_top_cleat_tilt_assessment/object_detection/test` (YOLO boxes, classes in data.yaml) | CC-BY-4.0 | No, same as above |
| `pd_vegetation` | EPDCL/pd-defect `vegetation_conductor` validation images + `GT_ValidationSet.csv` (Risky / Safe) | CC-BY-4.0 | No, same as above |
| `osm_toowoomba` | OpenStreetMap `node[power=pole]` in bbox (-27.62,151.88,-27.50,152.02), ~1,191 poles | ODbL, "© OpenStreetMap contributors" | Yes (derived positions, ODbL notice) |
| `wrc50` | Kaggle utilityanalytics/western-red-cedar-50ft-pole: 4,800 poles x inspections 1999/2009/2019; columns ID, Age, ST1, ST2, ST3, Surface conditions, GL, Wood pecker holes, Carrying transformer, Health Index (2019 sheet only, 1-5) | Unknown | No. Downloaded at build time |
| `wrc45` | Kaggle utilityanalytics/utility-power-pole-condition-dataset1: 3,000 poles x 1998/2008/2018, same columns, 2018 has Health Index | Unknown | No |

Download URLs:
- HF files: `https://huggingface.co/datasets/EPDCL/pd-defect/resolve/main/<path>`; list a folder with
  `https://huggingface.co/api/datasets/EPDCL/pd-defect/tree/main/<folder>` (paginated, follow `Link` header).
- Kaggle (no auth needed, verified): `https://www.kaggle.com/api/v1/datasets/download/utilityanalytics/<slug>` (zip with one xlsx, one sheet per inspection year).
- Overpass: `https://overpass.private.coffee/api/interpreter` (overpass-api.de resets connections from here). POST form `data=[out:json][timeout:90];node[power=pole](S,W,N,E);out body;`. It intermittently returns HTML 500: retry up to 6 times with backoff, validate JSON.

Shift inspection years forward so the newest inspection reads as 2026 (wrc50: +7 years,
so 2006/2016/2026; ages +7). State this in DATA_SOURCES.md.

## Asset register (built by `polehealth build-register`)

For each OSM pole (stable sort by OSM id), deterministically (seeded) pick one wrc50 pole
history. Add synthetic network attributes, seeded and plausible for regional Queensland:

- `bushfire_zone`: "high" | "medium" | "low" (about 15 / 30 / 55 %, higher on the bbox edges, which are rural fringe)
- `customers_served`: int, 1 to 400, long tail
- `critical_customer`: bool, about 3 % (hospital, aged care, water pumping); name the type in `critical_customer_type`
- `last_maintained_year`: between the second and third inspection, or null if never
- `years_since_maintenance`: computed (2026 minus last_maintained_year) or null
- `material`: "timber" (the wrc histories are timber; OSM `material` tag ignored)

Precompute everything numeric the model would otherwise have to calculate (ages, years
since, change in shell thickness between inspections). Decision models are weak at date
maths and arithmetic.

## Demo data contract (`demo/data/`, read by the static frontend)

`poles.json`:

```json
{"generated_at": "ISO8601", "town": "Toowoomba, QLD", "attribution": ["..."],
 "poles": [{
   "id": "osm-123456", "lat": -27.56, "lon": 151.95,
   "inspected": true,
   "photo": "photos/pd_lean_test_100.jpg",
   "photo_source": {"dataset": "pd_lean", "label": "Leaned", "credit": "PD-Defect, APEPDCL, CC BY 4.0"},
   "record": {
     "material": "timber", "age_years": 58, "install_year": 1968,
     "inspections": [{"year": 2006, "shell_thickness": [0.59, 0.56, 0.55], "groundline": 0.41, "surface": "Serious Defects", "woodpecker_holes": false},
                     {"year": 2016, "...": "..."},
                     {"year": 2026, "...": "...", "health_index": 1}],
     "shell_thickness_change_10y": -0.21,
     "transformer": "none | single_phase | three_phase",
     "last_maintained_year": 2019, "years_since_maintenance": 7,
     "bushfire_zone": "high", "customers_served": 140,
     "critical_customer": false, "critical_customer_type": null,
     "source_history_id": "wrc50-1"
   }
 }]}
```

`inspected: false` poles have no photo and get no decision (grey dots). Inspect a demo
subset (target 300 to 600 poles, one photo each, no photo reused).

`decisions.json`:

```json
{"backend": "http | space | mock", "model": "matilda-jev-v1", "hardware": "e.g. 1x NVIDIA A100 80GB",
 "recorded_at": "ISO8601", "engineer_threshold": 0.6,
 "decisions": {"osm-123456": {
   "v1": {"answers": {"lean": {...}, "crossarm": {...}, "vegetation": {...}, "health": {...}}, "latency_ms": 61.2, "model_ms": 48.0},
   "v2": {"answers": {"action": {...}, "safety_risk_now": {...}}, "latency_ms": 58.1, "model_ms": 45.0},
   "action": "replace | maintain | defer | reinspect",
   "confidence": 0.82,
   "route": "auto | engineer"
 }}}
```

`flip.json`: one photo, two contrasting records (young, low-risk, recently maintained vs
old, high bushfire zone, critical customer, falling shell thickness), and both v2 results.
Same shape as one decisions entry per side, plus the two records.

`eval_summary.json`: per task (lean, crossarm, vegetation, record_health_index):
`n`, `accuracy`, `macro_f1`, `confusion` (label x prediction counts), `ece` (10 bins),
`reliability` (bins: mean confidence, accuracy, count), `latency_ms` p50/p90/p99,
`model_ms` p50/p90/p99, `backend`, `hardware`, `recorded_at`. Plus `cost` block:
`gpu_hourly_usd`, `decisions_per_hour`, `usd_per_1000_poles` (only when backend is http).

## Questions (single source of truth: `src/polehealth/questions.py`)

v1 state: `{"asset": "electricity distribution pole", "image": "ground-level inspection photo"}`.
v1 questions:
- `lean` choice: straight / leaning / cannot_assess (pole top or base hidden, or angle too distorted).
  The benchmark scores only straight and leaning photos; cannot_assess stays as an answer so a bad
  uploaded photo routes to an engineer.
- `crossarm` choice: straight / tilted / not_visible.
- `vegetation` choice: clear / encroaching (vegetation touching or within reach of conductors) / not_visible.
- `health` score, lowest first: ["critical: imminent failure risk", "poor: significant defects", "fair: some defects", "good: minor wear", "as new"].

v2 state: `{"pole": <record without lat/lon>, "photo_findings": {lean, crossarm, vegetation with chosen option and probability, health score}, "policy": "<plain-English maintenance policy>"}` plus the same image.
v2 questions:
- `action` choice: replace (structural failure risk, replace this cycle) / maintain (repair or treat now, e.g. reinforce, re-tension, vegetation clearance) / defer (safe to leave until the next inspection cycle) / reinspect (photo or record is not enough, send a crew).
- `safety_risk_now` noul: "Is there a risk to public safety before the next scheduled inspection?"

The policy text is a parameter (`--policy-file`) because changing it mid-run is a demo beat.
Default policy, roughly: prioritise public safety and bushfire zones; critical customers lower
the threshold to act; prefer maintain over replace when structure is sound.

Routing: route to engineer when the chosen action's probability is below
`engineer_threshold` (default 0.6), or when v1 lean is cannot_assess.

Record-only benchmark (`record_health_index` task): text-only request with the wrc50 2019
record (age, shell thickness, groundline, surface, woodpecker, transformer) asking for the
5-level health score; compare argmax level to the real Health Index (1-5). This is the one
place where we have real ground truth for the record side.

## Python package layout

```
pyproject.toml          uv-managed; deps: click, httpx, pillow, pandas, openpyxl; extras: space = gradio_client; dev = pytest, ruff
src/polehealth/
  jev.py                Backend protocol; HttpBackend, SpaceBackend, MockBackend; retries; timing
  images.py             load, EXIF-transpose, resize, data URL
  questions.py          v1/v2 builders, default policy, routing rule
  sources.py            downloads: pd-defect subsets, kaggle wrc, overpass; cache in data/raw (gitignored)
  register.py           build register from OSM + wrc50 + seeded synthetic attributes
  evaluate.py           run tasks over labelled sets, write runs/<timestamp>/results.jsonl and eval_summary.json
  metrics.py            accuracy, macro F1, confusion, ECE, reliability bins, percentiles
  demo_data.py          build demo/data/*.json and resized photos from register + a run
  cli.py                click group `polehealth`: fetch, build-register, ask, eval, demo-data, flip, cost
scripts/gpu/setup.sh    fresh Ubuntu + CUDA GPU box: venv, torch 2.14 cu wheels, runtime reqs, hf download, start server on :8083 with MJ_API_KEY
scripts/gpu/RUNBOOK.md  rent, run, record, tear down, with cost maths
tests/                  pytest; no network; MockBackend + tiny fixtures
```

MockBackend: deterministic from a seed and from labels when the caller passes them (so the
UI shows plausible variety). Never used for any number that appears in README or video.

## Frontend (`demo/`)

Static, no build step: `index.html`, `app.js`, `styles.css`; Leaflet from cdnjs with an
exact version; OSM tiles with attribution. Served with `python -m http.server -d demo`.
Screens/beats:

1. Map of Toowoomba poles. "Run inspection" replays decisions in order with the recorded
   latencies (option to play at 1x real latency, or a fixed speed for video). Dots go from
   grey to red (replace) / amber (maintain) / green (defer) / purple (reinspect or engineer
   queue). Counters: poles decided, median model ms, decisions per second, engineer queue size.
2. Pole panel on click: photo, v1 answers as probability bars, record summary (age, years
   since maintenance, shell thickness trend sparkline across three inspections, bushfire
   zone, customers, critical customer), v2 action with probability bars, route.
3. Flip view: same photo, two records side by side, two decisions.
4. Engineer queue list.
5. Accuracy view from eval_summary.json: per task accuracy, reliability chart, latency.
Footer: data credits and the honesty note. No MOCK banner (Will, 2026-10-08): mock data is for
development only and only logs a console warning.

6. **Live inspection (default tab, the video's hero shot).** Large photo stage (about 60% width),
   output panel right, mini-map inset, filmstrip of upcoming photos. Per pole: photo slides in with
   credit; "reading photo" phase with an abstract full-frame scan sweep and a running ms timer (never
   boxes, crosshairs or highlights: the model does not localise); v1 bars animate and the health
   gauge settles on a level name; "checking the asset record" phase with the record card and
   shell-thickness sparkline; decision stamp (REPLACE / MAINTAIN / DEFER / REINSPECT or ENGINEER
   REVIEW) with probability, safety risk and measured "Photo read: N ms, Decision: N ms" (from
   model_ms, else latency_ms, else nothing); mini-map dot turns its colour. Modes: Explain (about
   4 to 5 s per pole, chip "Slowed down so you can see it. Real time per pole: N ms"), Real speed
   (actual model_ms), Ramp (3 poles Explain, then accelerate; default for capture). Keys: space,
   right arrow, R, 1/2/3. URL params `?autoplay=1&mode=ramp`. Cursor hides while playing. Order:
   clearest replace (highest leaning probability), clearest defer (highest straight probability),
   lowest-confidence engineer case, then decisions.json order.
7. **Try a photo** (only when served by `polehealth serve`, detected via `GET api/health` 200
   `{"status":"ready"}`): drop a photo, client resizes to max edge 1024 JPEG q85, `POST api/decide`
   `{"image": dataURL, "record": null | record}`, render the same sequence with the response (one
   decisions.json entry). Optional record picker with 3 presets. Hidden on static hosting.
No em dashes anywhere in UI text. Beyond Data branding is light: wordmark text "Beyond Data"
in the header, neutral palette with one accent, light and dark themes.
