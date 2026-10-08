# Pole health triage with Matilda Jev

An electricity network has far more ageing poles than it can repair or replace in a year, so
someone has to decide which ones to fix first. This repo shows an open decision model
doing that triage: it looks at a photo of each pole, reads the pole's asset record, and
recommends replace, maintain, defer or reinspect, with a probability on every answer and
an engineer review queue for the cases it is unsure about.

It is a working demo, not a production system. It is built by [Beyond Data](https://beyond-data.com.au)
to show what a fast, open decision model can do on one real operational decision.

## What it does

**Step 1: photo to findings.** A ground-level photo of a pole goes in. The model answers four
typed questions in one pass:

| Question | Options |
|---|---|
| Is the pole standing vertical? | straight, leaning, cannot assess |
| Are the crossarm and top cleat level? | straight, tilted, not visible |
| Is vegetation near the wires? | clear, encroaching, not visible |
| Overall visible condition | critical, poor, fair, good, as new |

**Step 2: findings plus record to action.** The same model gets the photo, the step 1
findings, the pole's record (age, three inspections of shell thickness and ground-line wood,
surface defects, woodpecker damage, years since maintenance, bushfire zone, customers served,
critical customers such as a hospital) and the network's maintenance policy in plain English.
It returns one action and a public safety risk probability.

**Routing.** When the chosen action's probability is below 60 percent, or the photo cannot be
assessed, the pole goes to an engineer instead of being decided automatically.

The policy is plain text inside each request, so a network can change what the model is asked to
weigh by editing that text, without retraining the model.

## The model

[Matilda Jev](https://huggingface.co/Maincode/matilda-jev-v1) is a decision model released by
Maincode, an Australian AI company, under Apache-2.0. A decision model does not write text.
You give it a state (text, JSON, and up to four images) and typed questions with the options
you define, and it returns a probability for every option from one pass through the network, without
generating any text. That is why it is fast and cheap to run, and why every answer here comes with a number you can audit.

Because the weights are open, it can run inside the network's own environment. Photos and
asset records never have to leave it.

Maincode reports a median of 56.8 ms per decision on an AMD MI355X. Our own measurements, on
the hardware named next to them, are in the results section once recorded.

## Results

No results are published yet. Every number in this section and in the demo will come from a
recorded run, stored in `runs/` with the backend, hardware and date alongside it.

The evaluation scores the model against real labels:

- **Lean, crossarm and vegetation** against the PD-Defect field labels.
- **Record-only health** against the real inspector Health Index (1 to 5) on 4,800 timber
  poles. The Health Index itself is never shown to the model, only the measurements behind it.
- **Calibration**: when the model says 80 percent, is it right about 80 percent of the time?
- **Speed and cost**: server time per decision, and cost per 1,000 poles at the GPU's hourly price.

## The demo

`demo/` is a static page that replays a recorded run across 1,191 real pole locations in
Toowoomba, Queensland:

- **Live inspection**: each photo, the model reading it, the findings, the record and the
  decision, first slowed down so you can follow it, then at real speed.
- **Map**: every inspected pole turns red, amber, green or purple as it is decided.
- **Flip**: the same photo with two different records, and two different decisions.
- **Engineer queue**: the cases the model was not sure about, lowest confidence first.
- **Accuracy**: per-task accuracy, calibration and measured speed.

## Quickstart

Needs Python 3.11 or newer and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run polehealth fetch            # photos, labels, inspection records, pole locations (cached in data/raw)
uv run polehealth build-register   # data/register.json: 1,191 Toowoomba poles with histories
```

Then pick a backend for the model:

- **Your own GPU** (one card with 80 GB or more): follow `scripts/gpu/RUNBOOK.md`. It runs the
  official Maincode runtime and records speed numbers you can quote. We estimate a full run at
  a few dollars of rented GPU time; the measured cost goes in the results section.
- **The public Hugging Face Space** (free, slow, good for trying it): set `HF_TOKEN`, install
  the extra with `uv sync --extra space`, and use `--backend space`. Space timings include
  queueing, so they are never reported as model speed.

```bash
uv run polehealth eval --backend http --task all       # writes runs/<time>-http/
uv run polehealth demo-data --backend http              # writes demo/data/
uv run polehealth serve --backend http                  # http://127.0.0.1:8000, with "Try a photo"
```

`polehealth ask --backend http --image photo.jpg` answers one photo from the command line.

## Data, and what is real

| What | Source | Licence |
|---|---|---|
| Pole photos and labels | [PD-Defect](https://huggingface.co/datasets/EPDCL/pd-defect), field photos by APEPDCL, Andhra Pradesh, India | CC BY 4.0 |
| Pole locations | OpenStreetMap `power=pole` nodes, Toowoomba, QLD | ODbL, © OpenStreetMap contributors |
| Inspection histories | Western red cedar 50 ft pole inspections, Utility Analytics Network on Kaggle (Canada) | Not stated; downloaded at build time, not redistributed |

The locations, photos and inspection records are all real. The pairing between them is
synthetic: no public dataset links a pole's photo to that pole's history, so each Toowoomba
location is given one photo and one Canadian inspection history, and inspection years are
shifted so the newest reads as 2026. Bushfire zone, customers served and critical customers
are synthetic too. Most labelled photos show concrete poles, while much of Australia's ageing
fleet is timber. Full details are in [DATA_SOURCES.md](DATA_SOURCES.md).

## Repo layout

```
src/polehealth/   questions, model backends, data sources, register, evaluation, demo data, local server
scripts/gpu/      setup script and runbook for a rented GPU
demo/             static demo page
tests/            pytest suite, no network needed
docs/spec.md      build spec and data contract
```

Run the checks with `uv run pytest -q` and `uv run ruff check .`.

## Licence

Code: Apache-2.0. Data keeps its own licence (see above). Matilda Jev is Apache-2.0 from Maincode.

Built by Beyond Data, Brisbane. We build working software on a business's own operational data to
improve one specific decision, such as which poles a network maintains this year.
