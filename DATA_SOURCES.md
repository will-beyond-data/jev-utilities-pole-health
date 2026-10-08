# Data sources

Everything this demo shows comes from public sources. This file lists each one, its licence, the attribution to
carry, and what is committed to the repository versus downloaded at build time. `polehealth fetch` downloads the
raw data into `data/raw/`, which is gitignored.

## Summary

| Key | Source | Licence | Committed? |
|---|---|---|---|
| `pd_lean` | PD-Defect, `pole_lean_assessment/classification` (test and val) | CC BY 4.0 | No. Downloaded at build time; `demo-data` writes a resized subset to `demo/data/photos/`, which is not committed |
| `pd_crossarm` | PD-Defect, `crossarm_top_cleat_tilt_assessment/object_detection` (test) | CC BY 4.0 | No, as above |
| `pd_vegetation` | PD-Defect, `vegetation_conductor` (validation images and `GT_ValidationSet.csv`) | CC BY 4.0 | No, as above |
| `osm` | OpenStreetMap `node[power=pole]`, Toowoomba bounding box | ODbL 1.0 | Yes: the raw extract `data/raw/osm_toowoomba.json` |
| `wrc50` | Kaggle, Western Red Cedar 50-ft Pole (utilityanalytics) | Not stated (Kaggle lists it as Unknown) | No. Downloaded at build time. The register and generated demo data carry derived records (see below) |
| `wrc45` | Kaggle, utility power pole condition dataset 1 (utilityanalytics, 45-ft poles) | Not stated on the dataset page | No. Downloaded on request, not used by the demo |
| Model | Maincode Matilda Jev v1 (`Maincode/matilda-jev-v1`) | Apache-2.0 | Not included. Downloaded by `scripts/gpu/setup.sh` |

## PD-Defect (pole photos)

- What: field-captured smartphone images of power distribution poles, collected with APEPDCL in Eastern Andhra
  Pradesh, India. The photos are not of Australian poles.
- Where: https://huggingface.co/datasets/EPDCL/pd-defect (also on Zenodo, doi 10.5281/zenodo.18074045).
- Licence: Creative Commons Attribution 4.0 International (https://creativecommons.org/licenses/by/4.0/).
- Attribution: Masabathula, S. B., Vanapalli, S., Kanchi, G., V.A.N, S. R., V, V. L., Chilamkurthy, S. (2026).
  PD-Defect: A Field-Captured Smartphone Image Dataset for AI-Based Inspection of Power Distribution
  Infrastructure (v1.0). Zenodo. https://doi.org/10.5281/zenodo.18074045. Short credit used in the demo:
  "PD-Defect, APEPDCL, CC BY 4.0".
- Changes made: images are resized to a maximum edge of 960 px and re-encoded as JPEG (quality 80) for the demo.
  The copies sent to the model are resized to a maximum edge of 1024 px.
- Labels used:
  - Lean: `Leaned` and `Straight` map to `leaning` and `straight`. The dataset's `Rejected` photos are not used:
    they are unusable inspection photos (blurred, blocked, or the pole too small or far away to judge).
  - Crossarm: the dataset gives YOLO boxes with six classes (`crossarm`, `topcleat`, `v-crossarm`, each `_straight`
    or `_tilted`). This repo derives an image-level label: `tilted` if any box is a tilted class, otherwise
    `straight`. That derivation is ours, not the dataset's.
  - Vegetation: `GT_ValidationSet.csv` gives `Risky` or `Safe`, mapped to `encroaching` or `clear`. The dataset
    describes these as image-level risk labels for the validation split.
- Committed: none of the photos. `polehealth demo-data` writes resized copies of the ones the demo uses to
  `demo/data/photos/`, each credited in `poles.json`. The demo video in `docs/media/` shows some of them, with the
  short credit on screen.

## OpenStreetMap (pole locations)

- What: nodes tagged `power=pole` inside the bounding box south -27.62, west 151.88, north -27.50, east 152.02
  (Toowoomba, Queensland), about 1,191 poles at the time of the build.
- Where: queried through the Overpass API (mirrors tried in order: overpass.private.coffee, overpass.kumi.systems).
- Licence: Open Data Commons Open Database License (ODbL) 1.0.
- Attribution: © OpenStreetMap contributors. Data available under the ODbL: https://www.openstreetmap.org/copyright
- Committed: the raw extract `data/raw/osm_toowoomba.json`, so a fresh checkout never has to call Overpass.
- Notice: `data/register.json` and `demo/data/poles.json` are derived databases of OpenStreetMap data and are
  offered under the ODbL as to the OpenStreetMap-derived content (positions and OSM node ids). The map tiles shown
  in the demo are OpenStreetMap tiles and carry their own attribution in the map corner.
- Ignored: all OSM tags other than `power=pole`, including `material`.

## Kaggle Western Red Cedar pole datasets (asset records)

- What: `wrc50` is 4,800 timber poles inspected in 1999, 2009 and 2019; `wrc45` is 3,000 poles inspected in 1998,
  2008 and 2018. Columns: ID, Age, ST1, ST2, ST3 (shell thickness readings), Surface conditions, GL (groundline),
  Wood pecker holes, Carrying transformer, and a Health Index (1 to 5) on the newest inspection.
- Where: https://www.kaggle.com/datasets/utilityanalytics/western-red-cedar-50ft-pole and
  https://www.kaggle.com/datasets/utilityanalytics/utility-power-pole-condition-dataset1
- Licence: not stated on the dataset pages; Kaggle lists it as Unknown. Treat it as unknown. For that reason the raw
  workbooks are never committed. They are downloaded by `polehealth fetch`.
- What is committed: no records. `data/register.json` is rebuilt by `polehealth build-register` and the demo data
  (`demo/data/poles.json`) by `polehealth demo-data`, and both are left out of the repository. They carry per-pole
  records derived from `wrc50` (shell thickness, groundline, surface condition, woodpecker holes, transformer type,
  health index, age), some of which appear in the demo video for this non-commercial demonstration. If you intend
  to redistribute generated demo data, check the licence position with the dataset owner first.
- Attribution: Utility Analytics, "Western Red Cedar 50-ft Pole", Kaggle.

### The year shift

The `wrc50` inspections were done in 1999, 2009 and 2019. So that the newest inspection reads as the present, the
register shifts every inspection year and every age forward by 7 years: 2006, 2016 and 2026, and ages plus 7.
Nothing else is changed. The record-only benchmark (`record_health_index`) uses the original unshifted values
because it compares the model with the dataset's real Health Index.

## The pairing is synthetic

No public dataset links a pole photo to the history of that pole. In the demo:

- A pole's location comes from OpenStreetMap (Toowoomba).
- Its inspection history comes from a `wrc50` pole chosen at random (seeded, each history used at most once).
- Its photo comes from PD-Defect, chosen at random (seeded, each photo used at most once).
- The network attributes (bushfire zone, customers served, critical customer, last maintained year) are generated
  with a seeded random number generator. Bushfire zone is higher towards the edges of the bounding box.

So no photo shows the pole its record describes, and the record does not describe the pole at that location. The
demo footer and the README say so.

## Matilda Jev

Maincode's Matilda Jev v1 is an open-weights decision model, Apache-2.0, https://huggingface.co/Maincode/matilda-jev-v1.
This repo only calls it; it does not contain weights or model code. Maincode validated its runtime on AMD MI355X.
Runs on NVIDIA GPUs are an independent test and are recorded as such in each run's hardware field.
