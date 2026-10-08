# GPU runbook: record the real run on a rented GPU

Goal: run Matilda Jev on a rented NVIDIA GPU for about two hours, record real accuracy and latency, copy the
results home, and shut everything down. Provider: RunPod. `setup.sh` itself is provider-neutral and works on any
Ubuntu 22.04 or 24.04 box with an NVIDIA GPU of 80 GB or more.

## Read this first

- Maincode validated this runtime on AMD MI355X only. NVIDIA is untested. The first thing you do on the box is the
  smoke test at the end of `setup.sh`. If the server will not start or the smoke test fails, stop the pod, keep the
  log, and do not spend more money guessing.
- Every number that appears in the README or the video must come from a run recorded on this box. Run
  `polehealth eval` and `polehealth demo-data` on the pod itself, so latencies are not polluted by internet round trips.
- Hard budget: AUD 20 in total, including the minimum credit top-up. The plan below uses about a quarter of it.

## 1. Before you rent anything (on your own machine, free)

```bash
uv sync
uv run pytest -q
uv run polehealth fetch --limit 40          # small download, proves the data sources work
uv run polehealth eval --backend mock --task all --limit 20
```

Check that `data/register.json` is committed and current (it is the asset register the demo uses). The pod does not
need OpenStreetMap, so you avoid the flaky Overpass step on the clock.

Decide the hard stop in advance: write down the time you will terminate the pod, whatever state the work is in.

## 2. RunPod: account, credit, pod

1. Create an account at runpod.io and verify your email.
2. Billing: add the minimum credit the page allows. It was USD 10 when this was written, which is about AUD 15.20 at
   1.52 AUD per USD. Confirm the number on the billing page. Do not turn on auto top-up. The balance is your hard
   cap: a pod stops when the credit runs out.
3. Pods, then Deploy. Choose on-demand (not spot, which can be interrupted mid-run). Secure Cloud or Community Cloud
   are both fine. Community Cloud is usually cheaper.
4. GPU: one A100 80GB or one H100 80GB. Take whichever is cheaper per hour at the time. An RTX PRO 6000 96GB is also
   fine. Anything under 80 GB will not hold the model. The 27B model in bf16 needs about 54 GB for weights plus room
   for activations.
5. Template: any recent RunPod PyTorch template on Ubuntu 22.04 or 24.04. Under additional filters, pick a machine
   whose driver reports CUDA 12.8 or newer if you can; otherwise you will use an older `CUDA_INDEX` (see section 3).
   The template's own PyTorch is not used. `setup.sh` builds its own Python 3.12 environment.
6. Disk: set container disk plus volume disk to at least 120 GB in total (for example 40 GB container and 100 GB
   volume). Weights are about 54 GB, the Python environment about 10 GB, the pd-defect images about 2 GB, and the
   Hugging Face download needs working room. Put `WORKDIR` on the volume (`/workspace` on RunPod templates).
7. Ports: leave them as they are. You run `eval` and `demo-data` on the pod, so port 8083 does not need to be exposed.
   Only expose or tunnel it for the live demo (section 6), and prefer an SSH tunnel to a public port.
8. Add your SSH public key in RunPod settings before deploying if you want `scp`. The web terminal also works.
9. Deploy. Note the time. The meter is running from now.

## 3. On the pod

Open a terminal (web terminal or SSH).

```bash
nvidia-smi                                   # note the GPU name and the "CUDA Version" in the header
export WORKDIR=/workspace/jev
git clone <your fork or the repo URL> /workspace/jev-utilities-pole-health
cd /workspace/jev-utilities-pole-health
bash scripts/gpu/setup.sh
```

If `nvidia-smi` reports a CUDA version below 12.8, run `CUDA_INDEX=https://download.pytorch.org/whl/cu126 bash
scripts/gpu/setup.sh` (or the cuXXX that matches). The script checks the GPU, installs uv, builds the Python 3.12
environment, installs `torch==2.14.0` and `torchvision==0.29.0`, downloads the weights, installs
`requirements-runtime.txt`, generates `MJ_API_KEY` into `$WORKDIR/mj_api_key`, starts the server on port 8083, waits
for `/health`, and sends one smoke-test request. Expect 15 to 40 minutes on the first run, mostly the download and
the first model load. Do not continue until the smoke test prints answers.

Then install the CLI and fetch what the benchmarks need:

```bash
export MJ_API_KEY=$(cat /workspace/jev/mj_api_key)
export MJ_URL=http://127.0.0.1:8083
curl -fsS $MJ_URL/health                                  # note the device name
uv sync
uv run polehealth fetch --only pd_lean,pd_crossarm,pd_vegetation,wrc50    # about 2 GB, a few minutes
```

### Warm up and record

```bash
HW="1x NVIDIA A100 80GB"        # write what nvidia-smi said, for the record
uv run polehealth eval --backend http --task all --limit 100 --hardware "$HW"       # quick pass first
uv run polehealth eval --backend http --task all --limit 500 --hardware "$HW"       # the recorded run
```

Use `--concurrency 1` (the default) for anything you quote as latency. The first few requests after a start can be
slower while kernels settle, which is why a quick pass comes first. Both commands write
`runs/<UTC timestamp>-http/`. If the box dies mid-run, start another pod, repeat section 3, and continue the same
directory with `--resume runs/<dir>`.

Then build the demo data, work out the cost, and rebuild so the cost block is copied into `demo/data/`:

```bash
RUN=$(ls -d runs/*-http | tail -n1)
uv run polehealth demo-data --backend http --run "$RUN" --inspect 400 --hardware "$HW"
uv run polehealth cost --gpu-hourly-usd <the hourly price you were charged> --run "$RUN"
uv run polehealth demo-data --backend http --run "$RUN" --inspect 400 --hardware "$HW"   # cached, takes seconds
```

`demo-data` runs the v1 and v2 decisions for the demo poles against the live server and writes `demo/data/`. `cost`
needs those decisions (a pole costs one v1 plus one v2 request), so it runs after the first `demo-data`. It saves
the cost block into the run's `eval_summary.json`, and the second `demo-data` copies that file across.

To show the "change the policy mid-run" beat, run `demo-data` again with `--policy-file my_policy.txt`. It re-runs
only the v2 decisions.

## 4. Copy results back

Option A, `runpodctl` (installed on RunPod pods), from the pod:

```bash
tar czf results.tgz runs/ demo/data/
runpodctl send results.tgz
```

It prints a one-time code. On your machine, install `runpodctl` and run `runpodctl receive <code>`.

Option B, `scp` (the pod page shows the host, port and user under Connect, over TCP):

```bash
scp -P <port> root@<host>:/workspace/jev-utilities-pole-health/results.tgz .
tar xzf results.tgz
```

Check on your machine that `runs/<dir>/results.jsonl`, `runs/<dir>/eval_summary.json`, `demo/data/decisions.json`
and `demo/data/poles.json` exist and that `decisions.json` says `"backend": "http"`, not mock. Only then tear down.

## 5. Tear down: stop AND terminate

1. Pods page: Stop the pod, then Terminate it. Stopping alone keeps the disks, and they keep billing. Terminate
   deletes them. Do not terminate before you have verified the copy in section 4.
2. Check the Pods page shows no pods and the Storage or Volumes page shows no network volumes you created.
3. Check the billing page: the balance should have gone down by about the cost table below, and nothing should still be
   running. Leave auto top-up off. Any remaining credit stays in the account.

## 6. Optional: the live "drop in a photo" moment

Keep the pod running only as long as the filming needs it. Do not expose port 8083 publicly. Forward it over SSH
instead and run the small web server on your own machine:

```bash
ssh -N -L 8083:127.0.0.1:8083 -p <port> root@<host>       # leave running
export MJ_API_KEY=<the key from /workspace/jev/mj_api_key on the pod>
uv run polehealth serve --backend http --base-url http://127.0.0.1:8083 --port 8000
```

Open http://127.0.0.1:8000/. The browser talks only to `polehealth serve`, which holds the key. The wall-clock
latency now includes your internet round trip, but `model_ms` is still the server's own time. Quote `model_ms` as
the model's speed, and call the wall time what it is.

## 7. Cost maths (AUD, budget AUD 20 including the top-up)

AUD per hour is the pod's USD hourly price times 1.52. The prices below are example rates to show the shape of the
bill, not quotes. Replace them with the live price on the day.

| GPU price (USD per hour) | AUD per hour | Cost of a 2 hour session (AUD) | Hours the AUD 15.20 minimum credit lasts |
|---|---|---|---|
| 1.50 | 2.28 | 4.56 | 6.7 |
| 2.00 | 3.04 | 6.08 | 5.0 |
| 2.50 | 3.80 | 7.60 | 4.0 |
| 3.00 | 4.56 | 9.12 | 3.3 |
| 4.00 | 6.08 | 12.16 | 2.5 |

Plan for about 2 hours on the meter, as an estimate:

| Step | Time (estimate) |
|---|---|
| Deploy, SSH in, clone | 10 min |
| `setup.sh` (download, install, first load, smoke test) | 30 to 40 min |
| `fetch` for the benchmarks | 5 min |
| `eval` quick pass plus the 500 per task run | 15 to 25 min |
| `demo-data` (400 poles, two requests each) | 5 to 10 min |
| Copy back, verify, terminate | 10 min |

Disk is billed per GB while the pod exists, which is small next to the GPU (120 GB of volume is a few cents an hour,
check the page). Card fees or tax may add a little on top of the top-up. Rules that keep you under AUD 20:

- Top up the minimum once. Never enable auto top-up. Do not add a second top-up to finish a run.
- Hard stop at 3 hours on the meter. At USD 3 per hour that is still under AUD 14.
- If the smoke test fails after 45 minutes, terminate and think. A failed NVIDIA start is information, not a reason to
  keep paying.
- If the setup is clean, a second session (for example a retake of the policy-change beat) fits inside the remaining
  credit. A fresh pod means downloading the 54 GB again, so it is cheaper to keep the first pod until the work is
  copied back.

After the run, `polehealth cost` turns the measured `model_ms` into cost per 1,000 poles, using the hourly price you
paid. It assumes the GPU is busy back to back, so it is a floor for a production system, not a quote.
