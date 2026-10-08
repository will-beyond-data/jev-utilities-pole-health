#!/usr/bin/env bash
# Provider-neutral setup for Matilda Jev on a freshly rented NVIDIA box.
#
# Target: Ubuntu 22.04 or 24.04, one GPU with 80 GB or more of VRAM (A100 80GB, H100 80GB,
# RTX PRO 6000 96GB), about 80 GB free disk (the weights are about 54 GB).
#
# Maincode validated this runtime on AMD MI355X only. NVIDIA is untested, so the last step here is a
# smoke test. If it fails, stop the box and read scripts/gpu/RUNBOOK.md before spending more money.
#
# Usage:   bash scripts/gpu/setup.sh
# Options (environment variables):
#   WORKDIR     where everything goes. Use a path on the persistent volume. Default $HOME/jev
#   MODEL_DIR   where the weights go. Default $WORKDIR/matilda-jev-v1
#   CUDA_INDEX  PyTorch wheel index. The right cuXXX depends on the driver: run `nvidia-smi` and read
#               "CUDA Version" in the header. cu128 needs a driver that reports 12.8 or newer.
#               Use cu126 for a 12.6 driver. Default https://download.pytorch.org/whl/cu128
#   PORT        server port. Default 8083
#   HOST        bind address. Default 0.0.0.0 (the API key below still applies). Use 127.0.0.1 to keep it local.
#   HF_TOKEN    optional Hugging Face token, only needed if the download is rate limited or gated

set -euo pipefail

WORKDIR="${WORKDIR:-$HOME/jev}"
MODEL_DIR="${MODEL_DIR:-$WORKDIR/matilda-jev-v1}"
VENV="${VENV:-$WORKDIR/venv}"
CUDA_INDEX="${CUDA_INDEX:-https://download.pytorch.org/whl/cu128}"
PORT="${PORT:-8083}"
HOST="${HOST:-0.0.0.0}"
TORCH_VERSION="2.14.0"
TORCHVISION_VERSION="0.29.0"
MIN_VRAM_MIB=75000
MIN_DISK_GB=80

say() { printf '\n== %s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

say "Checking the machine"
command -v nvidia-smi >/dev/null || die "nvidia-smi not found. This needs an NVIDIA GPU with the driver installed."
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
vram=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1)
[ "$vram" -ge "$MIN_VRAM_MIB" ] || die "GPU has ${vram} MiB; the 27B model in bf16 needs about 80 GB or more."
mkdir -p "$WORKDIR"
free_gb=$(df -BG --output=avail "$WORKDIR" | tail -n1 | tr -dc '0-9')
[ "$free_gb" -ge "$MIN_DISK_GB" ] || die "only ${free_gb} GB free in $WORKDIR; need ${MIN_DISK_GB} GB or more."
echo "CUDA wheel index: $CUDA_INDEX (check it against the driver version above)"

say "Installing uv"
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

say "Creating the Python 3.12 environment"
[ -d "$VENV" ] || uv venv --python 3.12 "$VENV"
PY="$VENV/bin/python"

say "Installing PyTorch $TORCH_VERSION and torchvision $TORCHVISION_VERSION"
uv pip install --python "$PY" "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" --index-url "$CUDA_INDEX"
"$PY" -c "import torch; print('torch', torch.__version__, 'cuda available:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
"$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || die "torch cannot see the GPU. The CUDA wheel index probably does not match the driver; set CUDA_INDEX."

say "Downloading Maincode/matilda-jev-v1 (about 54 GB, resumable)"
uv pip install --python "$PY" "huggingface_hub[cli]"
"$VENV/bin/hf" download Maincode/matilda-jev-v1 --local-dir "$MODEL_DIR"

say "Installing the runtime requirements"
# requirements-runtime.txt does not pin torch; the two wheels installed above stay in place.
uv pip install --python "$PY" -r "$MODEL_DIR/requirements-runtime.txt"

say "Preparing the API key"
KEY_FILE="$WORKDIR/mj_api_key"
if [ ! -s "$KEY_FILE" ]; then
  umask 077
  openssl rand -hex 24 > "$KEY_FILE"
fi
MJ_API_KEY="$(cat "$KEY_FILE")"
export MJ_API_KEY
echo "key stored in $KEY_FILE (not printed). Export it for polehealth with: export MJ_API_KEY=\$(cat $KEY_FILE)"

say "Starting the server on $HOST:$PORT"
if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
  echo "something already answers on port $PORT; leaving it alone"
else
  PYTHONPATH="$MODEL_DIR/runtime" PYTHONNOUSERSITE=1 \
    nohup "$PY" -m maincode_jev_serve.server \
      --checkpoint "$MODEL_DIR" --model-name matilda-jev-v1 \
      --host "$HOST" --port "$PORT" \
      > "$WORKDIR/server.log" 2>&1 &
  echo $! > "$WORKDIR/server.pid"
  echo "pid $(cat "$WORKDIR/server.pid"), log $WORKDIR/server.log"
fi

say "Waiting for the model to load (the first start can take several minutes)"
for i in $(seq 1 240); do
  if curl -fsS "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"status":"ready"'; then
    echo "ready after about $((i * 5)) seconds"
    break
  fi
  if [ -f "$WORKDIR/server.pid" ] && ! kill -0 "$(cat "$WORKDIR/server.pid")" 2>/dev/null; then
    tail -n 40 "$WORKDIR/server.log"
    die "the server exited. The log above is the reason; this is where an NVIDIA incompatibility would show."
  fi
  sleep 5
  [ "$i" -lt 240 ] || { tail -n 40 "$WORKDIR/server.log"; die "not ready after 20 minutes"; }
done
curl -fsS "http://127.0.0.1:$PORT/health"; echo

say "Smoke test (one text-only decision)"
curl -fsS -D - -o "$WORKDIR/smoke.json" -X POST "http://127.0.0.1:$PORT/v1/systemone" \
  -H "Authorization: Bearer $MJ_API_KEY" -H "Content-Type: application/json" \
  -d '{"model":"matilda-jev-v1","state":"A timber pole is 90 years old, has serious surface defects and has lost most of its shell thickness.","questions":{"action":{"type":"choice","instructions":"What should the network do?","criteria":{"replace":"replace it","defer":"leave it"}}}}' \
  | grep -i -E '^(HTTP|server-timing)'
cat "$WORKDIR/smoke.json"; echo
grep -q '"answers"' "$WORKDIR/smoke.json" || die "the smoke test did not return answers"

say "Done"
cat <<MSG
Server:   http://127.0.0.1:$PORT   (log $WORKDIR/server.log, pid in $WORKDIR/server.pid)
API key:  $KEY_FILE
Next:     clone the repo here, then follow scripts/gpu/RUNBOOK.md (polehealth eval, then demo-data).
Stop the billing: stop AND terminate the instance in the provider console when you are done.
MSG
