#!/usr/bin/env bash
# Provision a fresh CloudLab/Emulab node for qkd_rl training.
#
# Written because the platform re-issues a NEW hostname every time the
# experiment is restarted, so this setup gets repeated. Idempotent: safe to
# re-run, it skips whatever is already in place.
#
# Usage (from a machine that can ssh the node):
#   scp deployment/setup.sh qinglong@<node>:~/
#   ssh qinglong@<node> 'bash ~/deployment/setup.sh'
#
# Then, from the LOCAL machine, run deployment/upload_data.sh to push the data.
set -euo pipefail

# Install into /opt, NOT $HOME. Emulab strips user home directories out of disk
# images -- the portal says so outright when you create one: "The contents of
# your home directory is NOT saved and will be deleted during the imaging
# process." Anything left in ~ is therefore lost the moment the node is imaged,
# which defeats the entire point of baking an image.
INSTALL_ROOT="/opt/qkd"
VENV="${INSTALL_ROOT}/venv"
PY="${VENV}/bin/python"

echo "== node =="
hostname
nproc
free -g | head -2
echo

# --- 0. report whether this node is even a sensible choice ----------------
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
else
    echo "GPU: NONE -- this workload is ~4x slower on the update phase without one"
fi
# /proc/cpuinfo spells this "model name" (lowercase) while lscpu prints
# "Model name" -- and under `set -e` a grep that matches nothing aborts the
# whole script, so the match flag must be tolerant of both spellings.
grep -iE "^model name" /proc/cpuinfo | head -1 || true
echo

# --- 1. python3-venv (missing on bare Emulab images) ----------------------
if ! python3 -c "import ensurepip" >/dev/null 2>&1; then
    echo "== installing python3.10-venv =="
    sudo -n apt-get update -qq
    sudo -n apt-get install -y -qq python3.10-venv python3-pip
fi

# --- 2. virtualenv ---------------------------------------------------------
sudo -n mkdir -p "${INSTALL_ROOT}"
sudo -n chown "$(id -u):$(id -g)" "${INSTALL_ROOT}"

# Test for a WORKING pip, not just the python binary: a venv created while
# ensurepip was missing leaves bin/python in place but no pip at all, and the
# check `[ -x ${PY} ]` would then skip the rebuild and die on the next line.
if ! "${PY}" -m pip --version >/dev/null 2>&1; then
    echo "== creating venv at ${VENV} =="
    rm -rf "${VENV}"
    python3 -m venv "${VENV}"
fi
"${PY}" -m pip install --quiet --upgrade pip

# --- 3. dependencies -------------------------------------------------------
echo "== installing dependencies =="
# Pick the torch wheel to match the hardware. The disk image this node may have
# been booted from carries a CPU-only torch (it was built on a node with no
# GPU); leaving that in place on a GPU box would silently idle the GPU, and the
# update phase -- 40-60% of each iteration, 75% of it backward pass -- measures
# ~4x faster on GPU than on CPU. Re-running this script is the fix.
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "   GPU present -> CUDA torch wheel"
    "${PY}" -m pip install --quiet --upgrade torch --index-url https://download.pytorch.org/whl/cu124
else
    echo "   no GPU -> CPU-only torch wheel"
    "${PY}" -m pip install --quiet --upgrade torch --index-url https://download.pytorch.org/whl/cpu
fi
"${PY}" -m pip install --quiet numpy scipy h5py networkx pyyaml pandas matplotlib tqdm

echo
echo "== versions =="
"${PY}" - <<'PYEOF'
import importlib
for mod in ("torch", "numpy", "scipy", "h5py", "networkx", "yaml", "pandas"):
    try:
        m = importlib.import_module(mod)
        print(f"  {mod:10s} {getattr(m, '__version__', '?')}")
    except Exception as exc:
        print(f"  {mod:10s} MISSING ({exc})")
PYEOF

echo
echo "SETUP_DONE"
echo "Next: from the LOCAL machine run deployment/upload_data.sh <node>"
