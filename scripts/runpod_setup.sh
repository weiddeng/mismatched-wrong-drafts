#!/usr/bin/env bash
set -euo pipefail

# Ubuntu 24.04 / Python 3.12 pods mark the base environment "externally managed"
# (PEP 668), which makes `pip install` refuse to run. These pods are disposable, so
# permit system-wide installs (equivalent to passing --break-system-packages to every pip).
export PIP_BREAK_SYSTEM_PACKAGES=1

echo "=== System packages ==="
apt-get update && apt-get install -y --no-install-recommends tmux
apt-get clean && rm -rf /var/lib/apt/lists/*

echo "=== Python / pip info ==="
python --version
pip --version

echo "=== Python packages ==="
# Install unsloth/trl/datasets if not already present (some pod images lack them)
pip install --no-cache-dir \
  unsloth \
  trl \
  datasets \
  math-verify

# Pin vllm to 0.18.x AFTER other installs (0.19 has torch.compile issue with Unsloth)
# This ensures unsloth/trl can't silently pull in a newer vllm
pip install --no-cache-dir "vllm>=0.18,<0.19"

# A dependency can pull a flashinfer-cubin whose version mismatches flashinfer-python
# (e.g. cubin 0.6.8 vs python 0.6.6). That mismatch crashes at `import flashinfer` /
# vLLM engine init — i.e. at model load, before any training step. Working setups run
# on flashinfer-python alone, so drop the cubin if it was auto-installed. No-op if absent.
echo "=== Dropping any mismatched flashinfer-cubin (no-op if absent) ==="
pip uninstall -y flashinfer-cubin >/dev/null 2>&1 || true

echo "=== Sanity check ==="
python - <<'PY'
import importlib.metadata as md
required = ["vllm", "trl", "unsloth", "unsloth-zoo", "datasets", "math-verify", "torch"]
missing = []
for pkg in required:
    try:
        print(f"  {pkg}={md.version(pkg)}")
    except Exception:
        print(f"  {pkg}=NOT FOUND")
        missing.append(pkg)
import torch
print(f"  cuda={torch.version.cuda}")
if missing:
    raise RuntimeError(f"Missing packages: {missing}")
print("All OK")
PY

echo "=== Setup complete ==="
echo "Run: tmux new -s train"
