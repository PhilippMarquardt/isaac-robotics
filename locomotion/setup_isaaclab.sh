#!/bin/bash
set -ex
export OMNI_KIT_ACCEPT_EULA=Y
export DEBIAN_FRONTEND=noninteractive
export TMPDIR=/dev/shm/pip-tmp
export PIP_CACHE_DIR=/workspace/pip-cache
mkdir -p /dev/shm/pip-tmp /workspace/pip-cache

echo "=== apt deps ==="
apt-get update -qq
apt-get install -y -qq libxt6 libglu1-mesa git 2>&1 | tail -3

echo "=== uninstall pypi isaaclab fork ==="
pip uninstall -y isaaclab || true

echo "=== upgrade isaacsim 5.0 -> 5.1 ==="
pip install --no-input --upgrade 'isaacsim[all,extscache]==5.1.0' --extra-index-url https://pypi.nvidia.com 2>&1 | tail -3

echo "=== clone IsaacLab ==="
cd /workspace
if [ ! -d IsaacLab ]; then
  git clone --depth 1 https://github.com/isaac-sim/IsaacLab.git
fi
cd IsaacLab
git fetch --depth 1 origin 4df6560e1 2>&1 | tail -2 || true
git checkout 4df6560e1 2>&1 | tail -2 || true

echo "=== install IsaacLab sources ==="
cd /workspace/IsaacLab
pip install --no-input -e source/isaaclab
pip install --no-input -e source/isaaclab_assets
pip install --no-input -e source/isaaclab_tasks
pip install --no-input -e source/isaaclab_rl
pip install --no-input -e source/isaaclab_mimic

echo "=== DONE $(date) ==="
