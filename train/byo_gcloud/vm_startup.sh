#!/usr/bin/env bash
# Runs ON the ephemeral A100 VM (Deep Learning image, CUDA + torch preinstalled). Reads its job from
# instance metadata, trains, pushes to the user's HF repo, then DELETES ITSELF. set -e + the EXIT trap
# guarantee the VM dies whether we succeed or fail (max-run-duration in the launcher is the backstop).
set -euo pipefail
exec > >(tee -a /var/log/sailfish-train.log) 2>&1
echo "[sailfish-vm] start $(date -u +%FT%TZ)"

MD="http://metadata.google.internal/computeMetadata/v1/instance"
mdget() { curl -s -H "Metadata-Flavor: Google" "$MD/$1"; }

BUNDLE_URL="$(mdget attributes/bundle-url)"
HF_REPO="$(mdget attributes/hf-repo)"
HF_TOKEN="$(mdget attributes/hf-token)"
BASE="$(mdget attributes/base-model)"
EPOCHS="$(mdget attributes/epochs)"
NAME="$(mdget name)"
ZONE="$(mdget zone | awk -F/ '{print $NF}')"

selfdestruct() {
  echo "[sailfish-vm] self-destruct: deleting $NAME in $ZONE"
  gcloud compute instances delete "$NAME" --zone="$ZONE" --quiet || true
}
trap selfdestruct EXIT

export HF_TOKEN
mkdir -p /opt/sailfish-train && cd /opt/sailfish-train
gsutil cp "$BUNDLE_URL" ./bundle.zip
unzip -o ./bundle.zip

# The DL image ships torch matched to its CUDA — do NOT reinstall torch (would break the match).
pip install --no-cache-dir -r requirements-train.txt

python3 finetune_target.py \
  --base "$BASE" \
  --data train.jsonl \
  --out out/adapter \
  --epochs "$EPOCHS" \
  --merge

echo "[sailfish-vm] pushing merged weights -> $HF_REPO"
export _HF_REPO="$HF_REPO"
python3 - <<'PY'
import os
from huggingface_hub import HfApi
repo = os.environ["_HF_REPO"]
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(repo, exist_ok=True, private=True, repo_type="model")
api.upload_folder(folder_path="out/adapter", repo_id=repo)
print("uploaded", repo)
PY

echo "[sailfish-vm] done $(date -u +%FT%TZ) — model at https://huggingface.co/$HF_REPO"
