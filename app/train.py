"""Sailfish training plane — the BYO Google Cloud path (PLAN.md §6.4, §10.9).

We hand the user (1) a zip bundle (curated data + the trainer + its requirements) and (2) a
self-contained script templated with their org/project/region/HF-repo. They run it in their own
gcloud shell; it stages the bundle, spins an ephemeral SPOT A100, trains, pushes to their HF repo,
and self-destructs. We never hold their credentials.
"""
import zipfile
from pathlib import Path
from typing import Optional

from app import data as data_mod

_ROOT = Path(__file__).parent.parent
BYO = _ROOT / "train" / "byo_gcloud"
TRAINER = _ROOT / "train" / "finetune_target.py"


def corpus_path() -> Path:
    """Prefer the curated corpus; fall back to the raw scrape."""
    cur = data_mod.DATA_DIR / "tool_calls.curated.jsonl"
    return cur if cur.exists() else (data_mod.DATA_DIR / "tool_calls.jsonl")


def build_training_bundle(dest: Optional[Path] = None) -> Path:
    """Zip everything the VM needs: the corpus (as train.jsonl), the trainer, its requirements."""
    corpus = corpus_path()
    if not corpus.exists():
        raise FileNotFoundError("no corpus yet — scrape (and optionally curate) first")
    dest = dest or (data_mod.DATA_DIR / "sailfish_training_bundle.zip")
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(corpus, arcname="train.jsonl")
        z.write(TRAINER, arcname="finetune_target.py")
        z.write(BYO / "requirements-train.txt", arcname="requirements-train.txt")
    return dest


def generate_byo_script(project: str, region: str, hf_repo: str, zone: Optional[str] = None,
                        base: str = "google/gemma-4-E4B-it", epochs: float = 1.0,
                        bucket: Optional[str] = None,
                        bundle_file: str = "sailfish_training_bundle.zip") -> str:
    """Fill the launcher template (with the VM startup script inlined) — one self-contained script."""
    if not project or not region or not hf_repo:
        raise ValueError("project, region, and hf_repo are required")
    zone = zone or f"{region}-a"
    bucket = bucket or f"gs://{project}-sailfish-train"
    launcher = (BYO / "train_on_gcloud.sh.tmpl").read_text(encoding="utf-8")
    startup = (BYO / "vm_startup.sh").read_text(encoding="utf-8")
    repl = {
        "{{PROJECT}}": project, "{{REGION}}": region, "{{ZONE}}": zone,
        "{{HF_REPO}}": hf_repo, "{{BASE_MODEL}}": base, "{{EPOCHS}}": str(epochs),
        "{{BUCKET}}": bucket, "{{BUNDLE_FILE}}": bundle_file,
        "{{VM_STARTUP}}": startup,  # inline last (contains no placeholders of its own)
    }
    for k, v in repl.items():
        launcher = launcher.replace(k, v)
    return launcher
