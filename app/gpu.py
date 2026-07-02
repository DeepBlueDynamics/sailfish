"""GPU autodetect + tier selection. The product's core branch:
Tier A (>=16GB): stock model + MTP drafter via vLLM.  Tier B (<16GB): fine-tuned model, llama.cpp + ngram.
Tier follows VRAM+checkpoint, not architecture (the A10G is Ampere and runs the full stack) — arch only
selects kernels (FP8 on Ada+, Marlin on Ampere).  See ARCHITECTURE.md."""
import subprocess
from typing import Dict, Tuple

# compute-capability -> arch family (for kernel/feature selection, not tiering)
_ARCH = {
    "8.6": "ampere", "8.9": "ada", "9.0": "hopper", "12.0": "blackwell", "10.0": "blackwell",
}


def detect_gpu() -> Dict:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode != 0 or not out.stdout.strip():
            return {"present": False, "reason": out.stderr.strip()[:200] or "no nvidia-smi output"}
        name, mem, cap = [x.strip() for x in out.stdout.strip().splitlines()[0].split(",")]
        vram_gb = round(float(mem) / 1024, 1)
        arch = _ARCH.get(cap, f"cc{cap}")
        return {"present": True, "name": name, "vram_gb": vram_gb, "compute_cap": cap, "arch": arch}
    except FileNotFoundError:
        return {"present": False, "reason": "nvidia-smi not found (no GPU / not passed into container)"}
    except Exception as e:
        return {"present": False, "reason": str(e)[:200]}


def capable_tier(gpu: Dict) -> str:
    """The BEST tier this card could run (by VRAM), ignoring which engine the image ships.
    Lets a big-card user see they're Tier-A capable even while the P1 image serves Tier B."""
    if not gpu.get("present"):
        return "none"
    return "A" if gpu.get("vram_gb", 0) >= 16 else "B"


def choose_tier(gpu: Dict, override: str = "auto") -> Tuple[str, str, str]:
    """Returns (tier, engine, drafter)."""
    if override in ("A", "B"):
        tier = override
    elif not gpu.get("present"):
        return ("none", "none", "none")
    else:
        vram = gpu.get("vram_gb", 0)
        tier = "A" if vram >= 16 else "B"

    if tier == "A":
        # stock INT4 target + trained MTP drafter, the challenge technique
        return ("A", "vllm", "mtp")
    # Tier B: fine-tuned target, llama.cpp, zero-VRAM n-gram speculation (measured 76/177 on a 3060)
    return ("B", "llama.cpp", "ngram")
