import argparse
import glob
import json
from pathlib import Path

import torch


def classify_checkpoint(obj):
    if isinstance(obj, dict) and {"model", "iteration", "optimizer", "learning_rate"}.issubset(obj.keys()):
        return "train_checkpoint"
    if isinstance(obj, dict) and all(hasattr(v, "shape") for v in obj.values()):
        return "state_dict"
    return "unknown"


def find_latest_checkpoint(model_dir: Path):
    ckpts = sorted(glob.glob(str(model_dir / "G_*.pth")), key=lambda p: int("".join(ch for ch in Path(p).stem if ch.isdigit()) or 0))
    return Path(ckpts[-1]) if ckpts else None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="Path to checkpoint file (.pth)")
    p.add_argument("--config", default=None, help="Optional path to config.json")
    p.add_argument("--logs_root", default="logs", help="Logs root to suggest resume targets")
    args = p.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    ckpt_type = classify_checkpoint(ckpt)

    print("=== Checkpoint Inspection ===")
    print("path:", ckpt_path)
    print("type:", ckpt_type)

    if ckpt_type == "train_checkpoint":
        print("iteration:", ckpt.get("iteration"))
        print("learning_rate:", ckpt.get("learning_rate"))
        model_keys = len(ckpt["model"].keys())
        print("model tensors:", model_keys)
    elif ckpt_type == "state_dict":
        print("state_dict tensors:", len(ckpt.keys()))
        print("note: this is usually inference-style checkpoint, not directly resumable with optimizer state")
    else:
        print("note: unknown structure; manual inspection needed")

    cfg_path = Path(args.config) if args.config else ckpt_path.parent / "config.json"
    if cfg_path.exists():
        print("config:", cfg_path)
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        data = cfg.get("data", {})
        print("config.data.text_cleaners:", data.get("text_cleaners"))
        print("config.data.sampling_rate:", data.get("sampling_rate"))
    else:
        print("config: NOT FOUND near checkpoint")

    logs_root = Path(args.logs_root)
    if logs_root.exists():
        print("\n=== Existing experiments under logs/ ===")
        for d in sorted([x for x in logs_root.iterdir() if x.is_dir()]):
            latest = find_latest_checkpoint(d)
            if latest:
                print(f"- {d.name}: latest={latest.name}")

    print("\n=== Recommended next step ===")
    exp_name = ckpt_path.parent.name
    if ckpt_type == "train_checkpoint" and cfg_path.exists():
        print(f"Resume command:")
        print(f"CUDA_VISIBLE_DEVICES=0 python train_ttv_v1.py -c {cfg_path} -m {exp_name}")
        print("(Use same -m so trainer picks latest G_*.pth from logs/<exp_name>.)")
    elif ckpt_type == "state_dict":
        print("This checkpoint looks inference-only. Safer path: run stage01/02 pipeline and resume from train-style logs checkpoints.")
    else:
        print("Checkpoint format unknown. Do not resume blindly.")


if __name__ == "__main__":
    main()
