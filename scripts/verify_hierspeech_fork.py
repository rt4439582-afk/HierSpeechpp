#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def has_text(path: Path, needle: str) -> bool:
    if not path.exists():
        return False
    try:
        return needle in path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False


def main() -> None:
    p = argparse.ArgumentParser(description="Verify whether a custom project is likely a HierSpeech++ fork")
    p.add_argument("--project_root", required=True)
    p.add_argument("--report_json", default="verify_hierspeech_report.json")
    args = p.parse_args()

    root = Path(args.project_root).resolve()

    checks = {
        "has_hierspeech_synth_file": (root / "hierspeechpp_speechsynthesizer.py").exists(),
        "has_ttv_transformer_file": (root / "ttv_v1" / "t2w2v_transformer.py").exists(),
        "has_inference_py": (root / "inference.py").exists(),
        "has_speechsr24k": (root / "speechsr24k" / "speechsr.py").exists(),
        "has_speechsr48k": (root / "speechsr48k" / "speechsr.py").exists(),
        "has_denoiser": (root / "denoiser" / "generator.py").exists(),
        "has_voice_conversion_noise_control": has_text(root / "hierspeechpp_speechsynthesizer.py", "voice_conversion_noise_control"),
        "has_infer_noise_control": has_text(root / "ttv_v1" / "t2w2v_transformer.py", "infer_noise_control"),
    }

    score = sum(1 for v in checks.values() if v)
    total = len(checks)
    ratio = score / total

    if ratio >= 0.85:
        verdict = "very_likely_hierspeechpp_fork"
    elif ratio >= 0.6:
        verdict = "likely_hierspeechpp_derived"
    elif ratio >= 0.35:
        verdict = "uncertain_partial_overlap"
    else:
        verdict = "unlikely_hierspeechpp"

    out = {
        "project_root": str(root),
        "score": score,
        "total": total,
        "ratio": round(ratio, 3),
        "verdict": verdict,
        "checks": checks,
    }

    Path(args.report_json).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\nSaved report: {Path(args.report_json).resolve()}")


if __name__ == "__main__":
    main()
