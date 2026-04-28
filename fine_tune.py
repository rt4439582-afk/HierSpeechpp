#!/usr/bin/env python3
import argparse
import json
import random
import subprocess
from pathlib import Path


def run(cmd):
    print("\n$", " ".join(str(x) for x in cmd))
    subprocess.run(cmd, check=True)


def split_train_test(filelist_dir: Path, test_ratio: float, seed: int = 1234):
    names = ["wav", "f0", "token", "w2v"]
    data = {}
    for n in names:
        p = filelist_dir / f"train_{n}.txt"
        lines = [x.strip() for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
        data[n] = lines

    n_total = len(data["wav"])
    idx = list(range(n_total))
    random.seed(seed)
    random.shuffle(idx)
    n_test = max(1, int(n_total * test_ratio)) if n_total > 1 else 0
    test_set = set(idx[:n_test])

    for n in names:
        train_lines = [x for i, x in enumerate(data[n]) if i not in test_set]
        test_lines = [x for i, x in enumerate(data[n]) if i in test_set]
        (filelist_dir / f"train_{n}.txt").write_text("\n".join(train_lines) + ("\n" if train_lines else ""), encoding="utf-8")
        (filelist_dir / f"test_{n}.txt").write_text("\n".join(test_lines) + ("\n" if test_lines else ""), encoding="utf-8")


def update_config(base_config: Path, out_config: Path, train_wav: Path, test_wav: Path, cleaner: str):
    cfg = json.loads(base_config.read_text(encoding="utf-8"))
    cfg.setdefault("data", {})
    cfg["data"]["train_filelist_path"] = str(train_wav.resolve())
    cfg["data"]["test_filelist_path"] = str(test_wav.resolve())
    cfg["data"]["text_cleaners"] = [cleaner]
    out_config.parent.mkdir(parents=True, exist_ok=True)
    out_config.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Compatibility fine-tune entrypoint for Persian TTV-v1")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--model_name", default="fine_tuned_model")
    ap.add_argument("--config", default="")
    ap.add_argument("--output_dir", default="./processed_data")
    ap.add_argument("--filelist_dir", default="./filelists")
    ap.add_argument("--cleaner", default="persian_cleaners")
    ap.add_argument("--test_ratio", type=float, default=0.1)
    ap.add_argument("--skip_preprocessing", action="store_true")
    ap.add_argument("--start_training", action="store_true")
    ap.add_argument("--wav_min", type=int, default=32)
    ap.add_argument("--wav_max", type=int, default=600)
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent
    data_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    filelist_dir = Path(args.filelist_dir).resolve()
    logs_dir = (repo / "logs" / args.model_name)

    if not args.skip_preprocessing:
        run(["python", str(repo / "scripts/validate_fa_dataset.py"),
             "--wav_dir", str(data_dir), "--text_dir", str(data_dir),
             "--report_csv", str(out_dir / "dataset_report.csv"),
             "--summary_json", str(out_dir / "dataset_summary.json")])

        run(["python", str(repo / "ttv_v1/preprocessing/extract_w2v_generic.py"),
             "--input_wav_dir", str(data_dir), "--output_w2v_dir", str(out_dir / "w2v")])
        run(["python", str(repo / "ttv_v1/preprocessing/extract_f0_generic.py"),
             "--input_wav_dir", str(data_dir), "--output_f0_dir", str(out_dir / "f0")])
        run(["python", str(repo / "ttv_v1/preprocessing/extract_token_generic.py"),
             "--input_text_dir", str(data_dir), "--output_token_dir", str(out_dir / "token"),
             "--cleaner", args.cleaner])
        run(["python", str(repo / "ttv_v1/preprocessing/prepare_filelist_generic.py"),
             "--wav_dir", str(data_dir),
             "--f0_dir", str(out_dir / "f0"),
             "--token_dir", str(out_dir / "token"),
             "--w2v_dir", str(out_dir / "w2v"),
             "--output_dir", str(filelist_dir),
             "--wav_min", str(args.wav_min), "--wav_max", str(args.wav_max)])

        split_train_test(filelist_dir, args.test_ratio)

    base_cfg = Path(args.config).resolve() if args.config else (repo / "ttv_v1/config_fa_template.json")
    if not base_cfg.exists():
        base_cfg = repo / "ttv_v1/config.json"

    out_cfg = logs_dir / "config.json"
    update_config(base_cfg, out_cfg, filelist_dir / "train_wav.txt", filelist_dir / "test_wav.txt", args.cleaner)

    print(f"\nPrepared config: {out_cfg}")
    print(f"Train list: {filelist_dir / 'train_wav.txt'}")
    print(f"Test list : {filelist_dir / 'test_wav.txt'}")

    if args.start_training:
        run(["python", str(repo / "train_ttv_v1.py"), "-c", str(out_cfg), "-m", args.model_name])


if __name__ == "__main__":
    main()
