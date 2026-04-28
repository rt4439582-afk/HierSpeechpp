import argparse
import csv
import json
from pathlib import Path

import torchaudio


def find_pairs(wav_dir: Path, txt_dir: Path):
    wavs = sorted(wav_dir.rglob("*.wav"))
    pairs = []
    missing_txt = []
    for wav in wavs:
        rel = wav.relative_to(wav_dir)
        txt = (txt_dir / rel).with_suffix(".txt")
        if txt.exists():
            pairs.append((wav, txt, rel))
        else:
            missing_txt.append(str(rel))
    return pairs, missing_txt


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--wav_dir", required=True)
    p.add_argument("--text_dir", required=True)
    p.add_argument("--report_csv", required=True)
    p.add_argument("--summary_json", required=True)
    p.add_argument("--min_sec", type=float, default=1.0)
    p.add_argument("--max_sec", type=float, default=20.0)
    args = p.parse_args()

    wav_dir = Path(args.wav_dir)
    txt_dir = Path(args.text_dir)

    pairs, missing_txt = find_pairs(wav_dir, txt_dir)
    if len(pairs) == 0:
        raise RuntimeError("No wav/txt pairs found. Check folder structure and matching filenames.")

    rows = []
    invalid_sr = 0
    invalid_channels = 0
    too_short = 0
    too_long = 0
    empty_txt = 0

    for wav, txt, rel in pairs:
        info = torchaudio.info(str(wav))
        sec = info.num_frames / info.sample_rate
        with txt.open("r", encoding="utf-8") as f:
            text = f.readline().strip()

        sr_ok = info.sample_rate == 16000
        ch_ok = info.num_channels == 1
        len_ok = args.min_sec <= sec <= args.max_sec
        txt_ok = len(text) > 0

        if not sr_ok:
            invalid_sr += 1
        if not ch_ok:
            invalid_channels += 1
        if sec < args.min_sec:
            too_short += 1
        if sec > args.max_sec:
            too_long += 1
        if not txt_ok:
            empty_txt += 1

        rows.append(
            {
                "rel_path": str(rel),
                "sample_rate": info.sample_rate,
                "channels": info.num_channels,
                "duration_sec": round(sec, 3),
                "text_len": len(text),
                "sr_ok": sr_ok,
                "channels_ok": ch_ok,
                "length_ok": len_ok,
                "text_ok": txt_ok,
            }
        )

    Path(args.report_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary = {
        "num_pairs": len(pairs),
        "missing_txt": len(missing_txt),
        "invalid_sr": invalid_sr,
        "invalid_channels": invalid_channels,
        "too_short": too_short,
        "too_long": too_long,
        "empty_txt": empty_txt,
        "missing_txt_examples": missing_txt[:20],
    }

    Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if summary["missing_txt"] > 0 or summary["invalid_sr"] > 0 or summary["invalid_channels"] > 0 or summary["empty_txt"] > 0:
        raise SystemExit("Dataset validation failed. See summary_json/report_csv.")


if __name__ == "__main__":
    main()
