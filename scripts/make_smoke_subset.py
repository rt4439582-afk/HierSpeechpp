import argparse
import random
import shutil
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src_wav_dir", required=True)
    p.add_argument("--src_text_dir", required=True)
    p.add_argument("--dst_root", required=True)
    p.add_argument("--num_samples", type=int, default=200)
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()

    src_wav = Path(args.src_wav_dir)
    src_txt = Path(args.src_text_dir)
    dst_wav = Path(args.dst_root) / "wav16k"
    dst_txt = Path(args.dst_root) / "text"

    wavs = sorted(src_wav.rglob("*.wav"))
    pairs = []
    for w in wavs:
        rel = w.relative_to(src_wav)
        t = (src_txt / rel).with_suffix(".txt")
        if t.exists():
            pairs.append((w, t, rel))

    if len(pairs) == 0:
        raise RuntimeError("No wav/txt pairs found")

    random.seed(args.seed)
    pick = random.sample(pairs, k=min(args.num_samples, len(pairs)))

    for w, t, rel in pick:
        ow = dst_wav / rel
        ot = (dst_txt / rel).with_suffix(".txt")
        ow.parent.mkdir(parents=True, exist_ok=True)
        ot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(w, ow)
        shutil.copy2(t, ot)

    print(f"Copied {len(pick)} samples to {Path(args.dst_root).resolve()}")


if __name__ == "__main__":
    main()
