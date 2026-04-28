import argparse
import glob
import os

import torch
from tqdm import tqdm

from text import text_to_sequence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_text_dir", required=True, help="Root dir containing .txt transcripts")
    parser.add_argument("--output_token_dir", required=True, help="Root dir to store .pt token files")
    parser.add_argument("--cleaner", default="transliteration_cleaners", help="Cleaner name from ttv_v1/text/cleaners.py")
    args = parser.parse_args()

    txt_paths = sorted(glob.glob(os.path.join(args.input_text_dir, "**", "*.txt"), recursive=True))
    print(f"Found {len(txt_paths)} text files")

    for txt_path in tqdm(txt_paths):
        rel_path = os.path.relpath(txt_path, args.input_text_dir)
        out_path = os.path.join(args.output_token_dir, rel_path).replace(".txt", ".pt")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.readline().strip()

        token = text_to_sequence(text, [args.cleaner])
        torch.save(torch.LongTensor(token), out_path)


if __name__ == "__main__":
    main()
