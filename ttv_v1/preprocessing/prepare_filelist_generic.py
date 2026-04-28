import argparse
import glob
import os

import torch
import torchaudio
from tqdm import tqdm


def filter_audio_len(data_len: int, wav_min: int, wav_max: int) -> bool:
    return wav_min <= data_len <= wav_max


def make_filelist(file_list, filename: str) -> None:
    with open(filename, "w", encoding="utf-8") as file:
        for item in file_list:
            file.write(item + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav_dir", required=True)
    parser.add_argument("--f0_dir", required=True)
    parser.add_argument("--token_dir", required=True)
    parser.add_argument("--w2v_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--wav_min", type=int, default=32)
    parser.add_argument("--wav_max", type=int, default=600)
    parser.add_argument("--text_min", type=int, default=1)
    parser.add_argument("--text_max", type=int, default=200)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    wavs = sorted(glob.glob(os.path.join(args.wav_dir, "**", "*.wav"), recursive=True))
    print("Wav num:", len(wavs))

    filtered_wavs = []
    for wav in tqdm(wavs):
        rel_path = os.path.relpath(wav, args.wav_dir)
        f0_path = os.path.join(args.f0_dir, rel_path).replace(".wav", ".pt")
        tok_path = os.path.join(args.token_dir, rel_path).replace(".wav", ".pt")
        w2v_path = os.path.join(args.w2v_dir, rel_path).replace(".wav", ".pt")

        if not (os.path.isfile(f0_path) and os.path.isfile(tok_path) and os.path.isfile(w2v_path)):
            continue

        f0_value = torch.load(f0_path, map_location="cpu")
        if f0_value.sum() == 0:
            continue

        data, _ = torchaudio.load(wav)
        data_len = data.size(-1) // 320
        if not filter_audio_len(data_len, args.wav_min, args.wav_max):
            continue

        txt = torch.load(tok_path, map_location="cpu")
        len_txt = txt.size(-1)
        if len_txt * 2 + 1 > data_len:
            continue
        if not filter_audio_len(len_txt, args.text_min, args.text_max):
            continue

        filtered_wavs.append(wav)

    print("Filtered num:", len(filtered_wavs))

    train_wav = os.path.join(args.output_dir, "train_wav.txt")
    make_filelist(filtered_wavs, train_wav)

    train_f0 = [os.path.join(args.f0_dir, os.path.relpath(w, args.wav_dir)).replace(".wav", ".pt") for w in filtered_wavs]
    train_tok = [os.path.join(args.token_dir, os.path.relpath(w, args.wav_dir)).replace(".wav", ".pt") for w in filtered_wavs]
    train_w2v = [os.path.join(args.w2v_dir, os.path.relpath(w, args.wav_dir)).replace(".wav", ".pt") for w in filtered_wavs]

    make_filelist(train_f0, os.path.join(args.output_dir, "train_f0.txt"))
    make_filelist(train_tok, os.path.join(args.output_dir, "train_token.txt"))
    make_filelist(train_w2v, os.path.join(args.output_dir, "train_w2v.txt"))


if __name__ == "__main__":
    main()
