import argparse
import glob
import os

import amfm_decompy.basic_tools as basic
import amfm_decompy.pYAAPT as pYAAPT
import numpy as np
import torch
import torchaudio
from tqdm import tqdm


def get_yaapt_f0(audio: np.ndarray, sr: int = 16000, interp: bool = False) -> np.ndarray:
    to_pad = int(20.0 / 1000 * sr) // 2
    f0s = []
    for y in audio.astype(np.float64):
        y_pad = np.pad(y.squeeze(), (to_pad, to_pad), "constant", constant_values=0)
        pitch = pYAAPT.yaapt(
            basic.SignalObj(y_pad, sr),
            **{"frame_length": 20.0, "frame_space": 5.0, "nccf_thresh1": 0.25, "tda_frame_length": 25.0},
        )
        f0s.append(pitch.samp_interp[None, None, :] if interp else pitch.samp_values[None, None, :])

    return np.vstack(f0s)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_wav_dir", required=True)
    parser.add_argument("--output_f0_dir", required=True)
    args = parser.parse_args()

    wav_paths = sorted(glob.glob(os.path.join(args.input_wav_dir, "**", "*.wav"), recursive=True))
    print(f"Found {len(wav_paths)} wav files")

    with torch.no_grad():
        for wav_path in tqdm(wav_paths):
            rel_path = os.path.relpath(wav_path, args.input_wav_dir)
            out_path = os.path.join(args.output_f0_dir, rel_path).replace(".wav", ".pt")
            if os.path.isfile(out_path):
                continue

            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            audio, _ = torchaudio.load(wav_path)
            p = (audio.shape[-1] // 1280 + 1) * 1280 - audio.shape[-1]
            audio = torch.nn.functional.pad(audio, (0, p), mode="constant").data

            f0 = get_yaapt_f0(audio.numpy())
            f0 = torch.FloatTensor(f0.astype(np.float32).squeeze(0))
            torch.save(f0, out_path)


if __name__ == "__main__":
    main()
