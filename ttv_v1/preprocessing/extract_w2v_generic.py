import argparse
import glob
import os

import torch
import torchaudio
import transformers
from tqdm import tqdm


class Wav2vec2(torch.nn.Module):
    def __init__(self, layer: int = 7):
        super().__init__()
        self.mms = transformers.Wav2Vec2ForPreTraining.from_pretrained("facebook/mms-300m")
        for param in self.mms.parameters():
            param.requires_grad = False
            param.grad = None
        self.mms.eval()
        self.feature_layer = layer

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.mms(x.squeeze(1), output_hidden_states=True)
        y = outputs.hidden_states[self.feature_layer]
        return y.permute((0, 2, 1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_wav_dir", required=True)
    parser.add_argument("--output_w2v_dir", required=True)
    parser.add_argument("--feature_layer", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = Wav2vec2(layer=args.feature_layer).to(device)

    wav_paths = sorted(glob.glob(os.path.join(args.input_wav_dir, "**", "*.wav"), recursive=True))
    print(f"Found {len(wav_paths)} wav files")

    with torch.no_grad():
        for wav_path in tqdm(wav_paths):
            rel_path = os.path.relpath(wav_path, args.input_wav_dir)
            out_path = os.path.join(args.output_w2v_dir, rel_path).replace(".wav", ".pt")
            if os.path.isfile(out_path):
                continue

            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            audio, _ = torchaudio.load(wav_path)
            p = (audio.shape[-1] // 1280 + 1) * 1280 - audio.shape[-1]
            audio = torch.nn.functional.pad(audio, (0, p), mode="constant").data
            y_pad = torch.nn.functional.pad(audio, (40, 40), "reflect")

            w2v_x = model(y_pad.to(device))
            torch.save(w2v_x.cpu(), out_path)


if __name__ == "__main__":
    main()
