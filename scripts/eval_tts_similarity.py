import argparse
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
import torchaudio
from jiwer import cer, wer
from speechbrain.inference.speaker import EncoderClassifier
from transformers import pipeline


def load_audio_16k(path: str):
    wav, sr = torchaudio.load(path)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    return wav


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a / (torch.norm(a) + 1e-9)
    b = b / (torch.norm(b) + 1e-9)
    return float(torch.dot(a, b).item())


def duration_seconds(path: str) -> float:
    y, sr = librosa.load(path, sr=16000)
    return float(len(y) / sr)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_csv", required=True, help="CSV with columns: system_id,audio_path,ref_text,ref_speaker_wav")
    parser.add_argument("--output_csv", default="eval_results.csv")
    parser.add_argument("--output_summary_csv", default="eval_summary.csv")
    parser.add_argument("--asr_model", default="openai/whisper-small")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = 0 if args.device == "cuda" and torch.cuda.is_available() else -1
    asr = pipeline(
        "automatic-speech-recognition",
        model=args.asr_model,
        device=device,
        generate_kwargs={"language": "fa", "task": "transcribe"},
    )
    spk_model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

    df = pd.read_csv(args.manifest_csv)
    required = {"system_id", "audio_path", "ref_text", "ref_speaker_wav"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    rows = []
    for row in df.itertuples(index=False):
        audio_path = str(row.audio_path)
        ref_text = str(row.ref_text)
        ref_speaker_wav = str(row.ref_speaker_wav)

        hyp = asr(audio_path)["text"].strip()
        row_wer = wer(ref_text, hyp)
        row_cer = cer(ref_text, hyp)

        gen_wav = load_audio_16k(audio_path)
        ref_wav = load_audio_16k(ref_speaker_wav)
        emb_gen = spk_model.encode_batch(gen_wav).squeeze().detach().cpu()
        emb_ref = spk_model.encode_batch(ref_wav).squeeze().detach().cpu()
        spk_cos = cosine_sim(emb_gen, emb_ref)

        rows.append(
            {
                "system_id": row.system_id,
                "audio_path": audio_path,
                "ref_text": ref_text,
                "asr_text": hyp,
                "wer": row_wer,
                "cer": row_cer,
                "speaker_cosine": spk_cos,
                "duration_sec": duration_seconds(audio_path),
            }
        )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.output_csv, index=False)

    summary = out_df.groupby("system_id", as_index=False).agg(
        sample_count=("audio_path", "count"),
        mean_wer=("wer", "mean"),
        mean_cer=("cer", "mean"),
        mean_speaker_cosine=("speaker_cosine", "mean"),
        mean_duration_sec=("duration_sec", "mean"),
    )
    summary["similarity_percent"] = (summary["mean_speaker_cosine"].clip(0, 1) * 100.0).round(2)
    summary.to_csv(args.output_summary_csv, index=False)

    print(f"Saved sample-level metrics: {Path(args.output_csv).resolve()}")
    print(f"Saved system-level summary: {Path(args.output_summary_csv).resolve()}")


if __name__ == "__main__":
    main()
