"""
آماده‌سازی چک‌پوینت warm-start برای fine-tune فارسی (پیشنهاد اول).

این اسکریپت یک چک‌پوینت pretrained (مثلاً TTV انگلیسی رسمی `ttv_lt960_ckpt.pth` با
n_vocab=178) را به یک چک‌پوینت سازگار با مدل فعلی (که n_vocab آن از config خوانده
می‌شود، مثلاً 246) تبدیل می‌کند:

  - همهٔ لایه‌هایی که شکلشان یکی است از pretrained کپی می‌شوند (decoder/flow/posterior/...).
  - لایه‌هایی که شکلشان فرق دارد (`enc_p.emb`, `phoneme_classifier`) با مقداردهی اولیهٔ
    خودِ مدلِ هدف نگه داشته می‌شوند (یعنی برای الفبای فارسی از نو شروع می‌شوند).
  - خروجی به‌صورت {model, iteration:0, learning_rate, optimizer} ذخیره می‌شود تا
    `utils.load_checkpoint` در `train_ttv_v1.py` بتواند آن را بدون خطا resume کند.

چرا مهم است: اگر چک‌پوینت خام pretrained را مستقیم در پوشهٔ logs بگذارید، به‌خاطر
عدم‌تطابق ابعاد (۱۷۸ در برابر ۲۴۶) `load_checkpoint` خطا می‌دهد و `train` بی‌صدا به
آموزش «از صفر تصادفی» می‌افتد — یعنی روزها GPU هدر می‌رود. این اسکریپت آن تله را می‌بندد.

نمونه:
  python scripts/prepare_warmstart_ckpt.py \
      --config config_gpu_3090.json \
      --pretrained pretrained/ttv_lt960_ckpt.pth \
      --model_dir checkpoints/persian_main
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ttv_v1.t2w2v_transformer import SynthesizerTrn


def _strip_module(sd):
    return {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True, help="مسیر config json (همان configِ آموزش)")
    ap.add_argument("-p", "--pretrained", required=True, help="چک‌پوینت pretrained (مثلاً ttv_lt960_ckpt.pth)")
    ap.add_argument("-m", "--model_dir", required=True, help="پوشهٔ خروجی؛ G_0.pth اینجا ساخته می‌شود")
    ap.add_argument("--out_name", default="G_0.pth")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    spec_channels = cfg["data"]["filter_length"] // 2 + 1
    seg_frames = cfg["train"]["segment_size"] // cfg["data"]["hop_length"]
    model = SynthesizerTrn(spec_channels, seg_frames, **cfg["model"])
    target_sd = model.state_dict()

    ckpt = torch.load(args.pretrained, map_location="cpu")
    pre_sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    pre_sd = _strip_module(pre_sd)

    new_sd = {}
    copied, reinit = [], []
    for k, v in target_sd.items():
        if k in pre_sd and tuple(pre_sd[k].shape) == tuple(v.shape):
            new_sd[k] = pre_sd[k]
            copied.append(k)
        else:
            new_sd[k] = v  # مقداردهی اولیهٔ خودِ مدل هدف (از نو)
            reinit.append(k)

    missing, unexpected = model.load_state_dict(new_sd, strict=True)
    n_total = len(target_sd)
    print(f"کپی‌شده از pretrained: {len(copied)}/{n_total}")
    print(f"از نو init شده (عدم‌تطابق/غایب): {len(reinit)} لایه")
    for k in reinit:
        print(f"   reinit: {k}  target={tuple(target_sd[k].shape)}"
              + (f"  pretrained={tuple(pre_sd[k].shape)}" if k in pre_sd else "  (در pretrained نبود)"))

    # optimizer تازه و سازگار با همین مدل تا load_checkpoint موفق شود
    optim = torch.optim.AdamW(
        model.parameters(),
        cfg["train"]["learning_rate"],
        betas=cfg["train"].get("betas", [0.8, 0.99]),
        eps=cfg["train"].get("eps", 1e-9),
    )

    os.makedirs(args.model_dir, exist_ok=True)
    out_path = os.path.join(args.model_dir, args.out_name)
    torch.save(
        {
            "model": model.state_dict(),
            "iteration": 0,
            "learning_rate": cfg["train"]["learning_rate"],
            "optimizer": optim.state_dict(),
        },
        out_path,
    )
    print(f"\n✅ ذخیره شد: {out_path}")
    print("اکنون: python train_ttv_v1.py -c {} -m {}".format(args.config, args.model_dir))


if __name__ == "__main__":
    main()
