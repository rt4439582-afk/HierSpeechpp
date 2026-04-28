# چطور بفهمم مدل فعلی‌ام با چه مسیر آموزشی ساخته شده و چطور ادامه بدهم؟

اگر مدل شما fork/تغییریافته است، قبل از ادامه training این 4 مرحله را انجام بده:

## 1) checkpoint را inspect کن

```bash
cd /workspace/HierSpeechpp
python scripts/inspect_ttv_checkpoint.py \
  --ckpt /path/to/your_checkpoint.pth \
  --logs_root logs
```

این اسکریپت تشخیص می‌دهد checkpoint شما:
- `train_checkpoint` است (قابل ادامه مستقیم)
- یا `state_dict` است (معمولاً فقط inference)

## 2) اگر `train_checkpoint` بود

همان experiment name را با `-m` ادامه بده:

```bash
CUDA_VISIBLE_DEVICES=0 python train_ttv_v1.py -c /path/to/config.json -m <exp_name>
```

## 3) اگر `state_dict` بود

ادامه مستقیم تضمینی نیست (optimizer state/iteration ندارد). مسیر امن:
- از pipeline stage01/stage02 این ریپو استفاده کن.
- checkpointهای train-style داخل `logs/<exp_name>/G_*.pth` بساز.
- ادامه را از همان‌ها انجام بده.

## 4) برای مدل سفارشی (مثل tanha/reava_tts)

- چون fork است، اول ساختار config + cleaners + checkpoint format را با این اسکریپت بررسی کن.
- بدون این بررسی، resume ممکن است خراب شود.



> برای تشخیص اینکه پروژه سفارشی‌تان واقعاً HierSpeech++ هست یا نه، `docs/FA_VERIFY_CUSTOM_MODEL_IDENTITY.md` را اجرا کنید.
