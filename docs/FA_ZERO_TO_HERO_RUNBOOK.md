# راهنمای صفر تا صد اجرای کامل (برای کاربر مبتدی)

این فایل دقیقاً می‌گوید چه کدی را اجرا کنید تا:
1) دیتا اعتبارسنجی شود.
2) فاین‌تیون انجام شود.
3) checkpoint آماده inference شود.
4) اگر خروجی مدل‌های مختلف دارید، ارزیابی کمی هم گرفته شود.

---

## قبل از شروع (خیلی مهم)

### چیزی که **می‌توانیم کامل اتوماتیک کنیم**
- فاین‌تیون TTV-v1 با دیتای فارسی
- ادامهٔ آموزش stage-by-stage از checkpoint قبلی
- اعتبارسنجی دیتا
- ارزیابی objective (WER/CER + speaker similarity)

### چیزی که **در این ریپو قابل آموزش کامل نیست**
- آموزش خود `Hierarchical Speech Synthesizer` (کد train آن منتشر نشده)
- بنابراین در inference، از checkpoint منتشرشده Synthesizer استفاده می‌کنیم و فقط TTV را فارسی می‌کنیم.

---

## 1) ساختار دیتای شما

باید این شکلی باشد:

```text
/data/fa/stage01/
├── wav16k/
│   ├── spk001/utt0001.wav
│   └── ...
└── text/
    ├── spk001/utt0001.txt
    └── ...
```

قوانین:
- هر wav باید txt همنام داشته باشد.
- wavها mono و 16k باشند.
- هر txt در خط اول، متن همان wav را داشته باشد.

---

## 2) اجرای مرحله اول (آموزش اولیه)

```bash
cd /workspace/HierSpeechpp
bash scripts/run_fa_master_pipeline.sh \
  --mode initial \
  --exp_name ttv_fa_v1 \
  --wav_dir /data/fa/stage01/wav16k \
  --text_dir /data/fa/stage01/text \
  --cleaner transliteration_cleaners \
  --device cuda \
  --run_eval 0
```

این دستور خودش:
- دیتای شما را validate می‌کند.
- کل preprocess و train را اجرا می‌کند.
- آخر کار فایل `runs/fa_master/next_steps.txt` می‌سازد که دستور inference آماده داخلش هست.

---

## 3) ادامه آموزش از checkpoint (دیتای جدید)

برای stage02:

```bash
bash scripts/run_fa_master_pipeline.sh \
  --mode continue \
  --exp_name ttv_fa_v1 \
  --stage_id stage02 \
  --wav_dir /data/fa/stage02/wav16k \
  --text_dir /data/fa/stage02/text \
  --cleaner transliteration_cleaners \
  --device cuda \
  --run_eval 0
```

برای stage03 هم مشابه stage_id را عوض کنید.

---

## 4) گرفتن خروجی TTS

بعد از پایان pipeline:
- فایل `runs/fa_master/next_steps.txt` را باز کنید.
- همان دستور inference را عیناً اجرا کنید.

---

## 5) ارزیابی مقایسه‌ای (اختیاری)

اگر خروجی چند مدل دارید (HierSpeech++, XTTS, VITS, ...):

1. CSV بسازید (`compare_manifest.csv`) با ستون‌های:
   - `system_id`
   - `audio_path`
   - `ref_text`
   - `ref_speaker_wav`

2. pipeline را با eval اجرا کنید:

```bash
bash scripts/run_fa_master_pipeline.sh \
  --mode continue \
  --exp_name ttv_fa_v1 \
  --stage_id stage03 \
  --wav_dir /data/fa/stage03/wav16k \
  --text_dir /data/fa/stage03/text \
  --device cuda \
  --run_eval 1 \
  --eval_manifest /path/compare_manifest.csv
```

خروجی ارزیابی:
- `runs/fa_master/eval_results.csv`
- `runs/fa_master/eval_summary.csv`

---

## 6) پیشنهاد کیفیت بالا

- چندگوینده‌ای جمع کنید (50+ گوینده، بهتر 100+).
- کیفیت ضبط را تمیز نگه دارید.
- متن‌ها را normalize کنید (اعداد، نیم‌فاصله، علائم).
- eval set ثابت داشته باشید و بین stageها تغییرش ندهید.



## 7) ساخت مسیر یکپارچه آماده (یک‌بار اجرا)

```bash
bash scripts/setup_fa_workspace.sh --workspace /workspace/HierSpeechpp/fa_workspace
```

بعد از اجرا، فقط دیتا را در همان مسیر کپی کنید و اسکریپت‌های آماده stage را اجرا کنید.


> اگر می‌خواهید خیلی سریع و بدون ابهام اجرا کنید، از `docs/START_HERE_FA.md` شروع کنید.
