# نقشه کامل فایل‌های ریپو برای بومی‌سازی فارسی (چه چیزی هست / چه چیزی نیست)

این سند دقیقاً جواب می‌دهد:
- کدام کدهای آموزش در مخزن وجود دارد؟
- برای فارسی باید از کدام فایل‌ها استفاده کنید؟
- مسیر مطمئن (کم‌ریسک) چیست؟

---

## 1) آیا کد آموزش همه اجزا در مخزن هست؟

### موجود است
- **آموزش TTV-v1**
  - `train_ttv_v1.py`
  - `ttv_v1/data_loader.py`
  - `ttv_v1/config.json`

### موجود نیست
- **کد آموزش Hierarchical Speech Synthesizer** منتشر نشده.
- نتیجه: مسیر عملی فارسی در این ریپو = آموزش/فاین‌تیون TTV + استفاده از checkpoint آماده Synthesizer در inference.

---

## 2) فایل‌های کلیدی که باید استفاده کنید

### مسیر اصلی یک‌دستوری
- `scripts/run_fa_master_pipeline.sh`  ← ورودی اصلی
- `scripts/validate_fa_dataset.py`
- `scripts/run_fa_ttv_finetune.sh`
- `scripts/continue_fa_ttv_from_checkpoint.sh`

### پیش‌پردازش
- `ttv_v1/preprocessing/extract_w2v_generic.py`
- `ttv_v1/preprocessing/extract_f0_generic.py`
- `ttv_v1/preprocessing/extract_token_generic.py`
- `ttv_v1/preprocessing/prepare_filelist_generic.py`
- `scripts/merge_filelists.py`

### ارزیابی
- `scripts/eval_tts_similarity.py`

### راهنماها
- `docs/FA_ZERO_TO_HERO_RUNBOOK.md`
- `docs/FA_CONTINUAL_TRAINING_PLAN.md`
- `docs/FA_FULL_STACK_GUIDE.md`

---

## 3) پاسخ دقیق به سؤال «از آخرین checkpoint زبان اصلی ادامه بدهم یا نه؟»

## مسیر مطمئن (پیشنهاد اصلی)
- با همین pipeline از دیتای فارسی خودتان train کنید (stage01 → stage02 → ...).
- این مسیر با کد موجود ریپو کاملاً سازگار و کم‌ریسک است.

## مسیر اختیاری (ریسک‌دار / فقط اگر وقت اضافه دارید)
- warm-start از checkpoint خارجی TTV انگلیسی ممکن است مفید باشد،
  اما فرمت checkpointها همیشه با `train_ttv_v1.py` سازگار نیست.
- چون سازگاری 100٪ تضمین‌شده نیست، این مسیر را در pipeline اصلی قرار ندادیم.

---

## 4) چالش‌های واقعی که باید از قبل بدانید

1. اگر wavها 16k mono نباشند، کیفیت/پایداری بد می‌شود.
2. اگر متن و wav mismatch باشند، آموزش خراب می‌شود.
3. کیفیت zero-shot به تنوع speaker وابسته است (چندگوینده لازم است).
4. کد inference این ریپو برای متن فارسی نیاز به preprocess متن خوب دارد.

---

## 5) یک مسیر واحد آماده برای شما

برای اینکه همه چیز «در یک مسیر» آماده باشد، این اسکریپت را اجرا کنید:

```bash
bash scripts/setup_fa_workspace.sh --workspace /workspace/HierSpeechpp/fa_workspace
```

بعد فقط دیتا را داخل:
- `/workspace/HierSpeechpp/fa_workspace/data/stage01/wav16k`
- `/workspace/HierSpeechpp/fa_workspace/data/stage01/text`

بریزید و اجرا کنید:

```bash
bash /workspace/HierSpeechpp/fa_workspace/scripts/run_stage01.sh
```

ادامه با stage02 و stage03 هم اسکریپت آماده دارد.

