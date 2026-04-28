# مسیر کامل آموزش ترتیبی (Continual) برای HierSpeech++/TTV روی فارسی

این سند برای سناریوی شماست:
- یک بار مدل را با فارسی فاین‌تیون کرده‌اید.
- حالا می‌خواهید مرحله‌به‌مرحله دیتای جدید اضافه کنید و ببینید آیا کیفیت بهتر می‌شود یا نه.

---

## 1) ایده اصلی مسیر ترتیبی

در این ریپو، اگر با همان `-m <exp_name>` دوباره `train_ttv_v1.py` را اجرا کنید،
آخرین checkpoint همان پوشهٔ `logs/<exp_name>` لود می‌شود و آموزش ادامه پیدا می‌کند.

بنابراین برای آموزش ترتیبی:
1. دیتای جدید stage-by-stage اضافه می‌کنیم.
2. filelist تجمعی می‌سازیم (cumulative).
3. دوباره همان experiment را اجرا می‌کنیم تا از checkpoint قبلی ادامه دهد.

---

## 2) ساختار پوشهٔ پیشنهادی برای Stageها

```text
/workspace/HierSpeechpp
├── datasets/fa/stages/
│   ├── stage01/
│   │   ├── wav16k  (دیتای خام این مرحله)
│   │   ├── text    (متن خام این مرحله)
│   │   ├── w2v
│   │   ├── f0
│   │   └── token
│   ├── stage02/
│   └── stage03/
└── filelists/fa/
    ├── stages/stage01/*.txt
    ├── stages/stage02/*.txt
    └── cumulative/*.txt
```

نکته: هر stage را جدا نگه دارید تا آزمایش‌های علمی شما بازتولیدپذیر بماند.

---

## 3) اجرای مرحله اول (Baseline)

می‌توانید از اسکریپت end-to-end اولیه استفاده کنید:

```bash
bash scripts/run_fa_ttv_finetune.sh \
  --wav_dir /path/stage01/wav16k \
  --text_dir /path/stage01/text \
  --exp_name ttv_fa_v1
```

---

## 4) ادامه از checkpoint با دیتای جدید (Stage2, Stage3, ...)

برای هر stage جدید:

```bash
bash scripts/continue_fa_ttv_from_checkpoint.sh \
  --exp_name ttv_fa_v1 \
  --stage_id stage02 \
  --new_wav_dir /path/stage02/wav16k \
  --new_text_dir /path/stage02/text \
  --cleaner transliteration_cleaners \
  --device cuda
```

این اسکریپت دقیقاً این کارها را می‌کند:
1. استخراج w2v/f0/token برای stage جدید.
2. ساخت filelist مخصوص stage.
3. merge filelist مرحلهٔ جدید با filelist تجمعی قبلی.
4. ساخت config تجمعی.
5. ادامه training روی همان `logs/<exp_name>`.

---

## 5) پروتکل ارزیابی برای اینکه بفهمید «دیتای بیشتر بهتر شد یا نه»

بعد از هر stage، مدل را روی یک eval set ثابت ارزیابی کنید (eval set را تغییر ندهید):

1. مجموعه جملات ثابت (مثلاً 200 جمله فارسی)
2. مجموعه speaker prompt ثابت
3. تولید خروجی با checkpoint مرحله‌ای
4. مقایسه با اسکریپت:

```bash
python scripts/eval_tts_similarity.py \
  --manifest_csv compare_manifest_stage02.csv \
  --output_csv eval_stage02.csv \
  --output_summary_csv eval_stage02_summary.csv
```

شاخص‌های اصلی:
- `similarity_percent` (شباهت گوینده)
- `mean_wer`, `mean_cer` (درست‌خوانی متن)

اگر stage جدید WER/CER را بدتر کرد اما speaker similarity بهتر شد، یعنی نیاز به تنظیم تعادل دارید
(مثلاً پاکسازی متن، تعادل speakerها، یا کاهش نرخ یادگیری).

---

## 6) دیتا برای Zero-shot speaker باید چه شکلی باشد؟

### جواب کوتاه
- بله، برای توانایی zero-shot بهتر، **چندگوینده‌ای (multi-speaker)** لازم و بسیار مهم است.

### چرا؟
- zero-shot یعنی مدل باید از گوینده‌های ندیده هم style را تعمیم بدهد.
- اگر داده تک‌گوینده یا کم‌تنوع باشد، مدل speaker space ضعیف می‌سازد و تعمیم کم می‌شود.

### توصیهٔ دیتاست
1. **تعداد speaker**: حداقل 50+ (بهتر: 100+)
2. **تنوع جنسیت/سن/لهجه**: بالا
3. **ساعت به‌ازای هر speaker**: یکنواخت نسبی (مثلاً 10 تا 60 دقیقه)
4. **کیفیت ضبط**: تا حد ممکن تمیز و هم‌سطح loudness
5. **متن**: پوشش واژگانی و آوایی خوب فارسی
6. **Prompt length** برای inference: 3 تا 10 ثانیه صدای تمیز

---

## 7) آیا بومی‌سازی برای فارسی ارزش علمی/دانشگاهی دارد؟

بله، کاملاً.

### ارزش علمی
1. فارسی نسبت به انگلیسی منابع کمتری دارد؛ نتایج شما gap پژوهشی را پوشش می‌دهد.
2. رفتار zero-shot speaker بین زبان‌ها یکسان نیست؛ به توزیع آوایی/وزنی/داده‌ای وابسته است.
3. می‌توانید نشان دهید انتقال از مدل چندزبانه به فارسی چه مقدار حفظ/افت در speaker similarity دارد.
4. موضوع مناسب پایان‌نامه/مقاله است اگر پروتکل ارزیابی دقیق و baseline قوی داشته باشید.

### آیا zero-shot در زبان‌های مختلف تفاوت دارد؟
- بله، معمولاً تفاوت دارد.
- علت‌ها:
  - تفاوت فونتیک و واج‌ها
  - کیفیت و تنوع دادهٔ آموزش
  - mismatch بین زبان pretrain و زبان target
  - کیفیت tokenizer/cleaner و نرمال‌سازی متن

پس «انتظار عملکرد یکسان در همه زبان‌ها» علمی نیست؛ باید empirical ارزیابی شود.

---

## 8) طراحی آزمایش دانشگاهی پیشنهادی

سه نقطهٔ گزارش داشته باشید:
- **Stage01**: دیتای پایه
- **Stage02**: دیتای پایه + دیتای جدید 1
- **Stage03**: دیتای پایه + دیتای جدید 1 + دیتای جدید 2

برای هر stage:
1. یک checkpoint مشخص انتخاب کنید.
2. روی eval set ثابت metrics بگیرید.
3. جدول روند بسازید (trend):
   - similarity_percent ↑ ؟
   - WER/CER ↓ ؟

این دقیقاً نشان می‌دهد دیتای بیشتر کمک کرده یا خیر.

---

## 9) دستورهای اصلی آماده (Copy/Paste)

### Stage01 (آموزش اولیه)
```bash
bash scripts/run_fa_ttv_finetune.sh \
  --wav_dir /data/fa/stage01/wav16k \
  --text_dir /data/fa/stage01/text \
  --exp_name ttv_fa_v1
```

### Stage02 (ادامه از checkpoint)
```bash
bash scripts/continue_fa_ttv_from_checkpoint.sh \
  --exp_name ttv_fa_v1 \
  --stage_id stage02 \
  --new_wav_dir /data/fa/stage02/wav16k \
  --new_text_dir /data/fa/stage02/text
```

### Stage03 (ادامه از checkpoint)
```bash
bash scripts/continue_fa_ttv_from_checkpoint.sh \
  --exp_name ttv_fa_v1 \
  --stage_id stage03 \
  --new_wav_dir /data/fa/stage03/wav16k \
  --new_text_dir /data/fa/stage03/text
```



> برای اجرای کاملاً قدم‌به‌قدم با یک دستور اصلی، `docs/FA_ZERO_TO_HERO_RUNBOOK.md` را ببینید.
