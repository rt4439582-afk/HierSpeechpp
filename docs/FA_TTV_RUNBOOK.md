# دستورالعمل جامع آموزش TTS فارسی (HierSpeech++ / TTV-v1)

این سند یک راهنمای عملیاتی گام‌به‌گام برای **بازآموزی و بهبود** مدل تبدیل متن‌به‌گفتار فارسی است.
هدف: یک‌بار تست کامل مسیر روی **Google Colab با دادهٔ محدود**، سپس **آموزش کامل روی GPU اجاره‌ای (~۸ روز)** بدون اتلاف وقت و بدون اشتباه.

> این سند را می‌توانید مستقیماً به ربات (Cursor / عامل ابری) بدهید. هر بخش «دستور آماده» دارد.

---

## ۰) خلاصهٔ تصمیم‌ها (TL;DR)

| سؤال شما | پاسخ کوتاه |
|----------|------------|
| آیا ربات ابری کرسر می‌تواند مسیر `C:\New folder...` را ببیند؟ | **نه.** عامل ابری در یک ماشین لینوکسِ ایزوله اجرا می‌شود و به فایل‌سیستم ویندوز شما دسترسی ندارد. باید فایل‌ها را **آپلود** یا **push/لینک دانلود** کنید (بخش ۱). |
| از checkpoint قبلی ادامه دهم یا از صفر؟ | **از صفر شروع نکنید.** از **checkpoint رسمی HierSpeech++ TTV** (یا checkpoint فارسی قبلی‌تان اگر هم‌تنظیم باشد) **fine-tune** کنید. با بودجهٔ ۸ روز روی یک GPU، آموزش از صفرِ کل معماری ریسک بالا و کیفیت پایین‌تر دارد (بخش ۲). |
| کدام G2P/توکنایزر؟ | **یکی را انتخاب و قفل کنید.** پیشنهاد برای اجرای اصلی: **espeak `fa` (`persian_cleaners`)** که خروجی IPA می‌دهد و با جدول نمادهای موجود و وزن‌های pretrained سازگار است. AvashoG2P/جدول ۲۴۶تایی فقط به‌عنوان ablation (بخش ۳). |
| پایهٔ کد؟ | شاخهٔ `main` همین‌طور **اجرا نمی‌شود** (چند مقدار هاردکد/گمشده). کار را روی شاخهٔ `codex/provide-fine-tuning-steps-for-persian-data` (یا merge آن) بنا کنید که اسکریپت‌ها و `persian_cleaners` را دارد (بخش ۴). |

**ترتیب کلی کار:** بخش ۱ (انتقال فایل‌ها) → بخش ۲ و ۳ (تصمیم) → بخش ۴ (preflight و رفع ایرادها) → بخش ۵ (تست دود Colab) → بخش ۶ (آموزش ۸ روزه) → بخش ۷ (ارزیابی) → بخش ۸ (inference).

---

## ۱) دسترسی ربات به فایل‌ها و خروجی‌های قبلی

عامل **ابری** کرسر مسیرهای ویندوزی مثل این‌ها را **نمی‌بیند**:

```
C:\New folder (3)\package_to_server\ZERO_TTS\ttv_v1\processing\logs
C:\New folder (3)\package_to_server\ttstts\rt\src\ttv_v1\processing\logs\TTV_model
```

برای اینکه ربات بتواند خروجی‌ها/چک‌پوینت‌های قبلی را ببیند، یکی از این کارها را بکنید:

1. **فایل‌های کوچک (log/config):** مستقیم در چت آپلود کنید (همان‌طور که logها و configها را آپلود کردید). ✅
2. **چک‌پوینت‌ها (`G_*.pth` که چند صد مگابایت‌اند):**
   - بهترین راه: روی **Hugging Face Hub** یا **Google Drive** بگذارید و **لینک دانلود مستقیم** بدهید تا ربات با `wget/curl` بگیرد.
   - یا با **Git LFS** به مخزن اضافه کنید (نه commit معمولی).
3. اگر از عامل **محلی** کرسر (روی همان ویندوز) استفاده می‌کنید، آن می‌تواند مسیر `C:\...` را بخواند؛ ولی عامل ابری نه.

**برای بازرسی یک چک‌پوینت قدیمی** (تعداد گام، کلیدها، تطابق با معماری فعلی) از این اسکریپت موجود استفاده کنید:

```bash
python scripts/inspect_ttv_checkpoint.py --ckpt /path/to/G_XXXX.pth
```

> نکته: قبل از تصمیم «ادامه از چک‌پوینت»، حتماً خروجی این بازرسی را ببینید تا مطمئن شوید ابعاد لایه‌ها (`enc_p.emb`, `phoneme_classifier`) با تنظیمات جدید می‌خوانند.

---

## ۲) ادامه از چک‌پوینت یا از صفر؟ (درخت تصمیم)

معماری مدل TTV (در `ttv_v1/t2w2v_transformer.py`) دو نقطهٔ وابسته به اندازهٔ واژگان دارد:
- `TextEncoder(n_vocab, ...)` → لایهٔ `emb`
- `phoneme_classifier = Conv1d(inter_channels, n_vocab, 1)` (سر CTC)

```
آیا توکنایزر/جدول نمادها را عوض می‌کنید؟
│
├─ نه (همان espeak-fa / IPA، n_vocab بدون تغییر)
│     → ✅ از چک‌پوینت رسمی HierSpeech++ TTV یا چک‌پوینت فارسی قبلی fine-tune کنید.
│       (سریع‌ترین، باکیفیت‌ترین، مناسب بودجهٔ ۸ روز)
│
└─ بله (مثلاً AvashoG2P یا جدول نماد ۲۴۶تایی سفارشی)
      → ابعاد emb و سر CTC عوض می‌شود؛ نمی‌توانید این دو لایه را مستقیم resume کنید.
        دو گزینه:
        ۱) بقیهٔ وزن‌ها را لود کنید و فقط emb + phoneme_classifier را از نو init کنید (warm-start). (متوسط)
        ۲) از صفر. (کندترین؛ فقط اگر داده بسیار زیاد و وقت کافی دارید)
```

### توصیهٔ نهایی برای شما
- صدای قبلی «بد نبود ولی عالی هم نبود» = نشانهٔ این‌که transfer از مدل پایه کار می‌کند ولی tuning/داده/هایپرها بهینه نبوده.
- پس: **توکنایزر را عوض نکنید**، از **چک‌پوینت قوی pretrained** ادامه دهید، و در عوض روی این‌ها تمرکز کنید: کیفیت/پاکسازی دادهٔ فارسی، تنظیم `c_pho`/loss، طول `segment`، و طول کافی آموزش. (بخش ۶)
- اگر AvashoG2P را برای **نوآوری پایان‌نامه** لازم دارید، آن را به‌صورت یک **آزمایش جداگانه (ablation)** اجرا کنید، نه اجرای اصلیِ تولید صدا.

### نحوهٔ ادامه از چک‌پوینت در عمل
اسکریپت آموزش، آخرین `G_*.pth` داخل `./logs/<exp_name>/` را خودکار لود می‌کند:

```bash
# چک‌پوینت پایه را اینجا بگذارید تا resume شود:
mkdir -p logs/ttv_fa_v1
cp /path/to/G_PRETRAINED.pth logs/ttv_fa_v1/G_0.pth
# سپس آموزش (بخش ۶) ادامه از همین می‌شود.
```

---

## ۳) انتخاب و قفل‌کردن توکنایزر (G2P)

در حال حاضر سه روش ناسازگار در مخزن پراکنده است (همین باعث خروجی نامرغوب می‌شود):
- `ttv_v1/preprocessing/extract_token.py` → هاردکد `english_cleaners2` ❌ (برای فارسی غلط)
- `config_fa_template.json` شاخهٔ codex → `transliteration_cleaners`
- configهای آپلودی شما → `persian_cleaners` با `n_vocab=246`

**فقط یکی را انتخاب کنید و همه‌جا یکسان کنید.** پیشنهاد:

```
text_cleaners = ["persian_cleaners"]      # espeak language=fa  → IPA
```

`persian_cleaners` در شاخهٔ `codex` این‌گونه تعریف شده (از espeak فارسی استفاده می‌کند):

```python
def persian_cleaners(text):
    text = lowercase(text)
    text = collapse_whitespace(text)
    phonemes = phonemize(text, language='fa', backend='espeak', strip=True)
    phonemes = collapse_whitespace(phonemes)
    return phonemes
```

**پیش‌نیاز سیستمی:** باید `espeak-ng` با دادهٔ زبان فارسی نصب باشد:

```bash
sudo apt-get update && sudo apt-get install -y espeak-ng
espeak-ng -v fa "سلام دنیا" --ipa   # تست سریع G2P فارسی
```

> اگر می‌خواهید نرمال‌سازی بهتر فارسی (نیم‌فاصله، عدد→حرف) داشته باشید، قبل از espeak یک مرحله با `hazm`/`parsivar` اضافه کنید؛ ولی **مهم‌تر از همه: همان cleaner را در preprocessing و inference یکسان نگه دارید.**

---

## ۴) Preflight — رفع ایرادهای حتمی کد (قبل از هر اجرا)

شاخهٔ `main` با configهای فارسی شما **بلافاصله کرش می‌کند**. این موارد باید برطرف باشند (بیشترشان روی شاخهٔ `codex` حل شده):

| # | مشکل در `main` | محل | راه‌حل |
|---|----------------|-----|--------|
| 1 | `config.data.train_data_ratio` در هیچ config نیست → AttributeError | `ttv_v1/data_loader.py:19` | روی codex با `getattr(..., 1.0)` حل شده ✅ |
| 2 | `persian_cleaners` تعریف نشده | `ttv_v1/text/cleaners.py` | روی codex اضافه شده ✅ |
| 3 | استخراج توکن با cleaner انگلیسی هاردکد | `extract_token.py` | از `extract_token_generic.py --cleaner persian_cleaners` استفاده کنید ✅ |
| 4 | `assert torch.cuda.is_available()` مانع اجرای CPU | `train_ttv_v1.py:28` | برای Colab با GPU مشکلی نیست؛ برای dry-run روی CPU باید موقتاً تغییر کند |
| 5 | `num_workers=32` هاردکد | `train_ttv_v1.py:64` | برای Colab به ۲ کاهش دهید |
| 6 | padding متن ۴۰۳/۲۰۱ هاردکد | `data_loader.py:68,71` | اگر طول متن فارسی شما بیشتر است، باید بزرگ‌تر شود |
| 7 | `n_vocab` در مدل **۱۷۸ هاردکد** است و از config خوانده نمی‌شود | `t2w2v_transformer.py:374,384` | فقط اگر جدول نماد را عوض کردید مهم است؛ با espeak-fa و جدول موجود مشکلی نیست |

**اقدام پیشنهادی:** کار را روی شاخهٔ `codex/provide-fine-tuning-steps-for-persian-data` انجام دهید (یا آن را در شاخهٔ کاری‌تان merge کنید):

```bash
git fetch origin
git checkout -b fa_train origin/codex/provide-fine-tuning-steps-for-persian-data
```

این شاخه آماده دارد: `persian_cleaners`، اسکریپت‌های `*_generic.py`، `config_fa_template.json`، و اسکریپت‌های end-to-end (`run_fa_ttv_finetune.sh`, `continue_fa_ttv_from_checkpoint.sh`, `make_smoke_subset.py`, `validate_fa_dataset.py`, `eval_tts_similarity.py`, `inspect_ttv_checkpoint.py`).

> اگر می‌خواهید من (عامل ابری) این فیکس‌ها را در یک شاخهٔ تمیز با configهای آپلودی شما یکپارچه کنم و تست دود بزنم، بگویید تا انجام دهم.

---

## ۵) فاز ۱ — تست دود کامل روی Google Colab (دادهٔ محدود)

**هدف:** اثبات این‌که کل مسیر (preprocess → filelist → train چند گام → ذخیرهٔ checkpoint → inference) بدون خطا کار می‌کند، قبل از خرج‌کردن پول GPU.

> در Colab یک GPU (T4/L4/A100) بگیرید. استخراج w2v با MMS-300m روی CPU بسیار کند است؛ با GPU سریع است.

### ۵.۱ نصب
```bash
git clone <REPO_URL> hier && cd hier
git checkout fa_train      # شاخهٔ آماده‌سازی‌شده (بخش ۴)
pip install -r requirements.txt
sudo apt-get update && sudo apt-get install -y espeak-ng
# monotonic_align را build کنید:
cd ttv_v1/monotonic_align && python setup.py build_ext --inplace && cd ../../
```

### ۵.۲ ساخت زیرمجموعهٔ کوچک (مثلاً ۲۰۰ نمونه)
```bash
python scripts/make_smoke_subset.py \
  --src_wav_dir  /content/data/wav16k \
  --src_text_dir /content/data/text \
  --dst_root     /content/smoke \
  --num_samples  200
```
> ساختار: هر فایل `*.wav` (۱۶kHz) یک `*.txt` هم‌نام با متن فارسی همان جمله داشته باشد.

### ۵.۳ استخراج ویژگی‌ها
```bash
python ttv_v1/preprocessing/extract_w2v_generic.py   --input_wav_dir /content/smoke/wav16k --output_w2v_dir /content/smoke/w2v   --device cuda
python ttv_v1/preprocessing/extract_f0_generic.py    --input_wav_dir /content/smoke/wav16k --output_f0_dir  /content/smoke/f0
python ttv_v1/preprocessing/extract_token_generic.py --input_text_dir /content/smoke/text --output_token_dir /content/smoke/token --cleaner persian_cleaners
```

### ۵.۴ ساخت filelist
```bash
python ttv_v1/preprocessing/prepare_filelist_generic.py \
  --wav_dir /content/smoke/wav16k --f0_dir /content/smoke/f0 \
  --token_dir /content/smoke/token --w2v_dir /content/smoke/w2v \
  --output_dir filelists/fa_smoke
```

### ۵.۵ اعتبارسنجی دیتاست (طول‌ها، فایل‌های خراب)
```bash
python scripts/validate_fa_dataset.py \
  --wav_dir /content/smoke/wav16k --text_dir /content/smoke/text \
  --report_csv /content/smoke/validate_report.csv \
  --summary_json /content/smoke/validate_summary.json
```

### ۵.۶ config کوچکِ تست دود
از `config_cpu_smoke` آپلودی شما به‌عنوان مبنا استفاده کنید، ولی فقط مقادیر زیر را تنظیم کنید (روی GPU Colab `allow_cpu` لازم نیست):
```jsonc
{
  "train": { "epochs": 3, "batch_size": 2, "num_workers": 2,
             "log_interval": 1, "eval_interval": 10, "save_interval": 10, "fp16_run": false },
  "data":  { "train_filelist_path": "filelists/fa_smoke/train_wav.txt",
             "test_filelist_path":  "filelists/fa_smoke/train_wav.txt",
             "text_cleaners": ["persian_cleaners"] }
}
```

### ۵.۷ اجرای چند گام آموزش
```bash
CUDA_VISIBLE_DEVICES=0 python train_ttv_v1.py -c ttv_v1/config_fa_smoke.json -m ttv_fa_smoke
```
**معیار موفقیت تست دود:**
- بدون خطا اجرا شود و چند گام جلو برود،
- مقدار loss **finite** باشد (NaN/Inf نباشد) و نسبتاً کاهش یابد،
- یک فایل `logs/ttv_fa_smoke/G_*.pth` ذخیره شود،
- (اختیاری) inference روی یک نمونه خروجی صوتی بدهد (بخش ۸).

> اگر `num_workers=32` هاردکد در `train_ttv_v1.py` روی Colab مشکل ساخت، آن را به ۲ کم کنید.

---

## ۶) فاز ۲ — آموزش کامل روی GPU اجاره‌ای (~۸ روز)

### ۶.۱ پیش از شروع (چک‌لیست ضد اتلاف وقت)
- [ ] تست دود فاز ۵ **سبز** شده باشد.
- [ ] کل دیتاست فارسی preprocess شده باشد (همان cleaner فاز ۵).
- [ ] چک‌پوینت پایه در `logs/<exp>/G_0.pth` گذاشته شده باشد (بخش ۲).
- [ ] `save_interval` معقول (مثلاً هر ۱۰۰۰۰ گام) تا در صورت قطعی، resume ممکن باشد.
- [ ] ذخیره‌سازی پایدار: `logs/` را روی دیسک ماندگار یا هر چند ساعت روی Drive/HF بک‌آپ بگیرید (ماشین اجاره‌ای ممکن است پاک شود).
- [ ] `nohup`/`tmux` تا اگر ترمینال قطع شد، آموزش ادامه یابد.

### ۶.۲ config آموزش کامل
از `config_gpu_3090` آپلودی به‌عنوان مبنا. روی GPU 24GB:
- `batch_size`: ۸ تا ۱۶ (اگر OOM شد، کم کنید و `grad_accum_steps` را زیاد کنید — توجه: استفاده از `grad_accum_steps` نیاز به وصله در حلقهٔ train دارد؛ اگر آن وصله نباشد این فیلد بی‌اثر است).
- `fp16_run: false` (پایداری؛ اگر سرعت لازم شد و GPU پشتیبانی می‌کند می‌توان bf16 را جداگانه افزود).
- `save_interval`/`eval_interval`: ۱۰۰۰۰.

### ۶.۳ اجرا (با tmux + nohup)
```bash
tmux new -s train
export CUDA_VISIBLE_DEVICES=0
nohup python train_ttv_v1.py -c ttv_v1/config_fa_full.json -m ttv_fa_full > train_full.log 2>&1 &
tail -f train_full.log
```

### ۶.۴ پایش (Monitoring)
```bash
tensorboard --logdir logs/ttv_fa_full --port 6006
```
به این‌ها نگاه کنید: `loss/g/total`, `loss/g/w2v`, `loss/g/kl`, `loss/g/f0`, `loss/g/dur`, `loss/g/ctc`.
- اگر `loss/g/ctc` پایین نمی‌آید → مشکل در alignment/توکنایزر (همان ریشهٔ کیفیت بد قبلی).
- اگر loss به‌سرعت NaN شد → lr را کم یا grad clipping را سفت‌تر کنید.

### ۶.۵ ادامه پس از قطعی
کافی است دوباره همان دستور ۶.۳ را بزنید؛ اسکریپت آخرین `logs/ttv_fa_full/G_*.pth` را لود می‌کند.

> برآورد بودجه: یک GPU 24GB روی ParsVoice/ManaTTS برای fine-tune در ۸ روز معقول است؛ **آموزش از صفرِ کل سامانه در ۸ روز قابل اتکا نیست** — دلیل دیگرِ توصیه به fine-tune.

---

## ۷) ارزیابی (مطابق پروپزال)

پس از آموزش، روی مجموعهٔ آزمون (گویندگانِ کاملاً جدا) خروجی بسازید و این معیارها را حساب کنید:

```bash
# نمونهٔ آمادهٔ مخزن برای شباهت گوینده/سبک:
python scripts/eval_tts_similarity.py --help
```

| معیار | ابزار پیشنهادی |
|-------|----------------|
| SECS (شباهت گوینده) | embeddingهای speaker encoder + cosine (در `eval_tts_similarity.py`) |
| UTMOS (کیفیت خودکار) | بستهٔ `utmos`/مدل عمومی UTMOS |
| WER/CER (رسایی) | یک ASR فارسی (مثلاً Whisper) روی خروجی و مقایسه با متن مرجع |
| MCD | کتابخانهٔ `pymcd` بین خروجی و مرجع |
| MOS/SMOS (ذهنی) | نظرسنجی انسانی با شنوندگان فارسی‌زبان |

هدف‌های پروپزال: MOS > 3.80، SMOS > 4.10، WER < 20٪، CER < 10٪.

---

## ۸) Inference نهایی (زنجیرهٔ کامل صدا)

TTV فقط متن→ویژگی w2v را می‌سازد. برای صدای نهایی، زنجیرهٔ کامل HierSpeech++ لازم است:

```
متن فارسی --(TTV-v1 شما)--> w2v + F0
        --(hierspeechpp_speechsynthesizer)--> موج ۱۶kHz
        --(SpeechSR 24k/48k)--> موج باکیفیت بالا
```

به `inference.py` و `inference.sh` مخزن نگاه کنید و چک‌پوینت TTV فارسی خود را به‌جای مدل پیش‌فرض بدهید. (وکودر/SpeechSR را معمولاً از وزن‌های رسمی استفاده می‌کنید.)

---

## ۹) چک‌لیست نهایی «بدون اشتباه»

- [ ] فایل‌های لازم به محیط ربات منتقل شده‌اند (بخش ۱) — مسیر `C:\` کار نمی‌کند.
- [ ] یک cleaner واحد در همهٔ مراحل (preprocess + inference).
- [ ] `espeak-ng` فارسی نصب و تست‌شده.
- [ ] preflight کد (بخش ۴) سبز است؛ روی شاخهٔ درست کار می‌کنید.
- [ ] تست دود Colab سبز است (loss finite + ذخیرهٔ checkpoint).
- [ ] چک‌پوینت پایه برای fine-tune آماده است.
- [ ] بک‌آپ خودکار `logs/` تنظیم شده.
- [ ] اسکریپت ارزیابی قبل از شروع آموزشِ گران، یک‌بار روی نمونه تست شده.

---

## ۱۰) واقع‌سنجی: پروپزال در برابر پیاده‌سازی فعلی

پروپزال شما بسیار جاه‌طلبانه است (CFM، VQ-VAE برای prosody، دو style-encoder، Prior-Mixup، MPNet، SpeechSR). اما کدِ این مخزن **HierSpeech++ / TTV-v1** است که:
- ✅ از قبل دارد: style encoder، normalizing flow، SpeechSR (24k/48k)، denoiser/MPNet، prosody/F0 و pitch predictor.
- ⚠️ ندارد/متفاوت است: «Conditional Flow Matching» به‌عنوان مولد آکوستیک (این‌جا flow + decoder است، نه CFM)، «VQ-VAE prosody»، و «دو style-encoder مجزا به سبک StyleTTS2».
- ⚠️ AvashoG2P و POS-tagging به‌عنوان نوآوری، در کد فعلی نیستند و با espeak جایگزین شده‌اند.

**توصیه برای دفاع پایان‌نامه:** یا (الف) ادعاهای نوآوری را با آنچه واقعاً پیاده می‌کنید هم‌تراز کنید (تمرکز روی بومی‌سازی فارسی + style encoder + G2P)، یا (ب) اجزای ادعاشده (CFM/VQ-VAE) را واقعاً پیاده و به‌صورت ablation بسنجید. ناسازگاری «ادعا در مقابل پیاده‌سازی» رایج‌ترین ایراد داوری است.

> اگر بخواهید، می‌توانم ارزیابی رسمی ۱۲ بندی پروپزال (طبق روبریک خودتان) را هم به‌صورت جداگانه بنویسم.
