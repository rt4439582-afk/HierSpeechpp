# راهنمای کامل، مرحله‌به‌مرحله و عملی HierSpeech++ برای فاین‌تیون فارسی + مقایسه با مدل‌های دیگر

این سند طوری نوشته شده که بتوانید:
1) خیلی راحت TTV-v1 را روی دادهٔ فارسی فاین‌تیون کنید.
2) خروجی HierSpeech++ را با TTSهای دیگر از نظر «شباهت گوینده» و «دقت متن» مقایسه کنید.

---

## 0) واقعیت مهم این ریپو (قبل از شروع)

- کد آموزشی منتشرشده در این ریپو برای **TTV-v1** است.
- کد آموزش `Hierarchical Speech Synthesizer` منتشر نشده و در inference از checkpoint آماده استفاده می‌شود.
- مسیر پیشنهادی عملی:
  - TTV را با دادهٔ فارسی خودتان آموزش دهید.
  - در inference، `--ckpt_text2w2v` را با مدل فارسی خودتان جایگزین کنید.

---

## 1) معماری کلی پروژه (ساده ولی دقیق)

### 1.1 اجزای اصلی
- **TTV (Text-to-Vec)**: متن را به نمایش self-supervised + F0 نگاشت می‌کند.
- **Hierarchical Speech Synthesizer**: از نمایش تولیدشده + style prompt، گفتار نهایی 16k می‌سازد.
- **SpeechSR**: اگر لازم باشد خروجی را 24k/48k بالا می‌برد.

### 1.2 فایل‌های کلیدی داخل ریپو
- آموزش TTV: `train_ttv_v1.py`
- دیتالودر TTV: `ttv_v1/data_loader.py`
- کانفیگ پایه: `ttv_v1/config.json`
- پیش‌پردازش قدیمی: `ttv_v1/preprocessing/*.py`
- پیش‌پردازش عمومی اضافه‌شده: `ttv_v1/preprocessing/*_generic.py`
- اسکریپت end-to-end اضافه‌شده: `scripts/run_fa_ttv_finetune.sh`
- اسکریپت ارزیابی مقایسه‌ای اضافه‌شده: `scripts/eval_tts_similarity.py`

---

## 2) ساختار دیتاست استاندارد پیشنهادی

```text
/path/to/your_dataset
├── wav16k
│   ├── spk001/utt0001.wav
│   ├── spk001/utt0002.wav
│   └── ...
└── text
    ├── spk001/utt0001.txt
    ├── spk001/utt0002.txt
    └── ...
```

قانون طلایی:
- مسیر نسبی wav و txt باید یکسان باشد (فقط پسوند فرق دارد).
- همه wavها تک‌کاناله و 16k باشند (یا قبلش resample کنید).

---

## 3) نصب محیط

```bash
cd /workspace/HierSpeechpp
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
pip install phonemizer unidecode pandas jiwer librosa speechbrain transformers sentencepiece
sudo apt-get update && sudo apt-get install -y espeak-ng
```

---

## 4) سریع‌ترین مسیر فاین‌تیون فارسی (یک دستور)

```bash
bash scripts/run_fa_ttv_finetune.sh \
  --wav_dir /path/to/your_dataset/wav16k \
  --text_dir /path/to/your_dataset/text \
  --exp_name ttv_fa_v1 \
  --cleaner transliteration_cleaners \
  --device cuda
```

این اسکریپت به‌ترتیب:
1. استخراج W2V
2. استخراج F0
3. استخراج Token
4. ساخت filelist
5. ساخت config خودکار
6. شروع آموزش

---

## 5) اگر بخواهید مرحله‌به‌مرحله دستی اجرا کنید

### 5.1 استخراج W2V
```bash
python ttv_v1/preprocessing/extract_w2v_generic.py \
  --input_wav_dir /path/to/your_dataset/wav16k \
  --output_w2v_dir /workspace/HierSpeechpp/datasets/fa/w2v \
  --device cuda
```

### 5.2 استخراج F0
```bash
python ttv_v1/preprocessing/extract_f0_generic.py \
  --input_wav_dir /path/to/your_dataset/wav16k \
  --output_f0_dir /workspace/HierSpeechpp/datasets/fa/f0
```

### 5.3 استخراج Token
```bash
python ttv_v1/preprocessing/extract_token_generic.py \
  --input_text_dir /path/to/your_dataset/text \
  --output_token_dir /workspace/HierSpeechpp/datasets/fa/token \
  --cleaner transliteration_cleaners
```

### 5.4 ساخت filelist
```bash
python ttv_v1/preprocessing/prepare_filelist_generic.py \
  --wav_dir /path/to/your_dataset/wav16k \
  --f0_dir /workspace/HierSpeechpp/datasets/fa/f0 \
  --token_dir /workspace/HierSpeechpp/datasets/fa/token \
  --w2v_dir /workspace/HierSpeechpp/datasets/fa/w2v \
  --output_dir /workspace/HierSpeechpp/filelists/fa
```

### 5.5 کانفیگ
- قالب آماده: `ttv_v1/config_fa_template.json`
- در صورت نیاز کپی بگیرید و تنظیم کنید.

### 5.6 آموزش
```bash
CUDA_VISIBLE_DEVICES=0 python train_ttv_v1.py \
  -c ttv_v1/config_fa_template.json \
  -m ttv_fa_v1
```

---

## 6) اتصال مدل فارسی TTV به inference

```bash
CUDA_VISIBLE_DEVICES=0 python inference.py \
  --ckpt "logs/hierspeechpp_eng_kor/hierspeechpp_v1.1_ckpt.pth" \
  --ckpt_text2w2v "logs/ttv_fa_v1/G_XXXXXX.pth" \
  --output_dir "tts_results_fa" \
  --noise_scale_vc "0.333" \
  --noise_scale_ttv "0.333" \
  --denoise_ratio "0"
```

---

## 7) مقایسه با TTSهای دیگر (درصد شباهت)

## 7.1 ساخت manifest مقایسه
یک CSV بسازید: `compare_manifest.csv`

```csv
system_id,audio_path,ref_text,ref_speaker_wav
hierspeechpp,results/hspp_001.wav,سلام حال شما چطور است,/path/ref_spk.wav
xtts,results/xtts_001.wav,سلام حال شما چطور است,/path/ref_spk.wav
vits,results/vits_001.wav,سلام حال شما چطور است,/path/ref_spk.wav
```

- `system_id`: نام مدل
- `audio_path`: خروجی هر سیستم
- `ref_text`: متن صحیح
- `ref_speaker_wav`: صدای مرجع گوینده برای سنجش شباهت

## 7.2 اجرای ارزیابی
```bash
python scripts/eval_tts_similarity.py \
  --manifest_csv compare_manifest.csv \
  --output_csv eval_results.csv \
  --output_summary_csv eval_summary.csv \
  --asr_model openai/whisper-small \
  --device cuda
```

## 7.3 خروجی معیارها
- `mean_speaker_cosine`: شباهت گوینده (0 تا 1)
- `similarity_percent`: همان شباهت به درصد (0 تا 100)
- `mean_wer`, `mean_cer`: خطای متن

برداشت ساده:
- شباهت بیشتر بهتر.
- WER/CER کمتر بهتر.

---

## 8) چک‌لیست عملی برای نتیجه بهتر

1. قبل از آموزش، 200 نمونه را دستی گوش کنید.
2. متن‌ها را normalize کنید (اعداد، نیم‌فاصله، علائم).
3. ابتدا با subset کوچک dry-run بگیرید.
4. بعد از هر checkpoint مهم، خروجی بگیرید و با اسکریپت ارزیابی مقایسه کنید.
5. بهترین checkpoint را بر اساس trade-off «شباهت گوینده + WER/CER» انتخاب کنید.

---

## 9) خطاهای رایج

- OOM: batch size را کم کنید.
- ASR فارسی ضعیف: مدل ASR قوی‌تر انتخاب کنید.
- Speaker similarity پایین: مرجع گوینده را تمیزتر و هم‌سبک‌تر انتخاب کنید.
- تلفظ بد: cleaner را تغییر دهید یا normalizer متنی اضافه کنید.

---

## 10) توصیه حرفه‌ای برای مقایسه منصفانه

برای مقایسه با مدل‌های دیگر:
- دقیقاً یک متن یکسان بدهید.
- دقیقاً یک مرجع گوینده یکسان بدهید.
- loudness را normalize کنید.
- تعداد نمونه‌ها را بالا نگه دارید (مثلاً 100+ جمله).

این‌طوری درصد شباهت شما قابل اتکا می‌شود.


> برای آموزش ترتیبی از checkpoint با دیتای جدید، به `docs/FA_CONTINUAL_TRAINING_PLAN.md` مراجعه کنید.
