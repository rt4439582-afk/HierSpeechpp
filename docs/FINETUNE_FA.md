# راهنمای فاین‌تیون HierSpeech++ برای دادهٔ فارسی (Persian/Farsi)

این راهنما روی کد فعلی همین ریپو نوشته شده و روی **TTV-v1** تمرکز دارد (بخش قابل آموزش منتشرشده).

> نکته مهم: در این ریپو، کد آموزش «Hierarchical Speech Synthesizer» هنوز منتشر نشده و فاین‌تیون مستقیم کل پایپ‌لاین ممکن نیست. مسیر رسمی فعلاً فاین‌تیون TTV است و سپس استفاده از چک‌پوینت آمادهٔ Synthesizer در inference.

---

## 1) ساختار پوشه‌های پیشنهادی

```bash
/workspace/HierSpeechpp
└── datasets
    └── fa
        ├── wav16k
        │   ├── spk001
        │   │   ├── utt0001.wav
        │   │   └── ...
        │   └── spk002
        ├── text
        │   ├── spk001
        │   │   ├── utt0001.txt
        │   │   └── ...
        │   └── spk002
        ├── f0
        ├── w2v
        └── token
```

**الزام نام‌گذاری فایل‌ها:** برای هر wav باید txt همنام وجود داشته باشد (مثل `utt0001.wav` و `utt0001.txt`).

---

## 2) پیش‌نیازها

```bash
cd /workspace/HierSpeechpp
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
pip install phonemizer unidecode
sudo apt-get update && sudo apt-get install -y espeak-ng
```

---

## 3) استخراج ویژگی‌ها (w2v / f0 / token)

### 3.1) w2v (از MMS)

```bash
cd /workspace/HierSpeechpp/ttv_v1/preprocessing

python extract_w2v.py \
  -i /workspace/HierSpeechpp/datasets/fa/wav16k
```

پس از اجرا، در صورت نیاز مسیر خروجی را از `LibriTTS_w2v` به مسیر دلخواه خود تغییر دهید (چون اسکریپت اصلی بر اساس رشتهٔ Libri نوشته شده).

### 3.2) F0

```bash
python extract_f0.py \
  -i /workspace/HierSpeechpp/datasets/fa/wav16k
```

### 3.3) Token متنی برای فارسی

اسکریپت اصلی `extract_token.py` به `english_cleaners2` و مسیر Libri وابسته است.
برای فارسی از کد زیر استفاده کنید:

```python
# save as: /workspace/HierSpeechpp/ttv_v1/preprocessing/extract_token_fa.py
import os
import glob
import argparse
import torch
from tqdm import tqdm
from text import text_to_sequence


def main(args):
    txts = sorted(glob.glob(os.path.join(args.input_dir, '**/*.txt'), recursive=True))
    print('txt num:', len(txts))

    for txt_path in tqdm(txts):
        rel = os.path.relpath(txt_path, args.input_dir)
        out_path = os.path.join(args.output_dir, rel).replace('.txt', '.pt')
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        with open(txt_path, 'r', encoding='utf-8') as f:
            text = f.readline().strip()

        token = text_to_sequence(text, [args.cleaner])
        torch.save(torch.LongTensor(token), out_path)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('-i', '--input_dir', required=True)
    p.add_argument('-o', '--output_dir', required=True)
    p.add_argument('--cleaner', default='transliteration_cleaners')
    args = p.parse_args()
    main(args)
```

اجرا:

```bash
cd /workspace/HierSpeechpp/ttv_v1/preprocessing
python extract_token_fa.py \
  -i /workspace/HierSpeechpp/datasets/fa/text \
  -o /workspace/HierSpeechpp/datasets/fa/token \
  --cleaner transliteration_cleaners
```

> اگر بخواهید IPA دقیق‌تر داشته باشید، می‌توانید cleaner اختصاصی فارسی اضافه کنید.

---

## 4) ساخت filelist

به دلیل فرض مسیر `16k -> f0/token/w2v` در `prepare_filelist.py`، مسیر پیشنهادی همین است.

```bash
mkdir -p /workspace/HierSpeechpp/filelists/fa

python /workspace/HierSpeechpp/ttv_v1/preprocessing/prepare_filelist.py \
  -i /workspace/HierSpeechpp/datasets/fa/wav16k \
  -o /workspace/HierSpeechpp/filelists/fa
```

فایل‌های خروجی:

- `train_wav.txt`
- `train_f0.txt`
- `train_token.txt`
- `train_w2v.txt`

اگر خروجی در مسیر دیگری ساخته شد، آن‌ها را به `filelists/fa/` منتقل کنید.

---

## 5) ساخت config مخصوص فارسی

یک کپی از کانفیگ بگیرید:

```bash
cp /workspace/HierSpeechpp/ttv_v1/config.json /workspace/HierSpeechpp/ttv_v1/config_fa.json
```

مقادیر مهمی که باید تغییر دهید:

```json
{
  "data": {
    "train_filelist_path": "filelists/fa/train_wav.txt",
    "test_filelist_path": "filelists/fa/train_wav.txt",
    "text_cleaners": ["transliteration_cleaners"],
    "sampling_rate": 16000
  },
  "train": {
    "batch_size": 16,
    "learning_rate": 0.0002,
    "epochs": 20000
  }
}
```

> برای سرعت/پایداری، ابتدا با `batch_size=8` و یک subset کوچک تست کنید.

---

## 6) آموزش (Fine-tune TTV)

```bash
cd /workspace/HierSpeechpp
CUDA_VISIBLE_DEVICES=0 python train_ttv_v1.py \
  -c ttv_v1/config_fa.json \
  -m ttv_fa_v1
```

چک‌پوینت‌ها در:

```bash
/workspace/HierSpeechpp/logs/ttv_fa_v1
```

---

## 7) استفاده در inference

در زمان inference، چک‌پوینت TTV فارسی را جایگزین `--ckpt_text2w2v` کنید:

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

## 8) نکات مهم برای کیفیت فارسی

1. **Sampling Rate** همه داده‌ها را 16k نگه دارید.
2. متن‌ها را normalize کنید (نیم‌فاصله، اعداد فارسی/انگلیسی، علائم).
3. utterance کوتاه/بلند غیرعادی را حذف کنید (اسکریپت filelist همین کار را تا حدی می‌کند).
4. اگر خروجی تلفظ ضعیف بود:
   - cleaner را عوض کنید (`basic_cleaners` یا transliteration سفارشی)
   - symbols اختصاصی فارسی/IPA اضافه کنید.
5. برای شروع، یک مدل پایهٔ TTV انگلیسی را warm-start کنید (در صورت سازگاری shape پارامترها).

---

## 9) عیب‌یابی رایج

- **خطای `espeak-ng not found`**: نصب ناقص espeak-ng.
- **خطای مسیرهای LibriTTS**: اسکریپت‌های preprocessing اصلی hard-coded هستند؛ در صورت نیاز مستقیم در کد replace را فارسی‌سازی کنید.
- **OOM روی GPU**: `batch_size` را کم کنید و `segment_size` را کاهش دهید.
- **عدم هم‌خوانی طول text/audio**: مشکل در tokenize یا متن خام؛ اول روی subset دیباگ بگیرید.



> نسخه کامل‌تر و عملیاتی‌تر این راهنما را در `docs/FA_FULL_STACK_GUIDE.md` ببینید.
