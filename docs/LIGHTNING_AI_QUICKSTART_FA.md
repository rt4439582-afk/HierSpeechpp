# تست سریع مسیر فارسی روی Lightning.ai (داده محدود)

این راهنما برای smoke test است (نه آموزش نهایی):
- دیتای کم (مثلاً 100 تا 500 نمونه)
- یک GPU کوچک
- بررسی اینکه pipeline سالم اجرا می‌شود

---

## 1) ساخت Studio

1. وارد Lightning.ai شوید و یک **Studio** بسازید.
2. یک GPU ساده انتخاب کنید (T4/A10 کافی است برای smoke test).
3. ترمینال باز کنید.

---

## 2) clone و نصب

```bash
git clone https://github.com/sh-lee-prml/HierSpeechpp.git
cd HierSpeechpp
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install phonemizer unidecode pandas jiwer librosa speechbrain transformers sentencepiece
sudo apt-get update && sudo apt-get install -y espeak-ng
```

---

## 3) ساخت subset کوچک (برای سرعت آپلود/تست)

اگر دیتای کامل بزرگ است، اول subset بساز:

```bash
python scripts/make_smoke_subset.py \
  --src_wav_dir /teamspace/studios/this_studio/data/full/wav16k \
  --src_text_dir /teamspace/studios/this_studio/data/full/text \
  --dst_root /teamspace/studios/this_studio/data/smoke_stage01 \
  --num_samples 300
```

---

## 4) setup workspace

```bash
bash scripts/setup_fa_workspace.sh --workspace /teamspace/studios/this_studio/fa_workspace
```

حالا subset را کپی کنید:

```bash
rsync -av /teamspace/studios/this_studio/data/smoke_stage01/wav16k/ /teamspace/studios/this_studio/fa_workspace/data/stage01/wav16k/
rsync -av /teamspace/studios/this_studio/data/smoke_stage01/text/ /teamspace/studios/this_studio/fa_workspace/data/stage01/text/
```

---

## 5) اگر checkpoint فارسی خودتان را دارید

- checkpoint را در `logs/ttv_fa_v1/` آپلود کنید.
- اگر `G_*.pth` + `config.json` داخل این پوشه باشد، stage02/03 از آن ادامه می‌گیرند.

اگر آپلود کند است:
- فعلاً stage01 را با subset اجرا کنید تا مسیر تست شود.
- بعد checkpoint فارسی را اضافه کنید و stage02 را ادامه دهید.

---

## 6) precheck و اجرای stage01

```bash
bash scripts/fa_pretrain_checklist.sh \
  --wav_dir /teamspace/studios/this_studio/fa_workspace/data/stage01/wav16k \
  --text_dir /teamspace/studios/this_studio/fa_workspace/data/stage01/text \
  --out_dir /teamspace/studios/this_studio/fa_workspace/reports/precheck_stage01 \
  --min_pairs 100

bash /teamspace/studios/this_studio/fa_workspace/scripts/run_stage01.sh
```

---

## 7) ادامه از checkpoint فارسی (وقتی آماده شد)

دیتای stage02 را بریزید:
- `/teamspace/studios/this_studio/fa_workspace/data/stage02/wav16k`
- `/teamspace/studios/this_studio/fa_workspace/data/stage02/text`

اجرا:

```bash
bash /teamspace/studios/this_studio/fa_workspace/scripts/run_stage02.sh
```

---

## آیا کد شما واقعاً HierSpeech++ است؟

کدی که فرستادید از نظر توابع کلیدی (مثل `infer_noise_control` و `voice_conversion_noise_control`) شبیه HierSpeech++ است،
اما ساختار پکیج `tanha/reava_tts` یک fork سفارشی است، نه همین ریپوی خام.

## آیا g2p لازم است؟

- برای فارسی، بله عملاً یک text cleaner/phonemizer لازم است.
- در این ریپو، `persian_cleaners` اضافه شد که از espeak phonemizer با زبان `fa` استفاده می‌کند.

