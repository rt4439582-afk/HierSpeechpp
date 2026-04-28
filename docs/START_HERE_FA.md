# START HERE (فارسی) — اجرای دقیق از اول تا انتها

این راهنما فقط 3 کار از شما می‌خواهد:
1) ساخت workspace
2) کپی دیتا در مسیر درست
3) اجرای stageها

> نکته مهم درباره خط قرمز/سبز: آن‌ها فقط **نمای diff** هستند (حالت مقایسه)، نه فایل واقعی.
> برای کپی فایل، همیشه از خود فایل در ریپو کپی کنید (Open File/Raw)، نه از پنجره Diff.

---

## قدم 0) پیش‌نیاز سریع

```bash
cd /workspace/HierSpeechpp
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install phonemizer unidecode pandas jiwer librosa speechbrain transformers sentencepiece
sudo apt-get update && sudo apt-get install -y espeak-ng
```

---

## قدم 1) ساخت workspace یکپارچه

```bash
cd /workspace/HierSpeechpp
bash scripts/setup_fa_workspace.sh --workspace /workspace/HierSpeechpp/fa_workspace
```

بعد از این دستور، فقط با این مسیر کار کنید:
- `/workspace/HierSpeechpp/fa_workspace`

---

## قدم 2) کپی دیتای stage01

- wav ها:
  `/workspace/HierSpeechpp/fa_workspace/data/stage01/wav16k`
- txt ها:
  `/workspace/HierSpeechpp/fa_workspace/data/stage01/text`

قوانین مهم:
- wav و txt باید همنام باشند.
- wav باید mono, 16k باشد.
- txt باید خط اول غیرخالی داشته باشد.

---

## قدم 3) اجرای آموزش اولیه

```bash
bash /workspace/HierSpeechpp/fa_workspace/scripts/run_stage01.sh
```

این دستور خودش انجام می‌دهد:
- validate دیتا
- preprocess (w2v/f0/token)
- ساخت filelist
- train TTV
- تولید next_steps برای inference

---

## قدم 4) ادامه آموزش با دیتای جدید

### Stage02
```bash
# دیتا را بریز داخل:
# /workspace/HierSpeechpp/fa_workspace/data/stage02/wav16k
# /workspace/HierSpeechpp/fa_workspace/data/stage02/text
bash /workspace/HierSpeechpp/fa_workspace/scripts/run_stage02.sh
```

### Stage03
```bash
# دیتا را بریز داخل:
# /workspace/HierSpeechpp/fa_workspace/data/stage03/wav16k
# /workspace/HierSpeechpp/fa_workspace/data/stage03/text
bash /workspace/HierSpeechpp/fa_workspace/scripts/run_stage03.sh
```

---

## قدم 5) inference

فایل زیر را باز کنید و همان دستور داخلش را اجرا کنید:

```bash
/workspace/HierSpeechpp/fa_workspace/reports/stage01/next_steps.txt
```

(در stage02/stage03 هم فایل مشابه داخل پوشه report همان stage ساخته می‌شود.)

---

## نکته شفاف

- این ریپو train کامل Synthesizer را ندارد.
- مسیر عملی و صحیح: train/fine-tune **TTV-v1** + inference با checkpoint منتشرشده Synthesizer.



> برای تست سریع در Lightning.ai با دیتای کم، این فایل را ببینید: `docs/LIGHTNING_AI_QUICKSTART_FA.md`.
