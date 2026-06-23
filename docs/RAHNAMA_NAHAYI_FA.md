# راهنمای نهایی و اجرایی — آموزش TTS فارسی (HierSpeech++ / TTV-v1)

**نسخه:** ۱.۱ — ۲۳ ژوئن ۲۰۲۶
**پروژهٔ واقعی شما:** `perfect_project` (شاخهٔ `fix/fa-gpu-readiness`) — توکنایزر **کاراکتری ۲۴۶ نمادی**، `n_vocab=246`.
**هدف:** ابتدا تست دود کامل روی **Google Colab** با بخشی از داده‌ها و دو چک‌پوینت Drive شما؛ سپس **آموزش کامل روی GPU اجاره‌ای (~۸ روز)**.

> این فایل، نسخهٔ تکمیل‌شدهٔ `RAHNAMA_NAHAYI_AVARZESH_TTS_FA.md` خودتان است، با افزودن **روش اجرای دقیق Colab وصل‌شده به فایل‌های Drive شما** و تصحیح یک نکتهٔ مهم دربارهٔ کدام مخزن.

---

## ۰) مهم‌ترین نکته قبل از هر کاری: کدام کد را روی Colab اجرا کنید؟

دو مخزن کاملاً متفاوت در میان است؛ اشتباه‌گرفتن این دو، علت اصلی شکست تست Colab قبلی شماست:

| مخزن | توکنایزر | n_vocab | مناسب شما؟ |
|------|----------|---------|-----------|
| **`perfect_project`** (لوکال شما، شاخهٔ `fix/fa-gpu-readiness`) | کاراکتری فارسی + hazm اختیاری | **۲۴۶** | ✅ **بله — همین را اجرا کنید** |
| `rt4439582-afk/HierSpeechpp` شاخهٔ `codex/...` (روی GitHub) | espeak `fa` → IPA | ۱۷۸ (هاردکد) | ❌ نه — با داده/چک‌پوینت شما ناسازگار |

**اثبات زنده (تست آفلاین روی همین کد GitHub):** مدل شاخهٔ GitHub با وجود `n_vocab=246` در config، باز هم لایهٔ متن را **۱۷۸** می‌سازد (هاردکد). یعنی داده‌های کاراکتری ۲۴۶تایی شما روی آن کد کرش می‌کنند. به همین دلیل باید **کد `perfect_project`** را روی Colab بیاورید، نه این مخزن GitHub را.

### پس اولین کار: رساندن کد `perfect_project` به Colab
چون `perfect_project` روی این GitHub **push نشده**، یکی از این دو را انجام دهید:

1. **زیپ کنید و در Drive بگذارید** (ساده‌ترین): کل پوشهٔ `perfect_project` را zip کنید (بدون پوشه‌های سنگین `datasets/`, `checkpoints/`, `pretrained/` اگر می‌خواهید کوچک بماند)، در Drive آپلود کنید و **شناسهٔ فایل (file id)** آن را در نوت‌بوک بگذارید.
2. یا **به GitHub push کنید** (مثلاً یک repo خصوصی) و در Colab `git clone` کنید.

> نوت‌بوک `notebooks/FA_TTV_Colab_perfect_project.ipynb` (در همین مخزن) برای هر دو حالت جای پرکردن `CODE_ZIP_ID` یا `CODE_GIT_URL` دارد.

---

## ۱) خلاصهٔ وضعیت شما (الان کجایید)

| موضوع | وضعیت |
|--------|--------|
| کد `perfect_project` | اصلاحات بحرانی اعمال شده (۲۴۶، CTC، fp16=false، F0 خام، padding از config). تست `tests/test_cpu_dry_run.py` سبز (loss_total≈41.1). |
| داده‌ها | روی Drive آپلود شده (۷ فایل). ساختار واقعی باید پس از باز کردن zip بررسی شود. |
| چک‌پوینت‌ها | `G_3135000.zip` (مدل فارسی قدیمی) و `G_0.pth` روی Drive. **اندازهٔ واژگانشان باید بازرسی شود** (سلول بازرسی نوت‌بوک). |
| filelistها | احتمالاً مسیر قدیمی لینوکس دارند → باید دوباره ساخته شوند. |
| آموزش واقعی GPU | هنوز انجام نشده — بعد از سبز شدن تست دود. |

---

## ۲) بررسی کامل پروژهٔ `perfect_project` (نتیجهٔ بازبینی)

(بر اساس `fixes_fa_applied.md` و `RAHNAMA_NAHAYI_AVARZESH_TTS_FA.md` خودتان)

### ۲.۱ ثابت‌های فنی (تغییر ندهید)
- W2V: `facebook/mms-300m`، لایهٔ ۷، بُعد ۱۰۲۴، hop=320 (50Hz).
- F0: `pyin(fmin=65, fmax=2093, hop=80)`، **Hz خام** روی دیسک؛ `log(f0+1)` فقط داخل train.
- نسبت F0:W2V = ۴ (برش pitch: `ids_slice*4`, طول ۲۴۰).
- Mel: ۸۰ باند، filter=1280، win=1280، hop=320، fmin=0، fmax=8000.
- CTC blank = index 0 = `"_"`؛ `len(symbols)=246=n_vocab`.

### ۲.۲ اصلاحات تأییدشده (از گزارش شما)
C-1 (۲۴۶)، C-2 (blank=0)، C-3 (فیلتر CTC)، H-1 (max_text_len=256)، H-2 (وزن CTC=1.0)، H-3 (F0 خام)، H-4 (برش F0-W2V)، M-1 (fp16=false)، M-2 (grad_accum_steps)، M-4 (تست slice) — همه ✅.

### ۲.۳ شکاف‌های باقی‌مانده (باید قبل از Colab حل شوند)
| مورد | اقدام |
|------|--------|
| `filelists/train_1pct.txt` مسیر لینوکس قدیمی دارد | `prepare_filelist.py` را دوباره روی مسیر محلی Colab اجرا کنید |
| feature‌ها (`.pt`, `_f0.pt`, `_w2v.pt`) موجود نیستند | روی Colab با GPU دوباره استخراج کنید |
| `pretrained/G_1125000.pth` خالی | یا از Drive (G_0.pth شما) یا از HierSpeech++ رسمی |
| `validate_fa_dataset.py` نیست | از `validate_dataset.py` + `pipeline_check.py` استفاده کنید |
| `hazm` نصب نیست | روی Colab `pip install hazm` (برای نرمال‌سازی بهتر) |

---

## ۳) بررسی کارهای گذشته و علت کیفیت پایین قبلی

### ۳.۱ خروجی‌های قبلی شما
- مسیرهای ویندوزی (`C:\New folder (3)\...\logs`) فقط `train.log` و `config.json` دارند؛ فایل `G_*.pth` در آن کپی‌ها نیست (احتمالاً روی سرور لینوکس قدیمی مانده).
- `G_3135000` احتمالاً از زنجیرهٔ آموزشِ **۱۷۸تایی** قدیمی است (طبق log نوامبر ۲۰۲۴ که از `G_1125000` رسمی resume شده بود).

### ۳.۲ چرا «بد نبود ولی عالی هم نبود»
ترکیب این عوامل (همه در `perfect_project` رفع شده‌اند):
1. **توکنایزر ناهمگون** در زمان‌های مختلف (انگلیسی/espeak/G2P/کاراکتری) → CTC و alignment خراب.
2. `n_vocab=178` قدیمی → ناسازگاری لایهٔ متن/CTC.
3. `c_pho=45` و رفتار نامشخص loss → انفجار گرادیان.
4. طول متن > فریم W2V در فارسی کاراکتری → NaN/نمونهٔ خراب.
5. padding ۴۰۳/۲۰۱ با truncate خاموش → جملات بلند ناقص.
6. `fp16_run=true` در برخی configها → NaN در Flow/SDP.
7. filelist با مسیر اشتباه بین سرورها.
8. **restart ناخواسته از `G_0`** (۲۰ نوامبر) → از دست رفتن پیشرفت fine-tune.

> نتیجه: transfer از مدل پایه کار می‌کرد (loss ~۳–۵)، ولی این باگ‌ها کیفیت نهایی را پایین آوردند. با رفع آن‌ها، انتظار بهبود محسوس می‌رود.

---

## ۴) تصمیم چک‌پوینت (با شواهد، نه حدس)

**قاعده:** مدل ۲۴۶تایی فقط چک‌پوینتی را مستقیم resume می‌کند که لایهٔ `enc_p.emb.weight` آن `(246, ...)` باشد.

پس **اولین گام در Colab، بازرسی هر دو چک‌پوینت است** (سلول بازرسی نوت‌بوک، فقط به torch نیاز دارد):

```
نتیجهٔ inspect برای هر فایل:
├─ enc_p.emb.weight = (246, 256)  → ✅ مستقیماً قابل ادامه؛ آن را در checkpoints/<exp>/ بگذارید.
├─ enc_p.emb.weight = (178, 256)  → ناسازگار با ۲۴۶.
│     → با perfect_project/transfer_checkpoint.py وزن‌های سازگار را منتقل کنید
│       (همه‌چیز جز emb و phoneme_classifier) و آن دو لایه از نو init شوند → warm-start.
└─ خطای لود  → از G_1125000 رسمی + scripts/init_finetune_from_pretrained.py شروع کنید.
```

- **حدس محتمل:** `G_0.pth` شما = چک‌پوینت پایهٔ از-قبل-آماده‌شدهٔ ۲۴۶ (خروجی init) → بهترین مبنای fine-tune.
- `G_3135000` = مدل فارسی قدیمی ۱۷۸ → اگر می‌خواهید استفاده کنید، اول `transfer_checkpoint.py`.

**برای بودجهٔ ۸ روز: حتماً fine-tune/warm-start، نه train from scratch.**

---

## ۴.۵) «پیشنهاد اول» آماده شد — warm-start از pretrained (مرحله‌به‌مرحله)

این همان مسیری است که تصویب کردید: اجرای نو، ولی روی پایهٔ pretrained (نه وزن تصادفی).

### گام ۱: گرفتن چک‌پوینت pretrained رسمی
TTV انگلیسی رسمی HierSpeech++ = `ttv_lt960_ckpt.pth` (۱۰۷M، LibriTTS-960). از پوشهٔ Google Drive رسمی:

- پوشهٔ TTV: https://drive.google.com/drive/folders/1QiFFdPhqhiLFo8VXc0x7cFHKXArx7Xza
- در Colab: `!gdown --folder "https://drive.google.com/drive/folders/1QiFFdPhqhiLFo8VXc0x7cFHKXArx7Xza" -O /content/ttv_official`

> یا اگر `G_0.pth`ی که در Drive آپلود کردید همان pretrained ۱۷۸ است (سلول ۸ مشخص می‌کند)، از همان به‌عنوان منبع استفاده کنید.

### گام ۲: ساخت چک‌پوینت warm-start سازگار با ۲۴۶
اسکریپت `scripts/prepare_warmstart_ckpt.py` (در همین مخزن، تست‌شده) این کار را می‌کند:

```bash
export PYTHONPATH=.
python scripts/prepare_warmstart_ckpt.py \
  -c config_gpu_3090.json \
  -p pretrained/ttv_lt960_ckpt.pth \
  -m checkpoints/persian_main
# خروجی: checkpoints/persian_main/G_0.pth  (iteration=0, optimizer تازه)
```

این اسکریپت:
- همهٔ لایه‌های هم‌شکل (decoder/flow/posterior/style/pitch/duration) را از pretrained کپی می‌کند،
- فقط `enc_p.emb` و `phoneme_classifier` را (که ۱۷۸→۲۴۶ تغییر کرده‌اند) از نو init می‌کند،
- چک‌پوینت را با `iteration=0` و یک optimizer تازه ذخیره می‌کند.

### ⚠️ تله‌ای که این اسکریپت می‌بندد (بسیار مهم)
اگر چک‌پوینت خام ۱۷۸ را مستقیم در `checkpoints/<exp>/` بگذارید، به‌خاطر عدم‌تطابق ابعاد،
`utils.load_checkpoint` خطا می‌دهد و بلوک `try/except` در `train_ttv_v1.py` آن را می‌بلعد و
**بی‌صدا از صفرِ تصادفی** آموزش می‌دهد — یعنی روزها GPU هدر می‌رود بدون اینکه بفهمید.
با این اسکریپت، لایه‌ها از قبل ۲۴۶ می‌شوند و لود **تمیز** انجام می‌شود.

### گام ۳: تأیید لود (قبل از خرج GPU)
```bash
python -c "import torch; d=torch.load('checkpoints/persian_main/G_0.pth',map_location='cpu'); print('iter',d['iteration'],'tensors',len(d['model']),'optim',('optimizer' in d))"
# سپس شروع آموزش -> در log باید ببینید: Loaded checkpoint ... (iteration 0)
```

> اثبات: این اسکریپت روی CPU تست شد — ۸۳۳/۸۳۵ لایه کپی، ۲ لایهٔ متن reinit، و `load_checkpoint` با `iteration=0` بدون خطا لود کرد.

در نوت‌بوک Colab، این کار در سلول‌های ۱۲–۱۳ (مسیر B) خودکار انجام می‌شود.

---

## ۵) فاز ۱ — تست دود روی Colab (گام‌به‌گام، دقیق)

> از نوت‌بوک `notebooks/FA_TTV_Colab_perfect_project.ipynb` استفاده کنید؛ شناسه‌های Drive شما از قبل در آن قرار دارد. این‌جا همان مراحل به‌صورت توضیحی آمده است.

۱. **Runtime → Change runtime type → GPU.** سپس `!nvidia-smi`.
۲. **Mount Drive** و تعیین `DRIVE_OUT = /content/drive/MyDrive/zero_tts_run` (خروجی هر مرحله این‌جا ذخیره می‌شود).
۳. **آوردن کد `perfect_project`**: یا `CODE_ZIP_ID` (zip در Drive) را پر کنید و unzip، یا `git clone` از repo خودتان. (کد GitHub `codex` را استفاده **نکنید**.)
۴. **نصب:** `pip install -r requirements.txt`، `pip install hazm gdown`، `apt-get install -y espeak-ng`، سپس build:
   `cd ttv_v1/monotonic_align && python setup.py build_ext --inplace`.
۵. **preflight:**
   ```bash
   python -c "from ttv_v1.text.symbols import symbols; print(len(symbols), symbols[0])"   # باید: 246 _
   python tests/test_cpu_dry_run.py
   ```
۶. **دانلود چک‌پوینت‌ها و داده از Drive** (با `gdown`؛ لینک‌ها باید روی «هرکسی با لینک» باشند).
۷. **بازرسی چک‌پوینت‌ها** (`scripts/inspect_ttv_checkpoint.py` یا سلول inline) → تصمیم فاز ۴.
۸. **آماده‌سازی زیرمجموعه + preprocess** (با اسکریپت‌های خود `perfect_project`):
   ```bash
   export PYTHONPATH=.
   python ttv_v1/preprocessing/prepare_filelist.py --datasets_dir datasets --config config_cpu_smoke.json
   python scripts/validate_dataset.py
   python ttv_v1/preprocessing/extract_token.py --config config_cpu_smoke.json --input_filelist filelists/train_wav.txt
   python ttv_v1/preprocessing/extract_f0.py    --input_filelist filelists/train_wav.txt --skip_existing
   python ttv_v1/preprocessing/extract_w2v.py   --input_filelist filelists/train_wav.txt --device cuda --skip_existing
   python pipeline_check.py --full-report
   ```
   (یا یک‌جا: `bash run_preprocessing.sh`)
۹. **(در صورت سازگاری) قرار دادن چک‌پوینت مبنا برای resume:**
   ```bash
   mkdir -p checkpoints/colab_smoke
   cp /content/ckpts/G_0.pth checkpoints/colab_smoke/G_0.pth   # اگر inspect نشان داد ۲۴۶ است
   ```
۱۰. **آموزش تست دود (۳ epoch):**
    ```bash
    CUDA_VISIBLE_DEVICES=0 python train_ttv_v1.py -c config_cpu_smoke.json -m checkpoints/colab_smoke
    ```
۱۱. **ذخیره روی Drive:**
    ```bash
    rsync -a checkpoints/colab_smoke "$DRIVE_OUT/"; cp -r filelists "$DRIVE_OUT/"
    ```

### معیار موفقیت تست دود
- بدون crash تا پایان ۳ epoch.
- همهٔ lossها **finite** (نه NaN/Inf).
- فایل `checkpoints/colab_smoke/G_*.pth` ساخته شود.
- log شامل `loss/g/w2v` و `loss/g/ctc` باشد.

---

## ۶) فاز ۲ — آموزش کامل GPU (~۸ روز)

فقط بعد از سبز شدن تست دود:
۱. **کل** دیتاست را با همان cleaner preprocess کنید (`bash run_preprocessing.sh`).
۲. مطمئن شوید `filelists/train_wav.txt` مسیر **محلیِ سرور** دارد.
۳. چک‌پوینت مبنا را در `checkpoints/persian_main/` بگذارید (G_0 ۲۴۶ یا خروجی `init_finetune_from_pretrained.py`/`transfer_checkpoint.py`).
۴. در `tmux`/`screen` اجرا کنید:
   ```bash
   bash run_gpu_training.sh start
   # یا: CUDA_VISIBLE_DEVICES=0 python train_ttv_v1.py -c config_gpu_3090.json -m checkpoints/persian_main
   ```
   (`config_gpu_3090.json`: `batch_size=16`, `grad_accum_steps=2`, `fp16_run=false`.)
۵. **بک‌آپ خودکار** `checkpoints/` روی Drive/HF هر چند ساعت (ماشین اجاره‌ای ممکن است پاک شود).
۶. پایش با TensorBoard (`--logdir checkpoints/persian_main`). اگر `loss/g/ctc` پایین نیامد → مشکل توکنایزر/دادهٔ متن. اگر NaN → lr را نصف کنید. اگر OOM → اول `grad_accum_steps`↑ بعد `batch_size`↓ (segment را کم نکنید).
۷. **resume بعد از قطعی:** همان دستور را دوباره بزنید؛ آخرین `G_*.pth` خودکار لود می‌شود.

> توجه: اگر چک‌پوینت ۲۴۶ آماده دارید، fine-tune در ۸ روز معقول است. **train from scratch در ۸ روز توصیه نمی‌شود.**

---

## ۷) فاز ۳ — ارزیابی و inference نهایی

زنجیرهٔ کامل صدا:
```
متن فارسی → TTV-v1 (شما): w2v(1,1024,T)+pitch → net_g رسمی HierSpeech++ → موج ۱۶kHz → SpeechSR → ۲۴k/۴۸k
```
ارزیابی (مطابق پروپزال): `bash run_final_eval.sh` و `python evaluate_persian_tts.py` → MOS, SMOS, WER/CER, MCD, SECS.

> خروجی مستقیم `inference.py` فقط بازسازی w2v/F0 است؛ صدای واقعی نیاز به vocoder رسمی (net_g) دارد.

---

## ۸) چک‌لیست نهایی «بدون اشتباه»

**قبل از Colab**
- [ ] کدِ اجراشده = `perfect_project` (۲۴۶)، نه GitHub `codex` (۱۷۸).
- [ ] `len(symbols)==246` و `config.model.n_vocab==246`.
- [ ] `tests/test_cpu_dry_run.py` سبز.
- [ ] لینک‌های Drive روی «هرکسی با لینک» باز است.

**بعد از preprocess**
- [ ] `pipeline_check.py` بدون خطای جدی.
- [ ] هیچ نمونه‌ای `len(tokens) > w2v_frames` نمانده.
- [ ] مسیر filelist محلی است (نه `/home/rprp/...`).

**قبل از آموزش ۸ روزه**
- [ ] چک‌پوینت مبنا inspect‌شده و ۲۴۶ است (یا با transfer آماده شده).
- [ ] بک‌آپ خودکار checkpoints تنظیم شده.
- [ ] `persian_cleaners` در config = همان زمان `extract_token`.

**حین/پس از آموزش**
- [ ] loss finite، checkpoint هر `save_interval` ذخیره.
- [ ] ارزیابی + inference با vocoder کامل برای چند گویندهٔ unseen.

---

## ۹) سه قدم بعدی شما (خلاصهٔ اجرایی)
1. **کد `perfect_project` را به Colab برسانید** (zip در Drive یا git) — مهم‌ترین قدم.
2. **نوت‌بوک `FA_TTV_Colab_perfect_project.ipynb` را اجرا کنید** تا بازرسی چک‌پوینت + preprocess + تست دود ۳ epoch سبز شود.
3. **بعد از سبز شدن** → GPU ۸ روزه با `run_gpu_training.sh start` روی دیتاست کامل.
