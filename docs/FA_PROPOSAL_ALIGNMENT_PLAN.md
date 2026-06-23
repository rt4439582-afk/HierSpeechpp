# نقشهٔ هم‌ترازسازی پیاده‌سازی با پروپزال (دستورالعمل برای ربات)

**هدف:** پیاده‌سازی نوآوری‌های ادعاشدهٔ پروپزال **فقط تا جایی که کیفیت را بالا می‌برد و کار را سخت/ناپایدار نمی‌کند**، و تعدیل ادعاهای غیرعملی.
**پروژهٔ هدف:** `perfect_project` (شاخهٔ `fix/fa-gpu-readiness`, HierSpeech++ / TTV-v1, توکنایزر کاراکتری ۲۴۶).

---

## ۰) قوانین طلایی (برای ربات — حتماً رعایت شود)

1. **هر تغییر پشت یک فلگ config باشد و پیش‌فرض آن خاموش/قدیمی باشد.** baselineِ فعلی (warm-start + char-246) نباید عوض شود.
2. **بعد از هر تغییر، این دو تست باید سبز بمانند:** `python tests/test_cpu_dry_run.py` و `python tests/test_slice_segments_audio.py`.
3. **سازگاری warm-start حفظ شود:** ماژول‌های جدید باید **additive** باشند (پارامتر جدید). اسکریپت `prepare_warmstart_ckpt.py` کلیدهای جدید را خودکار از نو init می‌کند و بقیه را کپی می‌کند (قبلاً تست شده).
4. **تغییری که vocab/توکنایزر را عوض کند، فقط با فلگ جدا و symbolهای جدا** انجام شود تا چک‌پوینت/داده‌های موجود خراب نشوند.
5. هر قابلیت = یک **آزمایش ablation** مستقل برای پایان‌نامه (baseline در برابر baseline+قابلیت).

---

## ۱) اولویت‌بندی (تصمیم نهایی)

| # | آیتم پروپزال | تصمیم | تأثیر کیفیت | سختی | ریسک |
|---|--------------|--------|-------------|------|------|
| A | فرانت‌اند فارسی پیشرفته (نرمال‌سازی + رفع ابهام هم‌نگاره با POS) | ✅ **پیاده کن** (مهم‌ترین) | بالا (WER/CER، طبیعی‌بودن) | کم–متوسط | کم |
| B | سبک‌کدگذار عروضی دوم (dual style encoder) | ✅ **پیاده کن (به‌صورت فلگ/ablation)** | متوسط (SMOS/SECS) | متوسط | کم |
| C | Prior-Mixup | ➖ **اختیاری/کم‌اولویت** | کم–نامشخص برای TTV | کم | کم |
| D | **CFM** به‌جای flow | ❌ **پیاده نکن — ادعا را تعدیل کن** | — | بسیار زیاد | زیاد |
| E | **VQ-VAE** عروضی | ❌ **فعلاً نه — تعدیل/ablation بعدی** | نامشخص | زیاد | متوسط |

**حداقلِ پیشنهادی برای دفاع قوی و کم‌دردسر: A + B.** این دو بیشترین بهبود را با کمترین ریسک می‌دهند و دو نوآوری پروپزال (بومی‌سازی G2P/POS و سبک‌کدگذار) را واقعی می‌کنند.

---

## ۲) آیتم A — فرانت‌اند فارسی پیشرفته (نرمال‌سازی + رفع ابهام هم‌نگاره)

**چرا:** کیفیت TTS فارسی عمدتاً در «متن→واج» برده/باخته می‌شود. رفع ابهام هم‌نگاره (کِشتی/کُشتی) و نرمال‌سازی درست، رسایی (WER/CER) و عروض را مستقیم بهبود می‌دهد. این دقیقاً نوآوری شمارهٔ ۵ پروپزال (POS برای رفع ابهام) را واقعی می‌کند.

### رویکرد بدون شکستن سازگاری (توصیه‌شده)
رفع ابهام را **قبل از** توکن‌سازی کاراکتری انجام بده و خروجی را **با همان نمادهای موجود** بنویس → vocab=246 و چک‌پوینت‌ها دست‌نخورده می‌مانند.

- ابتدا چک کن آیا جدول `ttv_v1/text/symbols.py` **اعراب** (`ـَ ـِ ـُ ـّ ـْ` و …) را دارد:
  ```bash
  python -c "from ttv_v1.text.symbols import symbols; print([c for c in 'َُِّْ' if c in symbols])"
  ```
  - **اگر دارد** → رفع ابهام = درج اعراب درست (مثلاً «کُشتی»). vocab عوض نمی‌شود. ✅
  - **اگر ندارد** → دو گزینه: (۱) چند نماد اعراب را به `symbols.py` اضافه کن (vocab کمی بزرگ‌تر می‌شود → فقط لایهٔ متن دوباره warm-start می‌شود، با همان اسکریپت)، یا (۲) به‌جای اعراب، «صورت املایی متمایز» انتخاب کن.

### فایل‌ها و کد
1. ماژول جدید `ttv_v1/text/persian_frontend.py`:
   ```python
   # نرمال‌سازی + رفع ابهام هم‌نگاره. اگر hazm نبود، fallback بدون آن.
   try:
       from hazm import Normalizer, POSTagger, word_tokenize
       _norm = Normalizer()
       _has_hazm = True
   except Exception:
       _has_hazm = False

   # دیکشنری هم‌نگاره: (واژه, برچسب POS) -> صورت با اعراب/متمایز
   HOMOGRAPHS = {
       ("کشتی", "N"): "کِشتی",     # boat
       ("کشتی", "V"): "کُشتی",     # wrestling/you killed (مثال — کامل کن)
       # ... موارد پرتکرار را اضافه کن
   }

   def normalize_fa(text: str) -> str:
       if _has_hazm:
           text = _norm.normalize(text)
       # نیم‌فاصله، عدد→حرف و ... (اگر در پروژه دارید، همان را صدا بزنید)
       return text

   def disambiguate(text: str, tagger=None) -> str:
       if not _has_hazm or tagger is None:
           return text
       out = []
       for w, pos in tagger.tag(word_tokenize(text)):
           key = (w, pos[0] if pos else "")
           out.append(HOMOGRAPHS.get(key, w))
       return " ".join(out)
   ```
2. در `ttv_v1/text/cleaners.py` یک cleaner جدید **بدون حذف cleaner قبلی**:
   ```python
   def persian_cleaners_advanced(text):
       from ttv_v1.text.persian_frontend import normalize_fa, disambiguate
       text = normalize_fa(text)
       text = disambiguate(text)            # اگر POSTagger در دسترس باشد
       # سپس همان مسیر persian_cleaners فعلی (کاراکتری) را ادامه بده
       return persian_cleaners(text)
   ```
3. فلگ config (در `config*.json`): `"data": { "text_cleaners": ["persian_cleaners_advanced"] }` فقط در configِ ablation؛ baseline روی `["persian_cleaners"]` بماند.
4. وزن POSTagger هضم را در `pretrained/` یا یک مسیر مشخص قرار بده (`postagger.model`) و مسیرش را در ماژول قابل‌تنظیم کن.

### نکتهٔ مهم سازگاری
**اگر `text_cleaners` را عوض کنی، باید featureهای `token` (`.pt`) را دوباره با همان cleaner استخراج کنی** (`extract_token.py`). cleaner زمان train و inference باید یکی باشد.

### معیار موفقیت A
- روی ۲۰ جملهٔ شامل هم‌نگاره، توکن خروجی برای دو معنی **متفاوت** شود.
- پس از آموزش، **WER/CER بهتر** از baseline.

---

## ۳) آیتم B — سبک‌کدگذار عروضی دوم (Dual Style Encoder)

**چرا:** الهام StyleTTS2/Spotlight: یک کدگذار برای هویت آکوستیک، یکی برای سبک عروضی (آهنگ/ریتم). این نوآوری شمارهٔ ۳ پروپزال را واقعی می‌کند و می‌تواند SMOS/SECS را بهبود دهد. چون **additive** است، warm-start سالم می‌ماند.

### فایل‌ها و کد — `ttv_v1/t2w2v_transformer.py` (کلاس `SynthesizerTrn`)
1. در `__init__` (بعد از `self.emb_g = StyleEncoder(...)`):
   ```python
   self.use_prosody_encoder = bool(kwargs.get("use_prosody_encoder", False))
   if self.use_prosody_encoder:
       self.emb_g_prosody = StyleEncoder(in_dim=80, hidden_dim=256, out_dim=256)
   ```
2. در `forward` و هر دو `infer`/`infer_noise_control`، جایی که `g` ساخته می‌شود:
   ```python
   g = self.emb_g(y_mel, y_mask).unsqueeze(-1)
   g_pro = self.emb_g_prosody(y_mel, y_mask).unsqueeze(-1) if self.use_prosody_encoder else g
   ```
   سپس **فقط** ورودی PitchPredictor را به `g_pro` بده:
   ```python
   pitch_predicted = self.pp(w2v_slice, g_pro)   # در forward
   pitch = self.pp(w2v, g_pro)                    # در infer
   ```
   بقیهٔ مسیر (TextEncoder/flow/decoder) همان `g` آکوستیک را نگه دارد.
3. فلگ config: `"model": { "use_prosody_encoder": true }` فقط در configِ ablation.

### سازگاری warm-start
`emb_g_prosody.*` در چک‌پوینت pretrained نیست → اسکریپت `prepare_warmstart_ckpt.py` آن را **خودکار از نو init می‌کند** و بقیه را کپی می‌کند. (این رفتار قبلاً تست و تأیید شده: «کلیدهای غایب/ناسازگار reinit، بقیه copy».) baseline (فلگ خاموش) اصلاً این لایه را نمی‌سازد.

### معیار موفقیت B
- `test_cpu_dry_run.py` با فلگ روشن هم سبز باشد (شکل `pitch_pred` تغییر نکند).
- پس از آموزش، **SMOS/SECS** نسبت به baseline بهتر یا برابر، بدون افت UTMOS.

---

## ۴) آیتم C — Prior-Mixup (اختیاری، کم‌اولویت)

Prior-Mixup در DDDM-VC برای **تبدیل صدا (VC)** و مخلوط‌کردن prior مبدأ/مقصد است؛ برای **TTV** (متن→w2v) کاربرد مستقیم و سود روشنی ندارد. توصیه:
- **فعلاً پیاده نکن.** اگر بعد از A و B وقت ماند، می‌توان نسخهٔ محافظه‌کارانه را در مرحلهٔ **synthesizer/VC** آزمود، نه TTV.
- در پروپزال: Prior-Mixup را به بخش «کارهای آتی» یا «مرحلهٔ VC» منتقل کن.

---

## ۵) آیتم D — CFM: ادعا را تعدیل کن (پیاده نکن)

جایگزینی مولد با **Conditional Flow Matching** یعنی بازنویسی کامل هدف آموزش و نمونه‌برداری ODE — ریسک و هزینهٔ بسیار بالا، نامناسب بودجهٔ فعلی.

**تعدیل پیشنهادی برای پروپزال** (جایگزین جملهٔ CFM):
> «مولد آکوستیک مبتنی بر **جریان نرمال‌سازی (normalizing flow) و رمزگشای سلسله‌مراتبی** در چارچوب HierSpeech++ است. بررسی CFM به‌عنوان کار آتی پیشنهاد می‌شود.»

---

## ۶) آیتم E — VQ-VAE عروضی: فعلاً نه

افزودن گلوگاه گسستهٔ VQ (codebook + commitment loss + پایداری) پرهزینه و پرریسک است.
- **تعدیل پروپزال:** «گسسته‌سازی نوای گفتار به‌عنوان مسیر توسعهٔ آتی / ablation اختیاری.»
- اگر بعداً خواستید، فقط به‌صورت یک ablation کوچک روی مسیر F0 آزموده شود، نه در اجرای اصلی.

---

## ۷) نقشهٔ ارزیابی/Ablation (برای فصل نتایج پایان‌نامه)

همه روی مجموعهٔ آزمونِ گویندگانِ **unseen**:

| مدل | تغییر | انتظار |
|-----|--------|--------|
| M0 | baseline (char-246 + warm-start) | مرجع |
| M1 | + A (فرانت‌اند فارسی) | WER/CER ↓ (بهتر) |
| M2 | + B (سبک‌کدگذار عروضی) | SMOS/SECS ↑ |
| M3 | + A + B | بهترین کلی |

معیارها: **WER/CER** (Whisper فارسی)، **SECS**، **UTMOS**، **MCD**، و **MOS/SMOS** انسانی. این جدول دقیقاً «اثر هر نوآوری» را نشان می‌دهد و دفاع را قوی می‌کند.

---

## ۸) ترتیب اجرای ربات (گام‌به‌گام)

1. شاخهٔ جدید بساز: `git checkout -b feat/proposal-alignment`.
2. **A را پیاده کن** (`persian_frontend.py` + `persian_cleaners_advanced` + فلگ). تست: `tests/test_cpu_dry_run.py` سبز.
3. **B را پیاده کن** (دو خط در `__init__`، شرط `g_pro` در `forward`/`infer`، فلگ). تست: dry-run با `use_prosody_encoder=true` سبز.
4. دو config ablation بساز: `config_ablation_A.json`, `config_ablation_AB.json` (بقیهٔ مقادیر = `config_gpu_3090.json`).
5. مطمئن شو baseline (`config_gpu_3090.json`) **هیچ تغییری نکرده** و فلگ‌ها پیش‌فرض خاموش‌اند.
6. اگر `text_cleaners` در ablation عوض شد → `extract_token.py` را دوباره روی داده اجرا کن.
7. پروپزال را طبق بخش‌های ۵ و ۶ تعدیل کن (CFM/VQ-VAE → کار آتی).
8. commit جداگانه برای هر آیتم + توضیح.

### چک‌لیست پذیرش هر تغییر
- [ ] فلگ config دارد و پیش‌فرض خاموش است.
- [ ] `tests/test_cpu_dry_run.py` و `tests/test_slice_segments_audio.py` سبز.
- [ ] با فلگ روشن، forward بدون خطا و شکل خروجی‌ها معتبر.
- [ ] warm-start هنوز کار می‌کند (لایه‌های جدید reinit، بقیه copy).
- [ ] baseline دست‌نخورده است.

---

## ۹) جمع‌بندی یک‌خطی
**A (فرانت‌اند فارسی) + B (سبک‌کدگذار عروضی) را پیاده کن — هر دو پشت فلگ، additive، کم‌ریسک و واقعاً بهبوددهنده. CFM و VQ-VAE را در پروپزال به «کار آتی» تبدیل کن. Prior-Mixup اختیاری.**
