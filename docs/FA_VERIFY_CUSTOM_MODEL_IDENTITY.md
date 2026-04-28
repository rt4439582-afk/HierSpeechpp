# آیا مدل سفارشی من واقعاً HierSpeech++ است؟

برای جواب قطعی‌تر، این اسکریپت را روی root پروژه سفارشی خودتان اجرا کنید:

```bash
python scripts/verify_hierspeech_fork.py \
  --project_root /path/to/your/custom_project \
  --report_json /tmp/verify_hierspeech_report.json
```

خروجی یکی از این‌هاست:
- `very_likely_hierspeechpp_fork`
- `likely_hierspeechpp_derived`
- `uncertain_partial_overlap`
- `unlikely_hierspeechpp`

## اگر می‌خواهید من 100٪ مطمئن‌تر نظر بدهم، این‌ها را بفرستید:

1. خروجی کامل `/tmp/verify_hierspeech_report.json`
2. فایل‌های زیر از پروژه سفارشی شما:
   - `inference.py`
   - `hierspeechpp_speechsynthesizer.py`
   - `ttv_v1/t2w2v_transformer.py`
   - `ttv_v1/text/cleaners.py`
   - `train_ttv_v1.py` (یا هر train script واقعی)
3. یک نمونه checkpoint + `config.json` کنار آن

## نکته مهم

- اگر `train_ttv_v1.py` واقعی دارید و checkpoint فرمت train-style باشد (`model/optimizer/iteration`)، ادامهٔ آموزش امن‌تر است.
- اگر فقط state_dict داشته باشید، مسیر امن‌تر ساختن checkpointهای train-style با pipeline فعلی است.

