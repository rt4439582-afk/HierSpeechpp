# چک‌لیست قبل از شروع آموزش فارسی (Copy/Paste)

اگر گفتی «اره بده»، این همان چک‌لیست عملی است که قبل از آموزش باید اجرا کنی.

## دستور اجرا

```bash
cd /workspace/HierSpeechpp
bash scripts/fa_pretrain_checklist.sh \
  --wav_dir /workspace/HierSpeechpp/fa_workspace/data/stage01/wav16k \
  --text_dir /workspace/HierSpeechpp/fa_workspace/data/stage01/text \
  --out_dir /workspace/HierSpeechpp/fa_workspace/reports/precheck_stage01 \
  --min_pairs 1000
```

## چه چیزهایی را چک می‌کند؟

1. وجود جفت wav/txt
2. mono و 16k بودن wav
3. غیرخالی بودن متن
4. حداقل تعداد نمونه (`min_pairs`)
5. میانه طول متن و میانه طول صوت

## اگر FAIL شد چه کنم؟

- `num_pairs کم`: داده بیشتر اضافه کن.
- `median text کم`: متن‌ها خیلی کوتاه‌اند (جمله‌های کامل اضافه کن).
- `median duration کم`: فایل‌ها بیش از حد کوتاه‌اند (۲–۸ ثانیه بهتر است).
- `invalid_sr/invalid_channels`: قبل از train، resample و mono کن.

## بعد از PASS

فقط این را اجرا کن:

```bash
bash /workspace/HierSpeechpp/fa_workspace/scripts/run_stage01.sh
```

