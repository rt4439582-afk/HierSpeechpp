# مهاجرت از اسکریپت قدیمی `fine_tune.py` به مسیر جدید (سازگار)

شما قبلاً این سبک را داشتید:

```bash
python fine_tune.py --data_dir ./my_data --model_name my_model --start_training
```

الان این دستور دوباره در همین ریپو **سازگار** شده و کار می‌کند.

## ورودی دیتا (flat layout)

```text
my_data/
├── audio1.wav
├── audio1.txt
├── audio2.wav
├── audio2.txt
└── ...
```

## اجرای سریع

```bash
python fine_tune.py --data_dir ./my_data --model_name my_model
```

## اجرای با شروع آموزش

```bash
python fine_tune.py --data_dir ./my_data --model_name my_model --start_training
```

## اسکریپت bash آماده

```bash
./fine_tune.sh ./my_data my_model
```

## این اسکریپت چه می‌کند؟
1. validate دیتا
2. استخراج w2v/f0/token
3. ساخت filelist
4. split train/test
5. ساخت config در `logs/<model_name>/config.json`
6. (اختیاری) شروع آموزش `train_ttv_v1.py`

