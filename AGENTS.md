# AGENTS.md

## Cursor Cloud specific instructions

HierSpeech++ is a research-grade PyTorch codebase for zero-shot speech synthesis (TTS + voice
conversion + speech super-resolution). There is **no web server / app** — everything runs as
CLI inference scripts (`inference.py`, `inference_vc.py`, `inference_speechsr.py`) and a training
script (`train_ttv_v1.py`).

### Environment notes (non-obvious)

- **No GPU in Cursor Cloud.** `torch.cuda.is_available()` is `False`. The original
  `requirements.txt` pins `torch==1.13.1+cu117` / `torchaudio==0.13.1+cu117`, which have no
  Python 3.12 / CPU wheels. The dev environment instead uses a CPU-only, Python-3.12-compatible
  pinned stack (`torch==2.2.2`, `torchvision==0.17.2`, `torchaudio==2.2.2` + `numpy<2`). These
  are installed by the startup update script; do not run `pip install -r requirements.txt`
  (it will fail on the `+cu117` pins).
- **torchaudio backend:** with `torchaudio==2.2.2`, `torchaudio.load(...)` uses the
  `soundfile`/`ffmpeg` backends (no `torchcodec`). `ffmpeg` and `libsndfile` are required and are
  installed at the system level. Newer torchaudio (>=2.9) switches to a `torchcodec` backend that
  pulls CUDA libs and breaks on this CPU box — keep the pinned versions.
- **System packages** (installed in the VM snapshot, not by the update script):
  `espeak-ng` (phonemizer backend for TTS G2P), `ffmpeg`, `libsndfile`-providing libs, and
  `python3-dev` + `build-essential` (needed to build the `pesq` wheel from source).

### Running things

- **`.cuda()` is hard-coded** in all inference scripts (`inference*.py`) and in `model_load`
  helpers, so they cannot run unmodified on this CPU-only box. The model definitions themselves
  (`speechsr*/speechsr.py`, `hierspeechpp_speechsynthesizer.py`, etc.) are device-agnostic and run
  on CPU if you place tensors/modules on CPU yourself.
- **Large checkpoints are NOT in the repo.** The main Hierarchical Synthesizer and TTV checkpoints
  must be downloaded from the Google Drive links in `README.md` into `./logs/...`. Only the small
  models are committed: `speechsr24k/G_340000.pth`, `speechsr48k/G_100000.pth`, and `denoiser/g_best`.
- **SpeechSR (super-resolution) is the only end-to-end pipeline runnable here** without downloads
  or a GPU, because its checkpoints are committed. To run it on CPU, load
  `speechsr24k`/`speechsr48k` `SynthesizerTrn` with `utils.load_checkpoint`, move the model and the
  input audio to `cpu`, and call `model(audio.unsqueeze(1))` (mirror `inference_speechsr.py` but
  drop the `.cuda()` calls). Verified: upsamples 16 kHz -> 24 kHz and 16 kHz -> 48 kHz.
- **Lint/build:** there is no configured linter, test suite, or build system in this repo
  (no `pytest`, no CI config, no `setup.py`). "Build" = ensure modules import; validate with
  `python3 -c "import inference, inference_speechsr"` style import checks.

### TTS / full pipeline (needs GPU + downloaded checkpoints)

`sh inference.sh` / `sh inference_vc.sh` are the documented entry points (see `README.md`). They
require a CUDA GPU (>=24 GB recommended) and the Google-Drive checkpoints under `./logs/`. Not
runnable in the current CPU-only Cursor Cloud VM without code changes.
