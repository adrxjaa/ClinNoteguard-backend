#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

# Pre-load virtualenv NVIDIA cuBLAS libraries before torch/ctranslate2
try:
    import ctypes
    for _p in sys.path:
        if "site-packages" in _p:
            _nv_lib = os.path.join(_p, "nvidia", "cu13", "lib")
            if os.path.exists(_nv_lib):
                os.environ["LD_LIBRARY_PATH"] = _nv_lib + ":" + os.environ.get("LD_LIBRARY_PATH", "")
                for _lib in ("libcublasLt.so.12", "libcublas.so.12"):
                    _lp = os.path.join(_nv_lib, _lib)
                    if os.path.exists(_lp):
                        ctypes.CDLL(_lp, mode=ctypes.RTLD_GLOBAL)
                break
except Exception:
    pass

GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def ok(msg: str) -> None:
    print(f"  {GREEN}[PASS]{RESET} {msg}")


def warn(msg: str) -> None:
    print(f"  {YELLOW}[WARN]{RESET} {msg}")


def fail(msg: str) -> None:
    print(f"  {RED}[FAIL]{RESET} {msg}")


def section(title: str) -> None:
    print(f"\n{BOLD}{CYAN}=== {title} ==={RESET}")


def load_dotenv_file(env_path: Path) -> None:
    if not env_path.exists():
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def check_python_and_venv() -> bool:
    section("1. Python Runtime & Virtual Environment")
    ok(f"Python {sys.version.split()[0]} ({sys.executable})")
    in_venv = sys.prefix != sys.base_prefix
    if in_venv:
        ok(f"Virtual environment active: {sys.prefix}")
    else:
        warn("Not running inside a virtual environment; prefer ./venv/bin/python")
    return True


def check_cuda() -> tuple[bool, bool]:
    section("2. CUDA & GPU Diagnostics")
    try:
        import torch
    except ImportError as e:
        fail(f"PyTorch import failed: {e}")
        return False, False

    ok(f"PyTorch {torch.__version__} (compiled CUDA: {torch.version.cuda})")

    if not torch.cuda.is_available():
        warn("torch.cuda.is_available() is False — pipeline will run on CPU")
        return True, False

    device_count = torch.cuda.device_count()
    for idx in range(device_count):
        props = torch.cuda.get_device_properties(idx)
        vram_gb = props.total_memory / (1024 ** 3)
        ok(
            f"GPU[{idx}]: {props.name} | VRAM: {vram_gb:.2f} GB | "
            f"Compute Capability: {props.major}.{props.minor}"
        )

    if torch.backends.cudnn.is_available():
        ok(f"cuDNN available (version: {torch.backends.cudnn.version()})")
    else:
        warn("cuDNN not available")

    try:
        x = torch.randn((256, 256), device="cuda", dtype=torch.float16)
        y = torch.matmul(x, x)
        torch.cuda.synchronize()
        ok(f"CUDA FP16 matrix multiply succeeded (output shape={tuple(y.shape)})")
    except Exception as e:
        fail(f"CUDA FP16 tensor test failed: {e}")
        return True, False

    # Check virtualenv NVIDIA cuBLAS shared libraries used by faster-whisper / CTranslate2
    preloaded = False
    for path in sys.path:
        if "site-packages" in path:
            for cuda_sub in ("cu13", "cu12"):
                nvidia_lib_dir = os.path.join(path, "nvidia", cuda_sub, "lib")
                if os.path.exists(nvidia_lib_dir):
                    ok(f"Found NVIDIA {cuda_sub} runtime libs at: {nvidia_lib_dir}")
                    preloaded = True
                    break
        if preloaded:
            break
    if not preloaded:
        warn("No virtualenv nvidia/cu*/lib directory found (system CUDA libs will be used if installed)")

    return True, True


def check_dependencies() -> bool:
    section("3. Python Package Dependencies")
    required_packages = [
        # Core Django & REST
        ("django", "Django", True),
        ("rest_framework", "djangorestframework", True),
        ("corsheaders", "django-cors-headers", True),
        ("asgiref", "asgiref", True),
        ("sqlparse", "sqlparse", True),
        ("dotenv", "python-dotenv (optional; settings.py has built-in .env loader)", False),
        # WebSocket & Audio
        ("websockets", "websockets", True),
        ("soundfile", "soundfile", True),
        ("torchaudio", "torchaudio", False),
        ("faster_whisper", "faster-whisper", False),
        ("speechbrain", "speechbrain", False),
        ("librosa", "librosa (optional non-16kHz resampler)", False),
        # ML & Verification
        ("torch", "torch", True),
        ("sentence_transformers", "sentence-transformers", True),
        ("sklearn", "scikit-learn", True),
        ("numpy", "numpy", True),
        ("scipy", "scipy", True),
        ("networkx", "networkx", True),
        ("rank_bm25", "rank-bm25", True),
        ("yaml", "PyYAML", True),
        # GenAI & Testing
        ("google.genai", "google-genai (SOAP LLM)", False),
        ("pytest", "pytest", True),
        ("matplotlib", "matplotlib", True),
    ]

    all_required_ok = True
    for module_name, label, is_required in required_packages:
        try:
            mod = importlib.import_module(module_name)
            ver = getattr(mod, "__version__", "installed")
            ok(f"{label}: {ver}")
        except Exception as e:
            if is_required:
                fail(f"{label} missing or broken: {e}")
                all_required_ok = False
            else:
                warn(f"{label} not available ({e}) — fallback mode will be used if needed")
    return all_required_ok


def check_audio_models(cuda_available: bool) -> bool:
    section("4. WebSocket Audio & Diarization Models (Stages 1)")
    all_ok = True

    # Pre-load virtualenv cuBLAS libraries exactly like run_websocket_server.py
    try:
        import ctypes
        for path in sys.path:
            if "site-packages" in path:
                nvidia_lib_dir = os.path.join(path, "nvidia", "cu13", "lib")
                if os.path.exists(nvidia_lib_dir):
                    os.environ["LD_LIBRARY_PATH"] = nvidia_lib_dir + ":" + os.environ.get("LD_LIBRARY_PATH", "")
                    for lib_name in ("libcublasLt.so.12", "libcublas.so.12"):
                        lib_path = os.path.join(nvidia_lib_dir, lib_name)
                        if os.path.exists(lib_path):
                            ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
                    break
    except Exception as e:
        warn(f"cuBLAS preload note: {e}")

    try:
        import numpy as np
        from faster_whisper import WhisperModel

        target_device = "cuda" if cuda_available else "cpu"
        compute_type = "float16" if cuda_available else "float32"
        model = WhisperModel("small", device=target_device, compute_type=compute_type)
        dummy_audio = np.zeros(16000, dtype=np.float32)
        list(model.transcribe(dummy_audio, beam_size=1)[0])
        ok(f"faster-whisper ('small') loaded & transcribed dry-run on {target_device.upper()} ({compute_type})")
    except Exception as e:
        warn(f"faster-whisper check failed or fell back: {e}")
        all_ok = False

    try:
        from speechbrain.inference.speaker import EncoderClassifier  # noqa: F401
        ok("SpeechBrain EncoderClassifier importable (speechbrain/spkrec-ecapa-voxceleb)")
    except Exception as e:
        warn(f"SpeechBrain import failed: {e}")
        all_ok = False

    return all_ok


def check_project_artifacts_and_db() -> bool:
    section("5. Checkpoints, Configs, .env & Django Database")
    all_ok = True

    required_files = [
        BASE_DIR / "checkpoints" / "verifier_demo.pt",
        BASE_DIR / "verify" / "concepts" / "concept_hierarchy.json",
        BASE_DIR / "verify" / "configs" / "verify_config.yaml",
        BASE_DIR / "manage.py",
    ]
    for path in required_files:
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            ok(f"{path.relative_to(BASE_DIR)} ({size_mb:.2f} MB)")
        else:
            fail(f"Missing required file: {path.relative_to(BASE_DIR)}")
            all_ok = False

    optional_ckpt = BASE_DIR / "checkpoints" / "transformer_only_synthea.pt"
    if optional_ckpt.exists():
        ok(f"{optional_ckpt.relative_to(BASE_DIR)} ({optional_ckpt.stat().st_size / (1024 * 1024):.2f} MB)")

    load_dotenv_file(BASE_DIR / ".env")
    if os.environ.get("GEMINI_API_KEY"):
        ok("GEMINI_API_KEY is set in environment/.env")
    else:
        warn("GEMINI_API_KEY not set — Stage 2 SOAP generation will use deterministic rule-based fallback")

    # Django setup & migration check
    try:
        os.environ.setdefault("DJANGO_SETTINGS_MODULE", "clinnote_backend.settings")
        import django
        django.setup()
        from django.db import connection
        from django.db.migrations.executor import MigrationExecutor

        executor = MigrationExecutor(connection)
        plan = executor.migration_plan(executor.loader.graph.leaf_nodes())
        if plan:
            warn(f"{len(plan)} unapplied Django migration(s) detected. Run: ./venv/bin/python manage.py migrate")
        else:
            ok("Django database & migrations are up to date")
    except Exception as e:
        fail(f"Django setup/DB check failed: {e}")
        all_ok = False

    return all_ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify CUDA, dependencies, and project readiness.")
    parser.add_argument(
        "--skip-audio-models",
        action="store_true",
        help="Skip loading faster-whisper / SpeechBrain models (faster check).",
    )
    args = parser.parse_args()

    print(f"{BOLD}ClinNoteGuard System & CUDA Environment Check{RESET}")
    print(f"Workspace: {BASE_DIR}")

    py_ok = check_python_and_venv()
    torch_ok, cuda_ok = check_cuda()
    deps_ok = check_dependencies()
    if not args.skip_audio_models:
        check_audio_models(cuda_ok)
    else:
        section("4. WebSocket Audio & Diarization Models (Skipped)")
        warn("Skipped via --skip-audio-models")
    artifacts_ok = check_project_artifacts_and_db()

    section("Summary")
    overall_ok = py_ok and torch_ok and deps_ok and artifacts_ok
    if overall_ok:
        ok("All core dependencies, checkpoints, and Django checks PASSED.")
        if cuda_ok:
            ok("CUDA acceleration is READY.")
        else:
            warn("Running in CPU mode (CUDA unavailable).")
        return 0

    fail("One or more required checks FAILED. Review errors above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
