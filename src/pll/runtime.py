from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _resolve_device(device: str) -> str:
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        try:
            if torch.backends.mps.is_available():  # type: ignore[attr-defined]
                return "mps"
        except Exception:
            pass
        return "cpu"
    return device


def _resolve_model_source() -> str:
    """Return either a local directory or a public HF model id."""

    local_override = os.environ.get("PLL_MODEL_DIR")
    if local_override:
        path = Path(local_override).expanduser().resolve()
        if path.exists():
            return str(path)
    models_root = Path(__file__).resolve().parents[2] / "models"
    if models_root.exists():
        candidates = []
        base = models_root / "base"
        if base.exists():
            candidates.append(base)
        for sub in models_root.iterdir():
            if sub.is_dir() and sub != base:
                candidates.append(sub)
        for cand in candidates:
            if any((cand / fname).exists() for fname in ("config.json", "generation_config.json", "tokenizer_config.json")):
                return str(cand)
    return os.environ.get("PLL_MODEL", "Qwen/Qwen3-0.6B")


def _load_model_and_tokenizer(device: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    model_src = _resolve_model_source()
    try:
        tok = AutoTokenizer.from_pretrained(model_src)
        model = AutoModelForCausalLM.from_pretrained(
            model_src,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        )
    except OSError as exc:  # pragma: no cover - network / filesystem errors
        msg = (
            "Unable to load PLL base model. Set PLL_MODEL_DIR to a local HuggingFace "
            "checkpoint directory or PLL_MODEL to a reachable model id."
        )
        raise RuntimeError(msg) from exc
    return model, tok


@lru_cache(maxsize=1)
def get_model(device: str = "auto"):
    dev = _resolve_device(device)
    model, _ = _load_model_and_tokenizer(dev)
    model.to(dev)
    model.eval()
    return model


@lru_cache(maxsize=1)
def get_tokenizer(device: str = "auto"):
    dev = _resolve_device(device)
    _, tok = _load_model_and_tokenizer(dev)
    return tok
