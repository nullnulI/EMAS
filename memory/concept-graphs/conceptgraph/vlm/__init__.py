from __future__ import annotations

import gc
from typing import Literal, Optional

from conceptgraph.vlm.base import VisionLanguageChat


def build_vlm_chat(
    backend: Literal["llava", "qwen"],
    model_path: Optional[str] = None,
    conv_mode: str = "default",
    num_gpus: int = 1,
) -> VisionLanguageChat:
    if backend == "llava":
        from conceptgraph.vlm.llava import LlavaChatAdapter

        return LlavaChatAdapter(
            model_path=model_path,
            conv_mode=conv_mode,
            num_gpus=num_gpus,
        )
    if backend == "qwen":
        from conceptgraph.vlm.qwen import QwenChatAdapter

        return QwenChatAdapter(model_path=model_path)

    raise ValueError(f"Unsupported VLM backend: {backend}")


def close_vlm_chat(chat: Optional[VisionLanguageChat]) -> None:
    if chat is None:
        return

    close = getattr(chat, "close", None)
    if callable(close):
        close()
    else:
        chat.reset()

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
