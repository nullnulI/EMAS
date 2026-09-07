from __future__ import annotations

import os
from typing import Optional

import torch
from PIL import Image

from conceptgraph.llava.llava_model import LLaVaChat
from conceptgraph.vlm.base import VisionLanguageChat


class LlavaChatAdapter(VisionLanguageChat):
    def __init__(
        self,
        model_path: Optional[str] = None,
        conv_mode: str = "default",
        num_gpus: int = 1,
    ) -> None:
        if model_path is None:
            model_path = os.getenv("LLAVA_CKPT_PATH")
        if model_path is None:
            raise ValueError(
                "Please provide --vlm-model-path or set LLAVA_CKPT_PATH for the llava backend."
            )

        self._chat = LLaVaChat(model_path, conv_mode, num_gpus)

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        return self._chat.image_processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
        ][0]

    def encode_image(self, image_tensor: torch.Tensor) -> torch.Tensor:
        if image_tensor.ndim == 3:
            image_tensor = image_tensor[None, ...]
        return self._chat.encode_image(image_tensor.half().cuda())

    def reset(self) -> None:
        self._chat.reset()

    def close(self) -> None:
        self._chat.reset()
        self._chat = None

    def __call__(
        self, query: str, image_features: Optional[torch.Tensor] = None
    ) -> str:
        return self._chat(query=query, image_features=image_features)
