from __future__ import annotations

import os
from typing import Optional

import torch
from PIL import Image
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, AutoTokenizer

from conceptgraph.vlm.base import VisionLanguageChat


class QwenChatAdapter(VisionLanguageChat):
    def __init__(
        self,
        model_path: Optional[str] = None,
        max_new_tokens: int = 256,
    ) -> None:
        if model_path is None:
            model_path = os.getenv(
                "PLANNING_MODEL_PATH", "/225010231/mwl/EMAS/Qwen3.5-9B"
            )

        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        planning_model_path = os.getenv("PLANNING_MODEL_PATH")
        planning_device = os.getenv("EMAS_PLANNING_DEVICE")
        is_planning_model = planning_model_path and (
            os.path.realpath(self.model_path) == os.path.realpath(planning_model_path)
        )
        self.device = planning_device if is_planning_model else None
        self.device_map = "auto"
        self.max_memory = None
        if self.device and self.device.startswith("cuda:"):
            planning_gpu = int(self.device.split(":", 1)[1])
            self.max_memory = {
                index: (
                    os.getenv("EMAS_PLANNING_GPU_MAX_MEMORY", "34GiB")
                    if index == planning_gpu
                    else 0
                )
                for index in range(torch.cuda.device_count())
            }
            self.max_memory["cpu"] = os.getenv(
                "EMAS_PLANNING_CPU_MAX_MEMORY", "128GiB"
            )
        config = AutoConfig.from_pretrained(self.model_path)
        self.is_vision_language = config.model_type in {
            "qwen2_5_vl",
            "qwen3_vl",
            "qwen3_5",
        }
        if self.is_vision_language:
            self.model = self._load_vision_language_model()
            self.processor = AutoProcessor.from_pretrained(self.model_path)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, **self._model_load_kwargs()
            )
            self.processor = AutoTokenizer.from_pretrained(self.model_path)
        self.reset()

    def _load_vision_language_model(self):
        import transformers

        errors = []
        model_classes = [
            getattr(transformers, class_name)
            for class_name in (
                "AutoModelForImageTextToText",
                "AutoModelForVision2Seq",
                "Qwen2_5_VLForConditionalGeneration",
            )
            if hasattr(transformers, class_name)
        ]
        for model_class in model_classes:
            try:
                return model_class.from_pretrained(
                    self.model_path,
                    **self._model_load_kwargs(),
                )
            except (ValueError, TypeError) as exc:
                errors.append(f"{model_class.__name__}: {exc}")
        raise RuntimeError(
            f"No compatible vision-language loader for {self.model_path}: "
            + "; ".join(errors)
        )

    def _model_load_kwargs(self) -> dict:
        kwargs = {"torch_dtype": "auto", "device_map": self.device_map}
        if self.max_memory is not None:
            kwargs["max_memory"] = self.max_memory
        return kwargs

    def reset(self) -> None:
        self.messages: list[dict] = []
        self.current_image: Optional[Image.Image] = None

    def close(self) -> None:
        self.reset()
        self.model = None
        self.processor = None

    def preprocess_image(self, image: Image.Image) -> Image.Image:
        return image.convert("RGB")

    def encode_image(self, image_input: Image.Image) -> Image.Image:
        self.current_image = image_input
        return image_input

    def _coerce_image(self, image: Image.Image | str) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        return Image.open(image).convert("RGB")

    def __call__(self, query: str, image_features: Optional[Image.Image] = None) -> str:
        if image_features is not None and not self.is_vision_language:
            raise ValueError(
                f"{self.model_path} is a text-only Qwen model and cannot accept images"
            )
        if image_features is not None:
            self.current_image = self._coerce_image(image_features)
            user_content = [
                {"type": "image"},
                {"type": "text", "text": query},
            ]
        else:
            user_content = (
                [{"type": "text", "text": query}]
                if self.is_vision_language
                else query
            )

        self.messages.append({"role": "user", "content": user_content})

        template_kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
        text = self.processor.apply_chat_template(self.messages, **template_kwargs)

        processor_kwargs = {
            "text": [text],
            "padding": True,
            "return_tensors": "pt",
        }
        if self.current_image is not None and self.is_vision_language:
            processor_kwargs["images"] = [self.current_image]

        inputs = self.processor(**processor_kwargs)
        inputs = {
            key: value.to(self.model.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
            )
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        answer = output_text[0].strip()

        self.messages.append({"role": "assistant", "content": answer})
        return answer
