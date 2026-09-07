from __future__ import annotations

import torch

from conceptgraph.vlm.qwen import QwenChatAdapter


class FakeProcessor:
    def __init__(self) -> None:
        self.template_kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.template_kwargs = kwargs
        return "prompt"

    def __call__(self, **kwargs):
        return {"input_ids": torch.tensor([[1, 2]])}

    def batch_decode(self, generated_ids, **kwargs):
        return ['{"subtasks": []}']


class FakeModel:
    device = torch.device("cpu")

    def generate(self, input_ids, **kwargs):
        suffix = torch.tensor([[3]])
        return torch.cat((input_ids, suffix), dim=1)


def test_qwen35_vision_adapter_disables_thinking() -> None:
    adapter = QwenChatAdapter.__new__(QwenChatAdapter)
    adapter.model_path = "Qwen3.5-4B"
    adapter.max_new_tokens = 32
    adapter.is_vision_language = True
    adapter.model = FakeModel()
    adapter.processor = FakeProcessor()
    adapter.reset()

    answer = adapter("Return JSON only")

    assert answer == '{"subtasks": []}'
    assert adapter.processor.template_kwargs["enable_thinking"] is False
