from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from PIL import Image


class VisionLanguageChat(ABC):
    @abstractmethod
    def preprocess_image(self, image: Image.Image) -> Any:
        """Convert a PIL image into the backend-specific image input."""

    @abstractmethod
    def encode_image(self, image_input: Any) -> Any:
        """Convert the preprocessed image into the backend-specific context object."""

    @abstractmethod
    def reset(self) -> None:
        """Reset any conversational state."""

    @abstractmethod
    def __call__(self, query: str, image_features: Optional[Any] = None) -> str:
        """Run a prompt against the backend, optionally conditioning on an image."""

    def answer_image(self, image: Image.Image, prompt: str) -> str:
        """Common single-image helper used by the scene-graph captioning flow."""
        self.reset()
        image_input = self.preprocess_image(image)
        image_features = self.encode_image(image_input)
        return self(query=prompt, image_features=image_features)
