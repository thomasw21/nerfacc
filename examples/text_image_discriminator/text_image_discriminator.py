from abc import abstractmethod
from typing import List, Tuple

import torch
from torch import nn

class TextImageDiscriminator(nn.Module):
    @abstractmethod
    def forward(self, encoded_images: torch.Tensor, encoded_texts: torch.Tensor) -> torch.Tensor:
        """Compute similarity between encoded image and encoded text"""

    @abstractmethod
    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        """Compute text encodings"""

    @abstractmethod
    def encode_images(self, images: torch.Tensor, encoded_texts: torch.Tensor):
        """Compute image encodings"""

    @property
    @abstractmethod
    def device(self):
        """Model device"""

    @property
    @abstractmethod
    def image_height_width(self) -> Tuple[int, int]:
        """Model image height/width"""


