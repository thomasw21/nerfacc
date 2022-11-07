from typing import List, Tuple

import torch
from transformers import CLIPModel, CLIPTokenizer, CLIPProcessor, AutoConfig

from .text_image_discriminator import TextImageDiscriminator


class CLIPTextImageDiscriminator(TextImageDiscriminator):
    """Clip based"""
    def __init__(self):
        super().__init__()
        model_name = "openai/clip-vit-base-patch32"
        self.model = CLIPModel.from_pretrained(model_name)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.image_processor = CLIPProcessor.from_pretrained(model_name)
        self.config = AutoConfig.from_pretrained(model_name)

        assert self.image_processor.feature_extractor.do_normalize
        self.register_buffer(
            "image_mean",
            torch.tensor(self.image_processor.feature_extractor.image_mean)
        )
        self.register_buffer(
            "image_std",
            torch.tensor(self.image_processor.feature_extractor.image_std)
        )

    def forward(self, encoded_images: torch.Tensor, encoded_texts: torch.Tensor) -> torch.Tensor:

        ### Copied form `modeling_clip.py`
        # normalized features
        encoded_images = encoded_images / encoded_images.norm(p=2, dim=-1, keepdim=True)
        encoded_texts = encoded_texts / encoded_texts.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity as logits
        # TODO @thomasw21: Check if one needs the actual logit_scale at all.
        # logit_scale = self.model.logit_scale.exp()
        # return - (encoded_texts * encoded_images).sum(-1)

        # # Intuitively you really want to reach one really fast (ie gains from 0.5 to 1 are more important than from 0 to 0.5), we need a more hardcore convex function maybe log(1 - cosine)
        return - (encoded_texts * encoded_images).sum(-1)
        # return torch.acos((encoded_texts * encoded_images).sum(-1))

    @property

    def device(self):
        return self.model.text_projection.weight.device

    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        inputs = self.tokenizer(texts, padding=True, return_tensors="pt").to(self.device)
        return self.model.get_text_features(**inputs)

    def encode_images(self, images: torch.Tensor, encoded_texts: torch.Tensor):
        assert len(images) == encoded_texts.shape[0], f"Image: {images.shape}\nEncoded_texts: {encoded_texts.shape}"
        assert images.shape[1] == 3, "RGB images"

        # normalize image
        images = (images - self.image_mean[None, :, None, None]) / self.image_std[None, :, None, None]

        inputs = {"pixel_values": images}
        return self.model.get_image_features(**inputs)

    @property
    def image_height_width(self) -> Tuple[int, int]:
        return self.config.vision_config.image_size, self.config.vision_config.image_size