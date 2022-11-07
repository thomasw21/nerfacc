import importlib
from typing import Tuple, List

import torch
from transformers import CLIPFeatureExtractor

from examples.text_image_discriminator import TextImageDiscriminator
from diffusers import DiffusionPipeline, AutoencoderKL


class SDTextImageDiscriminator(TextImageDiscriminator):
    def __init__(self):
        super().__init__()

        model_name = "runwayml/stable-diffusion-v1-5"
        self.config = DiffusionPipeline.get_config_dict(model_name)
        del self.config["_class_name"]
        del self.config["_diffusers_version"]
        print(self.config)

        # TODO @thomasw21: check if I need a thing called a scheduler
        for module_name in ["text_encoder", "tokenizer", "unet", "vae", "scheduler"]:
            libname, classname = self.config[module_name]
            lib = importlib.import_module(libname)
            class_ = getattr(lib, classname)
            setattr(self, module_name, class_.from_pretrained(model_name, subfolder=module_name))

        # TODO @thomasw21: understand how num_train_timesteps is setup
        self.num_train_timesteps = 1000
        self.min_step = int(self.num_train_timesteps * 0.02)
        self.max_step = int(self.num_train_timesteps * 0.98)

    def forward(self, encoded_images: torch.Tensor, encoded_texts: torch.Tensor) -> torch.Tensor:
        guidance_scale = 7.5

        # TODO @thomasw21: understand this
        t = torch.randint(self.min_step, self.max_step + 1, [1], dtype=torch.long, device=self.device) # timestep

        # Add random noise
        noise = torch.randn_like(encoded_images)
        latents_noisy = self.scheduler.add_noise(encoded_images, noise, t)

        # Now we predict the noise
        latent_model_input = torch.cat([latents_noisy] * 2)
        latent_model_input = self.scheduler.scale_model_input(latent_model_input)
        noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=encoded_texts)

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        loss = noise - noise_pred

        # TODO @thomasw21: Figure out weighting and how it works exactly
        # # w(t), sigma_t^2
        # w = (1 - self.alphas[t])
        # # w = self.alphas[t] ** 0.5 * (1 - self.alphas[t])
        # grad = w * (noise_pred - noise)

        return loss

    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        inputs = self.tokenizer(texts, padding=True, return_tensors="pt").to(self.device)
        return self.model.get_text_features(**inputs)

    def encode_images(self, images: torch.Tensor, encoded_texts: torch.Tensor):
        assert len(images) == encoded_texts.shape[0], f"Image: {images.shape}\nEncoded_texts: {encoded_texts.shape}"
        assert images.shape[1] == 3, "RGB images"

        # normalize image
        images = 2 * images - 1 # [0, 1] -> [-1, 1]

        encoded = self.vae.encode(images=images, return_dict=True)
        # sample and scale
        sampled_encoding = encoded.latent_dist.sample() * (0.18215 * self.scheduler.init_noise_sigma)

        return sampled_encoding

    @property
    def device(self):
        return self.text_encoder.device

    @property
    def image_height_width(self) -> Tuple[int, int]:
        # TODO @thomasw21: this is independent, you can set whatever size you want. And choose your own, it has to be a multiple of 8 due to vae.
        return self.vae.config.sample_size, self.vae.config.sample_size

if __name__ == "__main__":
    SDTextImageDiscriminator()