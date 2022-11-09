import importlib
from typing import Tuple, List

import torch

from .text_image_discriminator import TextImageDiscriminator
from diffusers import DiffusionPipeline


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
            if hasattr(class_, "from_pretrained"):
                setattr(self, module_name, class_.from_pretrained(model_name, subfolder=module_name))
            else:
                assert hasattr(class_, "from_config"), f"{class_} should have a `from_pretrained` or a `from_config` method"
                setattr(self, module_name, class_.from_config(model_name, subfolder=module_name))

        # TODO @thomasw21: understand how num_train_timesteps is setup
        self.min_step = int(self.scheduler.num_train_timesteps * 0.02)
        self.max_step = int(self.scheduler.num_train_timesteps * 0.98)

        self.register_buffer(
            "alphas",
            self.scheduler.alphas_cumprod
        ) # for convenience

        self.guidance_scale = 100

    def forward(self, encoded_images: torch.Tensor, encoded_texts: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            batch_size = encoded_images.shape[0]

            # TODO @thomasw21: understand this
            t = torch.randint(self.min_step, self.max_step + 1, [batch_size], dtype=torch.long, device=self.device) # timestep

            # Add random noise
            noise = torch.randn_like(encoded_images)
            latents_noisy = self.scheduler.add_noise(original_samples=encoded_images, noise=noise, timesteps=t)

            # Now we predict the noise
            # TODO @thomasw21: Do some sort of negative prompts for guidance
            latent_model_input = torch.repeat_interleave(latents_noisy, repeats=2, dim=0)
            t_model_input = torch.repeat_interleave(t, repeats=2, dim=0)
            # latent_model_input = latents_noisy
            # t_model_input = t

            latent_model_input = self.scheduler.scale_model_input(latent_model_input)

            noise_pred = self.unet(latent_model_input, t_model_input, encoder_hidden_states=encoded_texts).sample

            # TODO @thomasw21: Uncomment once we get guidance correctly setup
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)


        # TODO @thomasw21: Figure out weighting and how it works exactly
        # w(t), sigma_t^2
        # https://github.com/huggingface/diffusers/blob/3f7edc5f724862cce8d43bca1b531d962e963a3a/src/diffusers/schedulers/scheduling_pndm.py#L404
        w = self.alphas[t] ** 2
        # w = self.alphas[t][:, None, None, None]
        # w = self.alphas[t] ** 0.5 * (1 - self.alphas[t])
        # grad = w * (noise_pred - noise)

        # Compute loss as sum over every pixel and mean over batch size
        # Batch dot product
        loss = w * (
            torch.bmm(
                (noise_pred - noise).view(batch_size, 1, -1),
                encoded_images.view(batch_size, -1, 1)
            ).squeeze(2).squeeze(1)
        )
        # loss = torch.sum(torch.mean(w * (noise_pred - noise) * encoded_images, dim=0))

        return loss

    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        # We add negative text_encodings with empty strings ""
        negative_texts = [""] * len(texts)

        inputs = self.tokenizer(texts + negative_texts, padding=True, return_tensors="pt").to(self.device)
        return self.text_encoder(**inputs).last_hidden_state

    def encode_images(self, images: torch.Tensor, encoded_texts: torch.Tensor):
        if self.guidance_scale == 0:
            assert len(images) == encoded_texts.shape[0], f"Image: {images.shape}\nEncoded_texts: {encoded_texts.shape}"
        else:
            # For each positive sample, we have one negative sample
            assert 2 * len(images) == encoded_texts.shape[0], f"Image: {images.shape}\nEncoded_texts: {encoded_texts.shape}"
        assert images.shape[1] == 3, "RGB images"

        # normalize image
        images = 2 * images - 1 # [0, 1] -> [-1, 1]

        encoded = self.vae.encode(x=images, return_dict=True)
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