from train_ngp_nerf_text_to_3d import *

def main():
    H = 256
    W = 256
    original_image = torch.ones(1, H, W, 3) # black image
    opacity_image = torch.zeros(1, H, W, 1) # Fully transparent

    images_with_background = {}
    for background in Background:
        try:
            images_with_background[background] = data_augment(
                color=original_image,
                opacity=opacity_image,
                resize_shape=(H,W),
                random_resize_crop=False,
                backgrounds=[background],
                blur_background=True
            )[0]
        except NotImplementedError:
            # Ignore those that are not yet implemented
            pass

    save_image(
        tensor=[value.squeeze(0).permute(2, 0, 1) for value in images_with_background.values()],
        fp="background.jpg"
    )


if __name__ == "__main__":
    main()



