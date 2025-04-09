import torch
from torchvision.utils import save_image
import torch.nn.functional as F

def normalize_tensor(tensor):
    """
    Normalize a tensor image from [0, 1] to [-1, 1].
    """
    return (tensor - 0.5) / 0.5  # or tensor * 2 - 1


def unnormalize_tensor(tensor):
    """
    Unnormalize a tensor image from [-1, 1] back to [0, 1].
    """
    return tensor * 0.5 + 0.5


def save_minibatch_images(noisy_imgs, clean_imgs, denoised_imgs, epoch, output_dir="output/training_images", max_images=4):

    # Clamp if necessary
    denoised_imgs = torch.clamp(denoised_imgs, 0., 1.)

    for i in range(min(max_images, noisy_imgs.size(0))):
        save_image(unnormalize_tensor(noisy_imgs[i]), f"{output_dir}/{epoch+1}_input_{i}.png")
        save_image((denoised_imgs[i]), f"{output_dir}/{epoch+1}_output_{i}.png")
        save_image((clean_imgs[i]), f"{output_dir}/{epoch+1}_target_{i}.png")


def compute_psnr(img1, img2, max_val=1.0):
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(max_val / torch.sqrt(mse))

