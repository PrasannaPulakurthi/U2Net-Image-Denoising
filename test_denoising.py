import torch
from model.u2net import U2NET, U2NETP
from dataset import DIV2KWithSyntheticNoise
from torch.utils.data import DataLoader
from pytorch_msssim import ssim
from utils import compute_psnr
from tqdm import tqdm
from utils import save_minibatch_images
from config.all_config import AllConfig
from utils import unnormalize_tensor


# -----------------------------
# Parameters
# -----------------------------
config = AllConfig()

# -----------------------------
# Load Test Dataset
# -----------------------------
test_dataset = DIV2KWithSyntheticNoise(
    root_dir=config.data_dir,
    noise_std=config.noise_std,
    image_size=(config.img_size, config.img_size),
    training_mode='test'
)

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# -----------------------------
# Load Model
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if config.model_name == 'u2net':
    model = U2NET(in_ch=3, out_ch=3)
elif config.model_name == 'u2netp':
    model = U2NETP(in_ch=3, out_ch=3)
else:
    raise ValueError("Unsupported model name")

model.load_state_dict(torch.load(f"{config.model_dir}/{config.model_name}_{config.load_epoch}.pth", map_location=device))
model.to(device)
model.eval()

# -----------------------------
# Evaluation
# -----------------------------
total_psnr = 0.0
total_ssim = 0.0
num_samples = len(test_loader)

with torch.no_grad():
    for idx, (noisy_imgs, clean_imgs) in enumerate(tqdm(test_loader, desc="Evaluating")):
        noisy_imgs = noisy_imgs.to(device)
        clean_imgs = clean_imgs.to(device)

        output = model(noisy_imgs)
        if isinstance(output, tuple):  # Handle U2Net multi-output
            output = output[0]

        output = torch.clamp(output,0,1) # Clamp values between 0 and 1

        # Compute metrics
        batch_psnr = compute_psnr((output), (clean_imgs))
        batch_ssim = ssim((output), (clean_imgs), data_range=1.0, size_average=True)

        total_psnr += batch_psnr.item()
        total_ssim += batch_ssim.item()

        # Save images for visualization
        save_minibatch_images(noisy_imgs, clean_imgs, output, idx, config.test_img_dir, 1)

# -----------------------------
# Final Metrics
# -----------------------------
avg_psnr = total_psnr / num_samples
avg_ssim = total_ssim / num_samples

print(f"\n Evaluation Complete")
print(f"Average PSNR: {avg_psnr:.4f}")
print(f"Average SSIM: {avg_ssim:.4f}")
