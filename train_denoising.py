import torch
from model.u2net import U2NET,  U2NETP
from dataset import DIV2KWithSyntheticNoise
from torch.utils.data import DataLoader
from loss import get_loss_function
from tqdm import tqdm
from utils import save_minibatch_images
from config.all_config import AllConfig

# -----------------------------
# Parameters
# -----------------------------
config = AllConfig()

# -----------------------------
# Initialize DataLoader
# -----------------------------
dataset = DIV2KWithSyntheticNoise(
    root_dir=config.data_dir,
    noise_std=config.noise_std,
    image_size=(config.img_size, config.img_size),
    training_mode='train'  # or 'test'
)

train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

# -----------------------------
# Initialize model
# -----------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if(config.model_name=='u2net'):
    print("...load U2NET---173.6 MB")
    model = U2NET(in_ch=3, out_ch=3).to(device)
elif(config.model_name=='u2netp'):
    print("...load U2NEP---4.7 MB")
    model = U2NETP(in_ch=3, out_ch=3).to(device)

# -----------------------------
# Loss and optimizer
# -----------------------------
criterion = get_loss_function(config.loss_type)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)


# -----------------------------
# Training loop
# -----------------------------
for epoch in range(config.num_epochs):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)

    # Loop over the data
    for batch_idx, (noisy_imgs, clean_imgs) in enumerate(progress_bar):
        noisy_imgs, clean_imgs = noisy_imgs.to(device), clean_imgs.to(device)

        optimizer.zero_grad()
        denoised_imgs, d1,d2,d3,d4,d5,d6 = model(noisy_imgs)
        loss = (criterion(denoised_imgs, clean_imgs) + criterion(d1, clean_imgs) + 
                criterion(d2, clean_imgs) + criterion(d3, clean_imgs) + criterion(d4, clean_imgs) + 
                criterion(d5, clean_imgs) + criterion(d6, clean_imgs))

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())
        
        # Save images for visualization
        if batch_idx == 0:
            save_minibatch_images(noisy_imgs, clean_imgs, denoised_imgs, epoch, config.train_img_dir, 1)

    print(f"[Epoch {epoch+1}] Avg Loss: {running_loss/len(train_loader):.4f}")

    # Save model checkpoint
    # torch.save(model.state_dict(), f"{model_dir}/{config.model_name}_epoch{epoch+1}.pth")

# -----------------------------
# Save the final model checkpoint
# -----------------------------
torch.save(model.state_dict(), f"{config.model_dir}/{config.model_name}_last.pth")