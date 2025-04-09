import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class DIV2KWithSyntheticNoise(Dataset):
    def __init__(self, root_dir, noise_std=0.1, image_size=(256, 256), training_mode='train'):
        """
        Args:
            root_dir (str): Path to folder with clean images.
            noise_std (float): Standard deviation of Gaussian noise to add.
            image_size (tuple): Size to resize images to (height, width).
            training_mode (str): 'train' or 'test' to load from respective subfolder.
        """
        self.root_dir = root_dir + ('/Train' if training_mode == 'train' else '/Test')
        self.noise_std = noise_std
        self.image_size = image_size
        self.image_paths = sorted([
            os.path.join(self.root_dir, fname)
            for fname in os.listdir(self.root_dir)
            if fname.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        if training_mode == 'train':
            self.transform = transforms.Compose([
                transforms.RandomCrop(self.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ])
        else:
            self.transform = transforms.Compose([
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor()
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load clean image
        img_path = self.image_paths[idx]
        clean_img = Image.open(img_path).convert("RGB")
        clean_tensor = self.transform(clean_img)

        # Add synthetic Gaussian noise
        noise = torch.randn_like(clean_tensor) * self.noise_std
        noisy_tensor = torch.clamp(clean_tensor + noise, 0., 1.)

        return noisy_tensor, clean_tensor
