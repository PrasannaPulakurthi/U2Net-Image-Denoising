import argparse
from config.base_config import Config
import os

class AllConfig(Config):
    def __init__(self):
        super().__init__()

    def parse_args(self):
        description = 'U2-Image-Denoising'
        parser = argparse.ArgumentParser(description=description)
        
        # Basic training parameters
        parser.add_argument('--img_size', type=int, default=256)
        parser.add_argument('--num_epochs', type=int, default=50)
        parser.add_argument('--batch_size', type=int, default=12)
        parser.add_argument('--learning_rate', type=float, default=1e-3)

        # Noise & dataset
        parser.add_argument('--noise_std', type=float, default=0.1)
        parser.add_argument('--data_dir', type=str, default="data/DIV2K_512")

        # Model & loss
        parser.add_argument('--model_name', type=str, default='u2net', choices=['u2net', 'u2netp'])
        parser.add_argument('--loss_type', type=str, default='ssim_l1', choices=['mse', 'mae', 'ssim', 'ssim_l1', 'ssim_l2'])

        # Output folders
        parser.add_argument('--exp_name', type=str, default='exp')
        parser.add_argument('--output_dir', type=str, default='outputs')
        parser.add_argument('--model_dir', type=str, default='checkpoints')
        parser.add_argument('--train_img_dir', type=str, default='train_img_dir')
        parser.add_argument('--test_img_dir', type=str, default='test_img_dir')

        # Device
        parser.add_argument('--gpu', type=str, default=None)

        args = parser.parse_args()

        args.model_dir = os.path.join(args.output_dir,args.exp_name,args.model_dir)
        args.train_img_dir = os.path.join(args.output_dir,args.exp_name,args.train_img_dir)
        args.test_img_dir = os.path.join(args.output_dir,args.exp_name,args.test_img_dir)

        # Create those directories
        os.makedirs(args.model_dir, exist_ok=True)
        os.makedirs(args.train_img_dir, exist_ok=True)
        os.makedirs(args.test_img_dir, exist_ok=True)

        return args
