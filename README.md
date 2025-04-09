# <p align="center"> Image Denoising</p>


## 📸 Sample Results

| Noisy Image | Denoised Output | Ground Truth |
|-------------|------------------|---------------|
| ![](outputs/test_img_dir/3_input_0.png) | ![](outputs/test_img_dir/3_output_0.png) | ![](outputs/test_img_dir/3_target_0.png) |

## Results
|    Loss   |  PSNR | SSIM |
|-----------|-------|------|
| MSE (L2)  | 26.97 | 0.87 |
| MAE (L1)  | 29.02 | 0.91 |
| SSIM      | 28.95 | 0.92 |
| SSIM + L2 | 29.98 | 0.93 |
| SSIM + L1 | 30.28 | 0.92 |

## 🚀 Installation

   
1. Create a conda environment using Python 3.9.

~~~
conda create -n u2net python=3.9
conda activate u2net
~~~
    
2. Install Pytorch from [Pytorch](https://pytorch.org/get-started/locally/). For example, if you have CUDA 11.8, use the following: 
   
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

3. Clone this repo

```bash
git clone https://github.com/PrasannaPulakurthi/U2Net-Image-Denoising.git
cd U2Net-Image-Denoising
pip install -r requirements.txt
```


## 📥 Dataset Preparation

1. Download the train and test DIV2K dataset from [GoogleDrive](https://drive.google.com/drive/folders/1axZDefThLL6y0q1yjVMEkb4LIFfYVj85?usp=sharing).
2. Organize as:

```
data/
└── DIV2K_512/
    ├── Train/
    └── Test/
```

## 🎯 Training and Testing 

All the training and testing commands can be found in `run_experiments.sh`

### Testing

Download the pretrained model from `outputs/exp_5/checkpoints/u2net_last.pth` in [GoogleDrive](https://drive.google.com/drive/folders/1axZDefThLL6y0q1yjVMEkb4LIFfYVj85?usp=sharing) to `outputs/exp_5/checkpoints/` and run the following command:

```bash
python test_denoising.py --exp_name exp_5
```

### Training

To train the u2net with L1 and SSIM losses using the following command:

```bash
python train_denoising.py --exp_name exp_5 --loss_type ssim_l1
```

## 📈 Evaluation Metrics

During testing, we compute:
- **PSNR** (Peak Signal-to-Noise Ratio)
- **SSIM** (Structural Similarity Index)
