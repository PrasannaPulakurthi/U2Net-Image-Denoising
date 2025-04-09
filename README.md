# <p align="center"> Image Denoising</p>


## 📸 Sample Results

| Noisy Image | Denoised Output | Ground Truth |
|-------------|------------------|---------------|
| ![](outputs/test_img_dir/3_input_0.png) | ![](outputs/test_img_dir/3_output_0.png) | ![](outputs/test_img_dir/3_target_0.png) |

## Results
|    Loss   |  PSNR | SSIM |
|-----------|-------|------|
| MSE (L2)  | 24.66 | 0.82 |
| MAE (L1)  | 27.24 | 0.85 |
| SSIM      | 26.23 | 0.88 |
| SSIM + L2 | 26.91 | 0.88 |
| SSIM + L1 | 28.36 | 0.88 |

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


## 📈 Evaluation Metrics

During testing, we compute:
- **PSNR** (Peak Signal-to-Noise Ratio)
- **SSIM** (Structural Similarity Index)
