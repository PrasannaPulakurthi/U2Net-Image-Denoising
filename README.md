# <p align="center"> Image Denoising</p>


## 📸 Sample Results

| Noisy Image | Denoised Output | Ground Truth |
|-------------|------------------|---------------|
| ![](outputs/test_img_dir/0002_input.png) | ![](outputs/test_img_dir/0002_output.png) | ![](outputs/test_img_dir/0002_target.png) |

## Results
|    Loss   |  PSNR | SSIM |
|-----------|-------|------|
| MSE (L2)  | 24.66 | 0.82 |
| MAE (L1)  | 27.24 | 0.85 |
| SSIM      | 26.23 | 0.88 |
| SSIM + L2 | 26.91 | 0.88 |
| SSIM + L1 | 28.36 | 0.88 |

## 🎯 Training and Testing 

All the training and testing commands can be found in `run_experiments.sh`


## 📈 Evaluation Metrics

During testing, we compute:
- **PSNR** (Peak Signal-to-Noise Ratio)
- **SSIM** (Structural Similarity Index)
