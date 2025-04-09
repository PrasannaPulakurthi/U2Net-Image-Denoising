# <p align="center"> Image Denoising</p>


## 📸 Sample Results

| Noisy Image | Denoised Output | Ground Truth |
|-------------|------------------|---------------|
| ![](outputs/test_img_dir/0005_input.png) | ![](outputs/test_img_dir/0005_output.png) | ![](outputs/test_img_dir/0005_target.png) |

## Results
|    Loss   |  PSNR | SSIM |
|-----------|-------|------|
| MSE (L2)  | 24.50 | 0.81 |
| MAE (L1)  | 26.72 | 0.83 |
| SSIM      | 25.93 | 0.86 |
| SSIM + L2 | 26.49 | 0.86 |
| SSIM + L1 | 27.71 | 0.83 |

## 🎯 Training and Testing 

All the training and testing commands can be found in `run_experiments.sh`


## 📈 Evaluation Metrics

During testing, we compute:
- **PSNR** (Peak Signal-to-Noise Ratio)
- **SSIM** (Structural Similarity Index)
