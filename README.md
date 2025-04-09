# <p align="center"> Image Denoising</p>


## 📸 Sample Results

| Noisy Image | Denoised Output | Ground Truth |
|-------------|------------------|---------------|
| ![](outputs/test_outputs/0001_input.png) | ![](outputs/test_outputs/0001_output.png) | ![](outputs/test_outputs/0001_target.png) |

## Results
|    Loss   | PSNR | SSIM |
|-----------|------|------|
| MSE (L2)  |  |  |
| MAE (L1)  |  |  |
| SSIM      |  |  |
| SSIM + L2 |  |  |
| SSIM + L1 |  |  |

## 🎯 Training and Testing 

All the training and testing commands can be found in `run_experiments.sh`


## 📈 Evaluation Metrics

During testing, we compute:
- **PSNR** (Peak Signal-to-Noise Ratio)
- **SSIM** (Structural Similarity Index)
