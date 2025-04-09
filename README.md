# <p align="center"> Image Denoising</p>


## 📸 Sample Results

| Noisy Image | Denoised Output | Ground Truth |
|-------------|------------------|---------------|
| ![](outputs/test_outputs/0001_input.png) | ![](outputs/test_outputs/0001_output.png) | ![](outputs/test_outputs/0001_target.png) |

## Results
| Loss | PSNR | SSIM |
|-------------|------------------|---------------|
| MSE |  |  |

## 🎯 Training and Testing 

All the training and testing commands can be found in `run_experiments.sh`


## 📈 Evaluation Metrics

During testing, we compute:
- **PSNR** (Peak Signal-to-Noise Ratio)
- **SSIM** (Structural Similarity Index)

## 🙏 Acknowledgments

- [U2Net original repo](https://github.com/xuebinqin/U-2-Net)
- [pytorch-msssim](https://github.com/VainF/pytorch-msssim)
- [DIV2K dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/)
