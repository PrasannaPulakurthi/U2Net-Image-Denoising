# Training & Testing,  Loss = L2 or MSE (Mean Squared Error or L2 loss)
python train_denoising.py --exp_name exp_1 --loss_type mse
python test_denoising.py --exp_name exp_1 

# Training & Testing,  Loss = L1 or MAE (Mean Absolute Error or L1 loss)
python train_denoising.py --exp_name exp_2 --loss_type mae
python test_denoising.py --exp_name exp_2 

# Training & Testing,  Loss = SSIM (Structural Similarity Index Measure)
python train_denoising.py --exp_name exp_3 --loss_type ssim
python test_denoising.py --exp_name exp_3 

# Training & Testing,  Loss = SSIM + L2
python train_denoising.py --exp_name exp_4 --loss_type ssim_l2
python test_denoising.py --exp_name exp_4 

# Training & Testing,  Loss = SSIM + L1
python train_denoising.py --exp_name exp_5 --loss_type ssim_l1
python test_denoising.py --exp_name exp_5 

