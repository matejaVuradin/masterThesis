from skimage.metrics import structural_similarity as ssim
import numpy as np
import torch

def calculate_metrics(real_img, fake_img):
    real_np = real_img.cpu().numpy()
    fake_np = fake_img.cpu().numpy()
    
    # Structural Similarity Index
    ssim_value = ssim(real_np, fake_np, data_range=1.0)
    
    # Mean Squared Error
    mse = ((real_np - fake_np) ** 2).mean()
    
    # Peak Signal-to-Noise Ratio
    if mse == 0:
        psnr = 100
    else:
        max_pixel = 1.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    
    return {"ssim": ssim_value, "mse": mse, "psnr": psnr}

def calculate_metrics_fixed_crop(real_img, fake_img, crop_height=218, crop_width=182):
    if isinstance(real_img, torch.Tensor):
        real_img = real_img.cpu().numpy()
    if isinstance(fake_img, torch.Tensor):
        fake_img = fake_img.cpu().numpy()
    
    height, width = real_img.shape
    start_h = (height - crop_height) // 2
    start_w = (width - crop_width) // 2
    
    # Obrezivanje slika
    real_cropped = real_img[start_h:start_h+crop_height, start_w:start_w+crop_width]
    fake_cropped = fake_img[start_h:start_h+crop_height, start_w:start_w+crop_width]
    
    ssim_value = ssim(real_cropped, fake_cropped, data_range=1.0)
    mse = ((real_cropped - fake_cropped) ** 2).mean()
    
    if mse == 0:
        psnr = 100
    else:
        max_pixel = 1.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    
    return {"ssim": ssim_value, "mse": mse, "psnr": psnr}