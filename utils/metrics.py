from skimage.metrics import structural_similarity as ssim
import numpy as np
import torch

def calculate_metrics(real_img, fake_img):
    """Izračun metrika između stvarne i generirane slike"""
    # Pretvaramo tenzore u numpy arrays
    real_np = real_img.cpu().numpy()
    fake_np = fake_img.cpu().numpy()
    
    # Izračun SSIM (Structural Similarity Index)
    ssim_value = ssim(real_np, fake_np, data_range=1.0)
    
    # Izračun MSE (Mean Squared Error)
    mse = ((real_np - fake_np) ** 2).mean()
    
    # Izračun PSNR (Peak Signal-to-Noise Ratio)
    if mse == 0:
        psnr = 100
    else:
        max_pixel = 1.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    
    return {"ssim": ssim_value, "mse": mse, "psnr": psnr}

def calculate_metrics_fixed_crop(real_img, fake_img, crop_height=218, crop_width=182):
    """
    Izračunava metrike nakon obrezivanja na fiksne dimenzije
    
    Args:
        real_img: Originalna slika kao numpy array ili tensor
        fake_img: Generirana slika kao numpy array ili tensor
        crop_height: Visina obreza (182 za originalne MRI slike)
        crop_width: Širina obreza (218 za originalne MRI slike)
        
    Returns:
        Rječnik s metrikama SSIM, MSE i PSNR
    """
    # Pretvaranje u numpy ako je tensor
    if isinstance(real_img, torch.Tensor):
        real_img = real_img.cpu().numpy()
    if isinstance(fake_img, torch.Tensor):
        fake_img = fake_img.cpu().numpy()
    
    # Dobivanje trenutnih dimenzija
    height, width = real_img.shape
    
    # Izračun margina za obrezivanje
    start_h = (height - crop_height) // 2
    start_w = (width - crop_width) // 2
    
    # Obrezivanje slika na središnji dio
    real_cropped = real_img[start_h:start_h+crop_height, start_w:start_w+crop_width]
    fake_cropped = fake_img[start_h:start_h+crop_height, start_w:start_w+crop_width]
    
    # Izračun metrika na obrezanim slikama
    ssim_value = ssim(real_cropped, fake_cropped, data_range=1.0)
    mse = ((real_cropped - fake_cropped) ** 2).mean()
    
    if mse == 0:
        psnr = 100
    else:
        max_pixel = 1.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    
    return {"ssim": ssim_value, "mse": mse, "psnr": psnr}