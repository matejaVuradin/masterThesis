import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from metrics import calculate_metrics_fixed_crop
import seaborn as sns
from matplotlib.patches import Rectangle

def center_crop(img, crop_h, crop_w):
    """Centar crop funkcija"""
    if len(img.shape) == 3:  # ako ima kanal dimenziju [C, H, W]
        _, h, w = img.shape
        start_h = (h - crop_h) // 2
        start_w = (w - crop_w) // 2
        return img[:, start_h:start_h+crop_h, start_w:start_w+crop_w]
    else:  # [H, W]
        h, w = img.shape
        start_h = (h - crop_h) // 2
        start_w = (w - crop_w) // 2
        return img[start_h:start_h+crop_h, start_w:start_w+crop_w]

def generate_error_heatmap(real_img, fake_img, sigma=3):
    """
    Generira heatmap grešaka između stvarne i generirane slike
    
    Args:
        real_img: Originalna slika (numpy array)
        fake_img: Generirana slika (numpy array)
        sigma: Standardna devijacija za Gaussian blur za glađu heatmapu
    
    Returns:
        error_map: Mapa grešaka
    """
    # Izračunaj apsolutnu razliku
    error_map = np.abs(real_img - fake_img)
    
    # Primijeni Gaussian blur za glađu vizualizaciju
    from scipy.ndimage import gaussian_filter
    error_map = gaussian_filter(error_map, sigma=sigma)
    
    return error_map

def analyze_worst_cases(G_AB, G_BA, dataloader, config, num_worst=5, direction="T1->T2"):
    """
    Analizira najgore generirane slike na temelju SSIM metrike
    
    Args:
        G_AB, G_BA: Generirani modeli
        dataloader: DataLoader s test podacima
        config: Konfiguracija modela
        num_worst: Broj najgorih slučajeva za analizu
        direction: Smjer translacije ("T1->T2" ili "T2->T1")
    
    Returns:
        worst_cases: Lista tupleova (real_img, fake_img, ssim_score, batch_idx, sample_idx)
    """
    G_AB.eval()
    G_BA.eval()
    
    all_cases = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            real_A = batch["A"].to(config.device)  # T1
            real_B = batch["B"].to(config.device)  # T2
            
            if direction == "T1->T2":
                fake_B = G_AB(real_A)
                real_imgs = (real_B * 0.5 + 0.5).cpu()
                fake_imgs = (fake_B * 0.5 + 0.5).cpu()
            else:  # T2->T1
                fake_A = G_BA(real_B)
                real_imgs = (real_A * 0.5 + 0.5).cpu()
                fake_imgs = (fake_A * 0.5 + 0.5).cpu()
            
            # Izračunaj SSIM za svaku sliku u batchu
            for sample_idx in range(real_imgs.size(0)):
                real_img = real_imgs[sample_idx].squeeze().numpy()
                fake_img = fake_imgs[sample_idx].squeeze().numpy()
                
                # Izračunaj SSIM na crop-u
                metrics = calculate_metrics_fixed_crop(real_img, fake_img)
                ssim_score = metrics["ssim"]
                
                all_cases.append((
                    real_img, fake_img, ssim_score, 
                    batch_idx, sample_idx,
                    batch["A_path"][sample_idx], batch["B_path"][sample_idx]
                ))
    
    # Sortiraj po SSIM (najgori prvi)
    all_cases.sort(key=lambda x: x[2])
    
    return all_cases[:num_worst]

def visualize_worst_cases_with_heatmaps(worst_cases, direction="T1->T2", crop_height=218, crop_width=182):
    """
    Vizualizira najgore slučajeve s heatmapovima grešaka
    
    Args:
        worst_cases: Lista najgorih slučajeva iz analyze_worst_cases
        direction: Smjer translacije
        crop_height, crop_width: Dimenzije za crop
    """
    num_cases = len(worst_cases)
    
    fig, axes = plt.subplots(num_cases, 4, figsize=(16, 4*num_cases))
    
    if num_cases == 1:
        axes = axes.reshape(1, -1)
    
    # Postavi naslove
    axes[0, 0].set_title("Original")
    axes[0, 1].set_title("Generated")
    axes[0, 2].set_title("Error Heatmap")
    axes[0, 3].set_title("Overlay")
    
    for i, (real_img, fake_img, ssim_score, batch_idx, sample_idx, path_a, path_b) in enumerate(worst_cases):
        # Crop slike
        real_cropped = center_crop(real_img, crop_height, crop_width)
        fake_cropped = center_crop(fake_img, crop_height, crop_width)
        
        # Generiraj error heatmap
        error_map = generate_error_heatmap(real_cropped, fake_cropped)
        
        # 1. Originalna slika
        axes[i, 0].imshow(real_cropped, cmap='gray', vmin=0, vmax=1)
        axes[i, 0].axis('off')
        
        # 2. Generirana slika
        axes[i, 1].imshow(fake_cropped, cmap='gray', vmin=0, vmax=1)
        axes[i, 1].axis('off')
        
        # 3. Error heatmap
        im = axes[i, 2].imshow(error_map, cmap='hot', alpha=0.8)
        axes[i, 2].axis('off')
        
        # 4. Overlay - originalna + error heatmap
        axes[i, 3].imshow(real_cropped, cmap='gray', vmin=0, vmax=1)
        axes[i, 3].imshow(error_map, cmap='hot', alpha=0.5)
        axes[i, 3].axis('off')
        
        # Dodaj informacije o slici
        file_name = path_a.split('/')[-1] if direction == "T1->T2" else path_b.split('/')[-1]
        patient_id = file_name.split('_')[0]
        
        axes[i, 0].text(5, 25, f"{patient_id}\nSSIM: {ssim_score:.3f}", 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                       fontsize=10, va='top')
    
    plt.suptitle(f"Najgoriji rezultati za {direction} translaciju", fontsize=16)
    plt.tight_layout()
    plt.show()

def visualize_results_enhanced(G_AB, G_BA, dataloader, num_samples=4, config=None, 
                              crop_height=218, crop_width=182, show_errors=True,
                              analyze_worst=True, num_worst=3):
    """
    Proširena vizualizacija rezultata s analizom grešaka
    
    Args:
        G_AB, G_BA: Generirani modeli
        dataloader: DataLoader
        num_samples: Broj uzoraka za prikaz
        config: Konfiguracija modela
        crop_height, crop_width: Dimenzije za crop
        show_errors: Prikaži li error heatmapove
        analyze_worst: Analiziraj li najgore slučajeve
        num_worst: Broj najgorih slučajeva za analizu
    """
    if config is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = config.device
   
    G_AB.eval()
    G_BA.eval()
   
    dataset = dataloader.dataset
    sample_loader = DataLoader(
        dataset,
        batch_size=num_samples,
        shuffle=False,
        num_workers=1
    )
   
    # 1. Standardna vizualizacija rezultata
    with torch.no_grad():
        batch = next(iter(sample_loader))
        real_A = batch["A"].to(device)
        real_B = batch["B"].to(device)
       
        fake_B = G_AB(real_A)
        fake_A = G_BA(real_B)
        rec_A = G_BA(fake_B)
        rec_B = G_AB(fake_A)
       
        real_A = (real_A * 0.5 + 0.5).cpu()
        fake_B = (fake_B * 0.5 + 0.5).cpu()
        rec_A = (rec_A * 0.5 + 0.5).cpu()
        real_B = (real_B * 0.5 + 0.5).cpu()
        fake_A = (fake_A * 0.5 + 0.5).cpu()
        rec_B = (rec_B * 0.5 + 0.5).cpu()
       
        # Standardni prikaz
        cols = 6 if not show_errors else 9
        fig, axes = plt.subplots(num_samples, cols, figsize=(3*cols, 3*num_samples))
        
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        # Naslovi
        base_titles = ["Original T1", "Generated T2", "Reconstructed T1", 
                      "Original T2", "Generated T1", "Reconstructed T2"]
        
        if show_errors:
            base_titles.extend(["T1→T2 Error", "T2→T1 Error", "Cycle Error"])
            
        for j, title in enumerate(base_titles):
            axes[0, j].set_title(title)
       
        for i in range(num_samples):
            images_raw = [
                real_A[i].squeeze().cpu().numpy(),
                fake_B[i].squeeze().cpu().numpy(),
                rec_A[i].squeeze().cpu().numpy(),
                real_B[i].squeeze().cpu().numpy(),
                fake_A[i].squeeze().cpu().numpy(),
                rec_B[i].squeeze().cpu().numpy()
            ]
           
            images = [center_crop(img, crop_height, crop_width) for img in images_raw]
           
            # Standardni prikaz slika
            for j, img in enumerate(images):
                axes[i, j].imshow(img, cmap='gray', vmin=0, vmax=1)
                axes[i, j].axis('off')
                
            # Error heatmapovi ako su traženi
            if show_errors:
                # T1→T2 error
                error_t1_t2 = generate_error_heatmap(images[3], images[1])  # real_B vs fake_B
                axes[i, 6].imshow(error_t1_t2, cmap='hot')
                axes[i, 6].axis('off')
                
                # T2→T1 error
                error_t2_t1 = generate_error_heatmap(images[0], images[4])  # real_A vs fake_A
                axes[i, 7].imshow(error_t2_t1, cmap='hot')
                axes[i, 7].axis('off')
                
                # Cycle consistency error
                error_cycle = generate_error_heatmap(images[0], images[2])  # real_A vs rec_A
                axes[i, 8].imshow(error_cycle, cmap='hot')
                axes[i, 8].axis('off')
                
                # Dodaj SSIM score
                ssim_t1_t2 = calculate_metrics_fixed_crop(images[3], images[1])["ssim"]
                ssim_t2_t1 = calculate_metrics_fixed_crop(images[0], images[4])["ssim"]
                ssim_cycle = calculate_metrics_fixed_crop(images[0], images[2])["ssim"]
                
                axes[i, 1].text(5, 15, f"SSIM: {ssim_t1_t2:.3f}", 
                               bbox=dict(boxstyle="round,pad=0.2", facecolor="yellow", alpha=0.7),
                               fontsize=8)
                axes[i, 4].text(5, 15, f"SSIM: {ssim_t2_t1:.3f}", 
                               bbox=dict(boxstyle="round,pad=0.2", facecolor="yellow", alpha=0.7),
                               fontsize=8)
                axes[i, 2].text(5, 15, f"SSIM: {ssim_cycle:.3f}", 
                               bbox=dict(boxstyle="round,pad=0.2", facecolor="cyan", alpha=0.7),
                               fontsize=8)
       
        plt.tight_layout()
        plt.show()
    
    # 2. Analiza najgorih slučajeva
    if analyze_worst:
        print("Analiziram najgore T1→T2 translacije...")
        worst_t1_t2 = analyze_worst_cases(G_AB, G_BA, dataloader, config, 
                                         num_worst=num_worst, direction="T1->T2")
        visualize_worst_cases_with_heatmaps(worst_t1_t2, direction="T1->T2", 
                                          crop_height=crop_height, crop_width=crop_width)
        
        print("Analiziram najgore T2→T1 translacije...")
        worst_t2_t1 = analyze_worst_cases(G_AB, G_BA, dataloader, config, 
                                         num_worst=num_worst, direction="T2->T1")
        visualize_worst_cases_with_heatmaps(worst_t2_t1, direction="T2->T1", 
                                          crop_height=crop_height, crop_width=crop_width)

def plot_error_statistics(G_AB, G_BA, dataloader, config, crop_height=218, crop_width=182):
    """
    Plotira statistike grešaka za cijeli dataset
    
    Args:
        G_AB, G_BA: Generirani modeli
        dataloader: DataLoader
        config: Konfiguracija modela
        crop_height, crop_width: Dimenzije za crop
    """
    G_AB.eval()
    G_BA.eval()
    
    ssim_t1_t2 = []
    ssim_t2_t1 = []
    mse_t1_t2 = []
    mse_t2_t1 = []
    
    with torch.no_grad():
        for batch in dataloader:
            real_A = batch["A"].to(config.device)
            real_B = batch["B"].to(config.device)
            
            fake_B = G_AB(real_A)
            fake_A = G_BA(real_B)
            
            real_A_norm = (real_A * 0.5 + 0.5).cpu()
            real_B_norm = (real_B * 0.5 + 0.5).cpu()
            fake_A_norm = (fake_A * 0.5 + 0.5).cpu()
            fake_B_norm = (fake_B * 0.5 + 0.5).cpu()
            
            for i in range(real_A.size(0)):
                # T1→T2
                real_b_crop = center_crop(real_B_norm[i].squeeze().numpy(), crop_height, crop_width)
                fake_b_crop = center_crop(fake_B_norm[i].squeeze().numpy(), crop_height, crop_width)
                metrics_t1_t2 = calculate_metrics_fixed_crop(real_b_crop, fake_b_crop)
                ssim_t1_t2.append(metrics_t1_t2["ssim"])
                mse_t1_t2.append(metrics_t1_t2["mse"])
                
                # T2→T1
                real_a_crop = center_crop(real_A_norm[i].squeeze().numpy(), crop_height, crop_width)
                fake_a_crop = center_crop(fake_A_norm[i].squeeze().numpy(), crop_height, crop_width)
                metrics_t2_t1 = calculate_metrics_fixed_crop(real_a_crop, fake_a_crop)
                ssim_t2_t1.append(metrics_t2_t1["ssim"])
                mse_t2_t1.append(metrics_t2_t1["mse"])
    
    # Plotiranje distribucija
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # SSIM distribucije
    axes[0, 0].hist(ssim_t1_t2, bins=30, alpha=0.7, label='T1→T2', color='blue')
    axes[0, 0].axvline(np.mean(ssim_t1_t2), color='blue', linestyle='--', 
                       label=f'Mean: {np.mean(ssim_t1_t2):.3f}')
    axes[0, 0].set_title('SSIM Distribution T1→T2')
    axes[0, 0].set_xlabel('SSIM')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    
    axes[0, 1].hist(ssim_t2_t1, bins=30, alpha=0.7, label='T2→T1', color='red')
    axes[0, 1].axvline(np.mean(ssim_t2_t1), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(ssim_t2_t1):.3f}')
    axes[0, 1].set_title('SSIM Distribution T2→T1')
    axes[0, 1].set_xlabel('SSIM')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    
    # MSE distribucije
    axes[1, 0].hist(mse_t1_t2, bins=30, alpha=0.7, label='T1→T2', color='blue')
    axes[1, 0].axvline(np.mean(mse_t1_t2), color='blue', linestyle='--', 
                       label=f'Mean: {np.mean(mse_t1_t2):.4f}')
    axes[1, 0].set_title('MSE Distribution T1→T2')
    axes[1, 0].set_xlabel('MSE')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    
    axes[1, 1].hist(mse_t2_t1, bins=30, alpha=0.7, label='T2→T1', color='red')
    axes[1, 1].axvline(np.mean(mse_t2_t1), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(mse_t2_t1):.4f}')
    axes[1, 1].set_title('MSE Distribution T2→T1')
    axes[1, 1].set_xlabel('MSE')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.show()
    
    # Printaj statistike
    print("\n=== Statistike grešaka ===")
    print(f"T1→T2 SSIM - Mean: {np.mean(ssim_t1_t2):.4f}, Std: {np.std(ssim_t1_t2):.4f}, Min: {np.min(ssim_t1_t2):.4f}")
    print(f"T2→T1 SSIM - Mean: {np.mean(ssim_t2_t1):.4f}, Std: {np.std(ssim_t2_t1):.4f}, Min: {np.min(ssim_t2_t1):.4f}")
    print(f"T1→T2 MSE - Mean: {np.mean(mse_t1_t2):.6f}, Std: {np.std(mse_t1_t2):.6f}, Max: {np.max(mse_t1_t2):.6f}")
    print(f"T2→T1 MSE - Mean: {np.mean(mse_t2_t1):.6f}, Std: {np.std(mse_t2_t1):.6f}, Max: {np.max(mse_t2_t1):.6f}")