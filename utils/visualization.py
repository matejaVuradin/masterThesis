import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Subset, DataLoader

# Vizualizacija
def plot_training_curves(history, title):
    """Vizualizacija krivulja gubitka tijekom treniranja"""
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['G_losses'], label='Generator')
    plt.plot(history['D_A_losses'], label='Diskriminator A')
    plt.plot(history['D_B_losses'], label='Diskriminator B')
    plt.title('Gubici tijekom treniranja')
    plt.xlabel('Epoha')
    plt.ylabel('Gubitak')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_G_losses'], label='Validation Generator')
    plt.plot(history['val_cycle_losses'], label='Validation Cycle')
    plt.title('Validacijski gubici')
    plt.xlabel('Epoha')
    plt.ylabel('Gubitak')
    plt.legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def visualize_results(G_AB, G_BA, dataloader, num_samples=4, config=None, crop_height=218, crop_width=182):
    """
    Vizualiziraj rezultate na prvih num_samples uzoraka iz dataseta
    s croppanjem slika na specificiranu veličinu
    
    Args:
        G_AB: Generator za T1->T2 translaciju
        G_BA: Generator za T2->T1 translaciju
        dataloader: DataLoader s podacima
        num_samples: Broj uzoraka za vizualizaciju
        config: Konfiguracija (opcionalno)
        crop_height: Visina croppane slike
        crop_width: Širina croppane slike
        
    Returns:
        Matplotlib figura s rezultatima (ne prikazuje ju)
    """
    if config is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = config.device
    
    G_AB.eval()
    G_BA.eval()
    
    # Stvaramo novi dataloader koji uzima prvih num_samples uzoraka
    dataset = dataloader.dataset
    
    # Stvaramo novi dataloader s batch_size=num_samples i bez miješanja podataka
    sample_loader = DataLoader(
        dataset, 
        batch_size=num_samples, 
        shuffle=False,
        num_workers=1
    )
    
    # Funkcija za croppanje slika na sredini
    def center_crop(img, crop_h, crop_w):
        # img je numpy array [H, W]
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
    
    # Uzimamo prvi batch koji će sadržavati prvih num_samples uzoraka
    with torch.no_grad():
        batch = next(iter(sample_loader))
        real_A = batch["A"].to(device)
        real_B = batch["B"].to(device)
        
        # Generiraj lažne slike
        fake_B = G_AB(real_A)
        fake_A = G_BA(real_B)
        
        # Generiraj rekonstrukcije
        rec_A = G_BA(fake_B)
        rec_B = G_AB(fake_A)
        
        # Konvertiraj tenzore u slike za prikaz
        real_A = (real_A * 0.5 + 0.5).cpu()
        fake_B = (fake_B * 0.5 + 0.5).cpu()
        rec_A = (rec_A * 0.5 + 0.5).cpu()
        real_B = (real_B * 0.5 + 0.5).cpu()
        fake_A = (fake_A * 0.5 + 0.5).cpu()
        rec_B = (rec_B * 0.5 + 0.5).cpu()
        
        # Kreiraj figure za prikaz
        fig, axes = plt.subplots(num_samples, 6, figsize=(18, 3*num_samples))
        
        # Naslovi za stupce
        if num_samples > 1:
            axes[0, 0].set_title("Original T1")
            axes[0, 1].set_title("Generirani T2")
            axes[0, 2].set_title("Rekonstruirani T1")
            axes[0, 3].set_title("Original T2")
            axes[0, 4].set_title("Generirani T1")
            axes[0, 5].set_title("Rekonstruirani T2")
        else:
            axes[0].set_title("Original T1")
            axes[1].set_title("Generirani T2")
            axes[2].set_title("Rekonstruirani T1")
            axes[3].set_title("Original T2")
            axes[4].set_title("Generirani T1")
            axes[5].set_title("Rekonstruirani T2")
        
        # Prikaži slike
        for i in range(num_samples):
            # Pripremi slike i cropaj ih
            images_raw = [
                real_A[i].squeeze().cpu().numpy(),
                fake_B[i].squeeze().cpu().numpy(),
                rec_A[i].squeeze().cpu().numpy(),
                real_B[i].squeeze().cpu().numpy(),
                fake_A[i].squeeze().cpu().numpy(),
                rec_B[i].squeeze().cpu().numpy()
            ]
            
            # Cropaj slike
            images = [center_crop(img, crop_height, crop_width) for img in images_raw]
            
            for j, img in enumerate(images):
                if num_samples > 1:
                    ax = axes[i, j]
                else:
                    ax = axes[j]
                ax.imshow(img, cmap='gray', vmin=0, vmax=1)
                ax.axis('off')
                
                # Dodajemo informaciju o pacijentu ako imamo putanje
                if j == 0 and "A_path" in batch:  # Samo za prvi stupac
                    import os
                    filename = os.path.basename(batch["A_path"][i])
                    # Pretpostavljam format "IXI123_T1_slice.png"
                    if filename.startswith("IXI"):
                        patient_id = filename.split("_")[0]
                        ax.set_ylabel(f"Pacijent {patient_id}", fontsize=12, rotation=90, labelpad=10)
        
        plt.tight_layout()

