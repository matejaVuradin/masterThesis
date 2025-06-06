import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Subset, DataLoader

# Vizualizacija
def plot_training_curves(history, title):
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
    
    # prvi batch koji će sadržavati prvih num_samples uzoraka
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
        
        fig, axes = plt.subplots(num_samples, 6, figsize=(18, 3*num_samples))
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
            
            for j, img in enumerate(images):
                if num_samples > 1:
                    ax = axes[i, j]
                else:
                    ax = axes[j]
                ax.imshow(img, cmap='gray', vmin=0, vmax=1)
                ax.axis('off')
        
        plt.tight_layout()

