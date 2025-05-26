import torch
import matplotlib.pyplot as plt

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

def visualize_results(G_AB, G_BA, dataloader, num_samples=4, config=None):
    """Vizualiziraj rezultate na nekoliko uzoraka"""
    if config is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = config.device
    
    G_AB.eval()
    G_BA.eval()
    
    with torch.no_grad():
        batch = next(iter(dataloader))
        real_A = batch["A"].to(device)[:num_samples]
        real_B = batch["B"].to(device)[:num_samples]
        
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
            images = [
                real_A[i].squeeze().cpu(),
                fake_B[i].squeeze().cpu(),
                rec_A[i].squeeze().cpu(),
                real_B[i].squeeze().cpu(),
                fake_A[i].squeeze().cpu(),
                rec_B[i].squeeze().cpu()
            ]
            
            for j, img in enumerate(images):
                if num_samples > 1:
                    ax = axes[i, j]
                else:
                    ax = axes[j]
                ax.imshow(img.numpy(), cmap='gray')
                ax.axis('off')
        
        plt.tight_layout()
        plt.show()

def compare_experiments(original_t1, original_t2, models_dict, config):
    """
    Uspoređuje rezultate različitih eksperimenata vizualno
    
    Args:
        original_t1: Originalna T1 slika
        original_t2: Originalna T2 slika
        models_dict: Rječnik s modelima u formatu {'ime_eksperimenta': (G_AB, G_BA)}
        config: Konfiguracija (za device)
    """
    num_experiments = len(models_dict)
    
    plt.figure(figsize=(12, 4 * (num_experiments + 1)))
    
    # Prikaži originalne slike
    plt.subplot(num_experiments + 1, 2, 1)
    plt.title("Original T1")
    plt.imshow(original_t1.squeeze().cpu().numpy(), cmap='gray')
    plt.axis('off')
    
    plt.subplot(num_experiments + 1, 2, 2)
    plt.title("Original T2")
    plt.imshow(original_t2.squeeze().cpu().numpy(), cmap='gray')
    plt.axis('off')
    
    # Pripremi ulaz za modele
    t1_input = original_t1.unsqueeze(0).to(config.device)  # [1, 1, H, W]
    t2_input = original_t2.unsqueeze(0).to(config.device)  # [1, 1, H, W]
    
    # Generiraj i prikaži rezultate za svaki eksperiment
    for i, (exp_name, (G_AB, G_BA)) in enumerate(models_dict.items(), 1):
        G_AB.eval()
        G_BA.eval()
        
        with torch.no_grad():
            fake_t2 = G_AB(t1_input)
            fake_t1 = G_BA(t2_input)
            
            # Normaliziraj za prikaz
            fake_t2 = (fake_t2.squeeze().cpu() * 0.5 + 0.5).numpy()
            fake_t1 = (fake_t1.squeeze().cpu() * 0.5 + 0.5).numpy()
        
        # Prikaži generirane slike
        plt.subplot(num_experiments + 1, 2, 2*i + 1)
        plt.title(f"{exp_name}: T1→T2")
        plt.imshow(fake_t2, cmap='gray')
        plt.axis('off')
        
        plt.subplot(num_experiments + 1, 2, 2*i + 2)
        plt.title(f"{exp_name}: T2→T1")
        plt.imshow(fake_t1, cmap='gray')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
