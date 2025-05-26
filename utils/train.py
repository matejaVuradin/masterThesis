import torch
import torch.nn as nn
from tqdm import tqdm

from .config import CycleGANConfig
from .dataset import get_data_loaders
from .test import test_model
from .helper import LambdaLR, ReplayBuffer, create_model, sample_images

def train_cyclegan(config: CycleGANConfig, use_test_split=True, train_dataloader=None, val_dataloader=None, test_dataloader=None):
    """
    Trenira CycleGAN model prema zadanoj konfiguraciji
    
    Args:
        config: Konfiguracija modela
        train_dataloader: Unaprijed pripremljeni dataloader za trening (opcionalno)
        val_dataloader: Unaprijed pripremljeni dataloader za validaciju (opcionalno)
        test_dataloader: Unaprijed pripremljeni dataloader za testiranje (opcionalno)
    
    Returns:
        G_AB, G_BA: Istrenirani generatori
        train_history: Rječnik s povijesti gubitaka
        best_val_loss: Najbolji validacijski gubitak
    """
   
    # Inicijalizacija modela
    G_AB, G_BA, D_A, D_B = create_model(config)
    
    # Postavljanje optimizatora
    optimizer_G = torch.optim.Adam(
        list(G_AB.parameters()) + list(G_BA.parameters()),
        lr=config.lr, betas=(config.beta1, config.beta2)
    )
    optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=config.lr, betas=(config.beta1, config.beta2))
    optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=config.lr, betas=(config.beta1, config.beta2))
    
    # Postavljanje schedulera za learning rate
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
        optimizer_G, lr_lambda=LambdaLR(config.n_epochs, config.decay_epoch).step
    )
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D_A, lr_lambda=LambdaLR(config.n_epochs, config.decay_epoch).step
    )
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D_B, lr_lambda=LambdaLR(config.n_epochs, config.decay_epoch).step
    )
    
    # Stvaranje data loadera s train/validation/test splitom
    if train_dataloader is None or val_dataloader is None:
        if test_dataloader is None:
            train_dataloader, val_dataloader = get_data_loaders(config, test_split=False)
        else:
            train_dataloader, val_dataloader, new_test_dataloader = get_data_loaders(config, test_split=True)
            if test_dataloader is None:
                test_dataloader = new_test_dataloader
    # Stvaranje replay buffera
    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()
    
    # Definiranje funkcije gubitka
    criterion_GAN = nn.MSELoss()  # LSGAN gubitak
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()
    
    # Inicijalizacija najbolje vrijednosti validacijskog gubitka za spremanje modela
    best_val_loss = float('inf')
    
    # Priprema za spremanje podataka o treningu
    train_history = {
        'G_losses': [],
        'D_A_losses': [],
        'D_B_losses': [],
        'val_G_losses': [],
        'val_cycle_losses': []
    }
    
    # Petlja za treniranje
    for epoch in range(config.n_epochs):
        # Postavi modele u način za treniranje
        G_AB.train()
        G_BA.train()
        D_A.train()
        D_B.train()
        
        # Koristi tqdm za progress bar i gubitci za svaku epohu
        progress_bar = tqdm(train_dataloader, desc=f"Epoha {epoch+1}/{config.n_epochs}")
        
        G_losses = []
        D_A_losses = []
        D_B_losses = []

        
        for batch in progress_bar:
            # Postavi ulaz modela
            real_A = batch["A"].to(config.device)
            real_B = batch["B"].to(config.device)
            
            # Adversarial ground truths - ovisi o arhitekturi diskriminatora
            valid = torch.ones((real_A.size(0), 1, 15, 15), requires_grad=False).to(config.device)
            fake = torch.zeros((real_A.size(0), 1, 15, 15), requires_grad=False).to(config.device)
            
            # ---------------------
            # Treniraj Generatore
            # ---------------------
            optimizer_G.zero_grad()
            
            # Identity loss (opcionalno)
            # G_AB(B) trebao bi biti identičan B, i G_BA(A) trebao bi biti identičan A
            if config.lambda_identity > 0:
                id_A = G_BA(real_A)
                id_B = G_AB(real_B)
                loss_id_A = criterion_identity(id_A, real_A) * config.lambda_A * config.lambda_identity
                loss_id_B = criterion_identity(id_B, real_B) * config.lambda_B * config.lambda_identity
            else:
                loss_id_A = 0
                loss_id_B = 0
            
            # GAN loss
            # Generiraj lažne slike
            fake_B = G_AB(real_A)
            fake_A = G_BA(real_B)
            
            # Diskriminator bi trebao misliti da su lažne slike stvarne
            loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)
            loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)
            
            # Cycle consistency loss
            # Rekonstruirane slike trebale bi odgovarati originalima
            rec_A = G_BA(fake_B)
            rec_B = G_AB(fake_A)
            
            loss_cycle_A = criterion_cycle(rec_A, real_A) * config.lambda_A
            loss_cycle_B = criterion_cycle(rec_B, real_B) * config.lambda_B
            
            # Ukupan generator gubitak
            loss_G = loss_GAN_AB + loss_GAN_BA + loss_cycle_A + loss_cycle_B + loss_id_A + loss_id_B
            loss_G.backward()
            optimizer_G.step()
            
            # ----------------------
            # Treniraj Diskriminator A
            # ----------------------
            optimizer_D_A.zero_grad()
            
            # Gubitak za stvarne slike
            loss_real = criterion_GAN(D_A(real_A), valid)
            
            # Gubitak za lažne slike (koristeći buffer)
            fake_A_ = fake_A_buffer.push_and_pop(fake_A)
            loss_fake = criterion_GAN(D_A(fake_A_.detach()), fake)
            
            # Ukupan diskriminator gubitak
            loss_D_A = (loss_real + loss_fake) / 2
            loss_D_A.backward()
            optimizer_D_A.step()
            
            # ----------------------
            # Treniraj Diskriminator B
            # ----------------------
            optimizer_D_B.zero_grad()
            
            # Gubitak za stvarne slike
            loss_real = criterion_GAN(D_B(real_B), valid)
            
            # Gubitak za lažne slike (koristeći buffer)
            fake_B_ = fake_B_buffer.push_and_pop(fake_B)
            loss_fake = criterion_GAN(D_B(fake_B_.detach()), fake)
            
            # Ukupan diskriminator gubitak
            loss_D_B = (loss_real + loss_fake) / 2
            loss_D_B.backward()
            optimizer_D_B.step()
            
            # Spremi gubitke za kasniji ispis
            G_losses.append(loss_G.item())
            D_A_losses.append(loss_D_A.item())
            D_B_losses.append(loss_D_B.item())
            
            # Ažuriraj progress bar
            progress_bar.set_postfix(
                G_loss=f"{loss_G.item():.4f}", 
                D_A_loss=f"{loss_D_A.item():.4f}", 
                D_B_loss=f"{loss_D_B.item():.4f}"
            )
        
        # Izvedi validaciju nakon svake epohe
        G_AB.eval()
        G_BA.eval()
        D_A.eval()
        D_B.eval()
        
        val_G_losses = []
        val_cycle_losses = []
        
        with torch.no_grad():
            for val_batch in val_dataloader:
                real_A = val_batch["A"].to(config.device)
                real_B = val_batch["B"].to(config.device)
                
                # Generiraj lažne slike
                fake_B = G_AB(real_A)
                fake_A = G_BA(real_B)
                
                # Rekonstruiraj slike
                rec_A = G_BA(fake_B)
                rec_B = G_AB(fake_A)
                
                # Izračunaj generatorske i cycle consistency gubitke (samo za praćenje)
                valid = torch.ones((real_A.size(0), 1, 15, 15), device=config.device)

                    
                loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)
                loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)
                
                loss_cycle_A = criterion_cycle(rec_A, real_A) * config.lambda_A
                loss_cycle_B = criterion_cycle(rec_B, real_B) * config.lambda_B
                
                # Ukupan validacijski generator gubitak
                val_G_loss = loss_GAN_AB + loss_GAN_BA + loss_cycle_A + loss_cycle_B
                val_G_losses.append(val_G_loss.item())
                val_cycle_losses.append((loss_cycle_A + loss_cycle_B).item())
        
        # Izračunaj prosječne validacijske gubitke
        avg_val_G_loss = sum(val_G_losses) / len(val_G_losses)
        avg_val_cycle_loss = sum(val_cycle_losses) / len(val_cycle_losses)
        
        # Ažuriraj learning rate-ove
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()
        
        # Izračunaj prosječne gubitke za epohu
        avg_G_loss = sum(G_losses) / len(G_losses)
        avg_D_A_loss = sum(D_A_losses) / len(D_A_losses)
        avg_D_B_loss = sum(D_B_losses) / len(D_B_losses)
        
        # Spremi povijest treninga
        train_history['G_losses'].append(avg_G_loss)
        train_history['D_A_losses'].append(avg_D_A_loss)
        train_history['D_B_losses'].append(avg_D_B_loss)
        train_history['val_G_losses'].append(avg_val_G_loss)
        train_history['val_cycle_losses'].append(avg_val_cycle_loss)
        
        # Ispiši prosječne gubitke za epohu
        print(f"Epoha {epoch+1}/{config.n_epochs} - "
              f"Train gubici: G: {avg_G_loss:.4f}, D_A: {avg_D_A_loss:.4f}, D_B: {avg_D_B_loss:.4f} | "
              f"Val gubici: G: {avg_val_G_loss:.4f}, Cycle: {avg_val_cycle_loss:.4f}")
        
        # Uzorkuj slike
        if (epoch + 1) % config.sample_interval == 0:
            sample_images(epoch + 1, G_AB, G_BA, val_dataloader, config)
        
        # Spremi checkpoint modela na temelju poboljšanja validacijskog gubitka
        if avg_val_cycle_loss < best_val_loss:
            best_val_loss = avg_val_cycle_loss
            print(f"Novi najbolji validacijski gubitak: {best_val_loss:.4f}, spremam modele...")
            torch.save(G_AB.state_dict(), f"{config.checkpoints_dir}/G_AB_best.pth")
            torch.save(G_BA.state_dict(), f"{config.checkpoints_dir}/G_BA_best.pth")
            torch.save(D_A.state_dict(), f"{config.checkpoints_dir}/D_A_best.pth")
            torch.save(D_B.state_dict(), f"{config.checkpoints_dir}/D_B_best.pth")
            
        # Periodički spremi checkpoint
        if (epoch + 1) % config.checkpoint_interval == 0:
            torch.save(G_AB.state_dict(), f"{config.checkpoints_dir}/G_AB_{epoch+1}.pth")
            torch.save(G_BA.state_dict(), f"{config.checkpoints_dir}/G_BA_{epoch+1}.pth")
    
    # Spremi finalne modele
    torch.save(G_AB.state_dict(), f"{config.checkpoints_dir}/G_AB_final.pth")
    torch.save(G_BA.state_dict(), f"{config.checkpoints_dir}/G_BA_final.pth")
    
    # Testiraj model na test setu ako je dostupan
    if test_dataloader is not None:
        print("\nEvaluiram model na test setu...")
        test_model(G_AB, G_BA, test_dataloader, config)
    
    return G_AB, G_BA, train_history, best_val_loss