import random
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from .config import CycleGANConfig
from .models import UNetGenerator, UNetGeneratorDeep, Discriminator, VGG19Generator

# Pomoćne klase
class ReplayBuffer:
    def __init__(self, max_size=50):
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        result = []
        for element in data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                result.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    tmp = self.data[i].clone()
                    self.data[i] = element
                    result.append(tmp)
                else:
                    result.append(element)
        return torch.cat(result)

class LambdaLR: #linearno smanjivanje learning rate-a od neke epohe
    def __init__(self, n_epochs, decay_start_epoch):
        self.n_epochs = n_epochs
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)

# Funkcije za inicijalizaciju 
def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('InstanceNorm') != -1:
        # Provjera ima li InstanceNorm težine (affine=True)
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
        # Provjera ima li InstanceNorm bias
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)

def create_model(config: CycleGANConfig):
    # Odabir arhitekture za generatore
    if config.architecture == "standard":
        G_AB = UNetGenerator(config.input_channels, config.output_channels, config.ngf)
        G_BA = UNetGenerator(config.input_channels, config.output_channels, config.ngf)
    elif config.architecture == "vgg19":
        G_AB = VGG19Generator(config.input_channels, config.output_channels)
        G_BA = VGG19Generator(config.input_channels, config.output_channels)
    else:
        raise ValueError(f"Nepoznata arhitektura: {config.architecture}")
    
    D_A = Discriminator(config.input_channels, config.ndf)
    D_B = Discriminator(config.input_channels, config.ndf)
    
    # Inicijalizacija težina
    if "vgg19" in config.architecture:
        for module in G_AB.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)) and not hasattr(module, 'vgg_initialized'):
                nn.init.normal_(module.weight.data, 0.0, 0.02)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.constant_(module.bias.data, 0.0)
        
        for module in G_BA.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)) and not hasattr(module, 'vgg_initialized'):
                nn.init.normal_(module.weight.data, 0.0, 0.02)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.constant_(module.bias.data, 0.0)
    else:
        # Standardna inicijalizacija za ostale tipove generatora
        G_AB.apply(init_weights)
        G_BA.apply(init_weights)
    
    D_A.apply(init_weights)
    D_B.apply(init_weights)
    
    # Prebacivanje na uređaj
    G_AB = G_AB.to(config.device)
    G_BA = G_BA.to(config.device)
    D_A = D_A.to(config.device)
    D_B = D_B.to(config.device)
    
    return G_AB, G_BA, D_A, D_B


def sample_images(epoch, G_AB, G_BA, val_dataloader, config):
    G_AB.eval()
    G_BA.eval()
    
    with torch.no_grad():
        batch = next(iter(val_dataloader))
        real_A = batch["A"].to(config.device)
        fake_B = G_AB(real_A)
        real_B = batch["B"].to(config.device)
        fake_A = G_BA(real_B)
        
        # rekonstrukcije ciklusa
        rec_A = G_BA(fake_B)
        rec_B = G_AB(fake_A)
        
        # konvertiraj u slike u rasponu [0, 1]
        real_A = (real_A * 0.5 + 0.5).cpu()
        fake_B = (fake_B * 0.5 + 0.5).cpu()
        rec_A = (rec_A * 0.5 + 0.5).cpu()
        real_B = (real_B * 0.5 + 0.5).cpu()
        fake_A = (fake_A * 0.5 + 0.5).cpu()
        rec_B = (rec_B * 0.5 + 0.5).cpu()
        
        image_grid = make_grid([
            real_A[0], fake_B[0], rec_A[0],
            real_B[0], fake_A[0], rec_B[0]
        ], nrow=3, normalize=False)
        
        # soremanje
        plt.figure(figsize=(15, 10))
        plt.imshow(image_grid.permute(1, 2, 0).numpy(), cmap='gray')
        plt.axis('off')
        plt.title(f'Epoha {epoch}')
        plt.tight_layout()
        plt.savefig(f"{config.results_dir}/epoch_{epoch}.png")
        plt.close()

    G_AB.train()
    G_BA.train()