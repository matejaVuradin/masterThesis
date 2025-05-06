import os
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from sklearn.model_selection import train_test_split

# Konfiguracija
class CycleGANConfig:
    def __init__(self, 
                 t1_dir="../dataset/data/images/t1", 
                 t2_dir="../dataset/data/images/t2",
                 results_dir="../results",
                 checkpoints_dir="../checkpoints",
                 batch_size=8,
                 lr=0.0002,
                 beta1=0.5, #smanjeno s 0.9 na 0.5 zbog GAN-a, da bude osjetljiviji na prethodno
                 beta2=0.999, #default za Adam optimizator
                 n_epochs=100,
                 decay_epoch=100,
                 img_size=256,
                 input_channels=1,
                 output_channels=1,
                 ngf=64,
                 ndf=64,
                 lambda_A=10.0,
                 lambda_B=10.0,
                 lambda_identity=0.5,
                 architecture="standard",
                 sample_interval=20,
                 checkpoint_interval=10,
                 device=None):
        
        # Putanje
        self.t1_dir = t1_dir
        self.t2_dir = t2_dir
        self.results_dir = results_dir
        self.checkpoints_dir = checkpoints_dir
        
        # Parametri treniranja
        self.batch_size = batch_size
        self.lr = lr
        self.beta1 = beta1 #parametar za Adam optimizator
        self.beta2 = beta2 #parametar za Adam optimizator
        self.n_epochs = n_epochs
        self.decay_epoch = decay_epoch # epoha nakon koje se smanjuje learning rate
        self.img_size = img_size
        
        # Parametri modela
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.ngf = ngf # broj filtera u generatoru
        self.ndf = ndf # broj filtera u diskriminatoru
        self.architecture = architecture
        
        # Težine gubitaka
        self.lambda_A = lambda_A #težina cycle gubitka za A->B->A
        self.lambda_B = lambda_B  #težina cycle gubitka za B->A->B
        self.lambda_identity = lambda_identity
        
        # Hardware
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        # Intervali uzorkovanja
        self.sample_interval = sample_interval
        self.checkpoint_interval = checkpoint_interval
        
        # Stvaranje direktorija
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)
    
    def print_config(self):
        """Ispisuje trenutnu konfiguraciju"""
        print("\n=== Konfiguracija modela ===")
        print(f"Arhitektura: {self.architecture}")
        print(f"T1 direktorij: {self.t1_dir}")
        print(f"T2 direktorij: {self.t2_dir}")
        print(f"Batch veličina: {self.batch_size}")
        print(f"Learning rate: {self.lr}")
        print(f"Broj epoha: {self.n_epochs}")
        print(f"Veličina slike: {self.img_size}x{self.img_size}")
        print(f"Broj filtera u generatoru: {self.ngf}")
        print(f"Broj filtera u diskriminatoru: {self.ndf}")
        print(f"Lambda A: {self.lambda_A}")
        print(f"Lambda B: {self.lambda_B}")
        print(f"Lambda identity: {self.lambda_identity}")
        print(f"Uređaj: {self.device}")
        print("============================\n")

# Dataset
class MRIDataset(Dataset):
    def __init__(self, t1_dir, t2_dir, transform=None):
        """
        MRI dataset za parove T1 i T2 slika
        
        Args:
            t1_dir: Direktorij s T1 slikama
            t2_dir: Direktorij s T2 slikama
            transform: Opcionalna transformacija koja će se primijeniti
        """
        self.transform = transform
        self.t1_file_paths = sorted(glob.glob(os.path.join(t1_dir, "*.png"))) #nadi sve png slike
        self.t2_file_paths = sorted(glob.glob(os.path.join(t2_dir, "*.png")))
        
        print(f"Pronađeno {len(self.t1_file_paths)} T1 slika i {len(self.t2_file_paths)} T2 slika")
        
        self.t1_ids = [os.path.basename(f).split('_')[0].replace("IXI", "") for f in self.t1_file_paths]
        self.t2_ids = [os.path.basename(f).split('_')[0].replace("IXI", "") for f in self.t2_file_paths]
        
        # Pronalaženje zajedničkih ID-ova
        common_ids = list(set(self.t1_ids) & set(self.t2_ids))
        if not common_ids:
            raise ValueError("Nije pronađen nijedan par T1/T2 slika!")
            
        print(f"Pronađeno {len(common_ids)} sparenih T1/T2 slika")
        
        # Kreiranje mapiranja između ID-ova i putanja
        t1_id_to_path = {os.path.basename(f).split('_')[0].replace("IXI", ""): f for f in self.t1_file_paths}
        t2_id_to_path = {os.path.basename(f).split('_')[0].replace("IXI", ""): f for f in self.t2_file_paths}
        
        # Zadržavanje samo sparenih slika
        self.t1_files = [t1_id_to_path[id] for id in common_ids]
        self.t2_files = [t2_id_to_path[id] for id in common_ids]
        
        # Sortiranje po ID-u pacijenta
        sorted_indices = sorted(range(len(common_ids)), key=lambda i: common_ids[i])
        self.t1_files = [self.t1_files[i] for i in sorted_indices]
        self.t2_files = [self.t2_files[i] for i in sorted_indices]
        
    def __getitem__(self, index):
        # Učitavanje slika
        t1_img = Image.open(self.t1_files[index]).convert('L')  # Konvertiranje u greyscale- 8 bit
        t2_img = Image.open(self.t2_files[index]).convert('L')
        
        if self.transform:
            t1_img = self.transform(t1_img)
            t2_img = self.transform(t2_img)
            
        return {"A": t1_img, "B": t2_img, 
                "A_path": self.t1_files[index], 
                "B_path": self.t2_files[index]}
    
    def __len__(self):
        return len(self.t1_files) #vrati broj parova slika

# Transformacije
def pad_to_size(img, target_size=256):
    width, height = img.size
    padding_left = (target_size - width) // 2
    padding_right = target_size - width - padding_left
    padding_top = (target_size - height) // 2
    padding_bottom = target_size - height - padding_top
    padding = (padding_left, padding_top, padding_right, padding_bottom)
    return transforms.functional.pad(img, padding, fill=0)

class PadToSize:
    def __init__(self, size=256):
        self.size = size
        
    def __call__(self, img):
        return pad_to_size(img, self.size)

def get_transforms(config, use_padding=True):
    """Vraća transformacije za slike"""
    if use_padding:
        return transforms.Compose([
            PadToSize(config.img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # Normalizacija na [-1, 1]
        ])
    else:
        return transforms.Compose([
            transforms.Resize((config.img_size, config.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # Normalizacija na [-1, 1]
        ])




#####################################################################################################################################################################

# U-Net komponente za generator
class UNetDown(nn.Module): #encoder
    def __init__(self, in_channels, out_channels, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False)] # Stride=2 smanjuje dimenzije za pola
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
class UNetUp(nn.Module):  #decoder
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        return torch.cat((x, skip_input), 1) #prethodni iz dekodera i skip iz encodera, po kanalu dimenzije

# Standardni U-Net Generator
class UNetGenerator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=64):
        super(UNetGenerator, self).__init__()
        self.down1 = UNetDown(in_channels, features, normalize=False) # [1, 256, 256] -> [64, 128, 128]
        self.down2 = UNetDown(features, features * 2) # [64, 128, 128] -> [128, 64, 64]
        self.down3 = UNetDown(features * 2, features * 4) # [128, 64, 64] -> [256, 32, 32]
        self.down4 = UNetDown(features * 4, features * 8, dropout=0.5) # [256, 32, 32] -> [512, 16, 16]
        self.down5 = UNetDown(features * 8, features * 8, dropout=0.5) # [512, 16, 16] -> [512, 8, 8]
        self.down6 = UNetDown(features * 8, features * 8, dropout=0.5) # [512, 8, 8] -> [512, 4, 4]
        self.down7 = UNetDown(features * 8, features * 8, dropout=0.5) # [512, 4, 4] -> [512, 2, 2]
        self.down8 = UNetDown(features * 8, features * 8, normalize=False, dropout=0.5) # [512, 2, 2] -> [512, 1, 1]

        self.up1 = UNetUp(features * 8, features * 8, dropout=0.5) # [512, 1, 1] -> [512, 2, 2]
        self.up2 = UNetUp(features * 16, features * 8, dropout=0.5) 
        self.up3 = UNetUp(features * 16, features * 8, dropout=0.5)
        self.up4 = UNetUp(features * 16, features * 8, dropout=0.5)
        self.up5 = UNetUp(features * 16, features * 4)
        self.up6 = UNetUp(features * 8, features * 2)
        self.up7 = UNetUp(features * 4, features)
        
        self.final = nn.Sequential(
            nn.ConvTranspose2d(features * 2, out_channels, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        # Encoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6) # [512, 2, 2]
        d8 = self.down8(d7) # [512, 1, 1]
        
        # Decoder sa skip konekcijama
        u1 = self.up1(d8, d7)  # [1024, 2, 2]
        u2 = self.up2(u1, d6)  # [1024, 4, 4]
        u3 = self.up3(u2, d5) # [1024, 8, 8]
        u4 = self.up4(u3, d4) # [1024, 16, 16]
        u5 = self.up5(u4, d3) # [512, 32, 32]
        u6 = self.up6(u5, d2) # [256, 64, 64]
        u7 = self.up7(u6, d1) # [128, 128, 128]
        
        return self.final(u7) # [1, 256, 256]

# Dublji U-Net Generator s jednim dodatnim slojem
class UNetGeneratorDeep(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=64):
        super(UNetGeneratorDeep, self).__init__()
        self.down1 = UNetDown(in_channels, features, normalize=False)
        self.down2 = UNetDown(features, features * 2)
        self.down3 = UNetDown(features * 2, features * 4)
        self.down4 = UNetDown(features * 4, features * 8, dropout=0.5)
        self.down5 = UNetDown(features * 8, features * 8, dropout=0.5)
        self.down6 = UNetDown(features * 8, features * 8, dropout=0.5)
        self.down7 = UNetDown(features * 8, features * 8, dropout=0.5)
        self.down8 = UNetDown(features * 8, features * 8, dropout=0.5)
        self.down9 = UNetDown(features * 8, features * 8, normalize=False, dropout=0.5)  # Dodatni sloj!

        self.up1 = UNetUp(features * 8, features * 8, dropout=0.5)
        self.up2 = UNetUp(features * 16, features * 8, dropout=0.5)
        self.up3 = UNetUp(features * 16, features * 8, dropout=0.5)
        self.up4 = UNetUp(features * 16, features * 8, dropout=0.5)
        self.up5 = UNetUp(features * 16, features * 8, dropout=0.5)
        self.up6 = UNetUp(features * 16, features * 4)
        self.up7 = UNetUp(features * 8, features * 2)
        self.up8 = UNetUp(features * 4, features)
        
        self.final = nn.Sequential(
            nn.ConvTranspose2d(features * 2, out_channels, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        # Encoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        d9 = self.down9(d8)
        
        # Decoder sa skip konekcijama
        u1 = self.up1(d9, d8)
        u2 = self.up2(u1, d7)
        u3 = self.up3(u2, d6)
        u4 = self.up4(u3, d5)
        u5 = self.up5(u4, d4)
        u6 = self.up6(u5, d3)
        u7 = self.up7(u6, d2)
        u8 = self.up8(u7, d1)
        
        return self.final(u8)

# Diskriminator
class Discriminator(nn.Module):
    def __init__(self, in_channels=1, features=64):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            # Ulaz: 1 x 256 x 256
            nn.Conv2d(in_channels, features, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # 64 x 128 x 128
            nn.Conv2d(features, features * 2, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 128 x 64 x 64
            nn.Conv2d(features * 2, features * 4, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 256 x 32 x 32
            nn.Conv2d(features * 4, features * 8, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(features * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # 512 x 16 x 16
            # PatchGAN - klasificiramo jesu li pojedini patchevi stvarni ili lažni
            nn.Conv2d(features * 8, 1, 4, padding=1)
            # Izlaz: 1 x 15 x 15
        )

    def forward(self, img):
        return self.model(img)

# Dublji Diskriminator
class DiscriminatorDeep(nn.Module):
    def __init__(self, in_channels=1, features=64):
        super(DiscriminatorDeep, self).__init__()

        self.model = nn.Sequential(
            # Ulaz: 1 x 256 x 256
            nn.Conv2d(in_channels, features, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # 64 x 128 x 128
            nn.Conv2d(features, features * 2, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 128 x 64 x 64
            nn.Conv2d(features * 2, features * 4, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 256 x 32 x 32
            nn.Conv2d(features * 4, features * 8, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(features * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # 512 x 16 x 16
            nn.Conv2d(features * 8, features * 8, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(features * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # 512 x 8 x 8
            nn.Conv2d(features * 8, 1, 4, padding=1)
            # Izlaz: 1 x 7 x 7
        )

    def forward(self, img):
        return self.model(img)

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

# Funkcije za inicijalizaciju i treniranje

def init_weights(m):
    """Inicijalizacija težina modela prema GAN praksi"""
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
    """Stvara i inicijalizira modele prema zadanoj konfiguraciji"""
    # Odabir arhitekture za generatore
    if config.architecture == "standard":
        G_AB = UNetGenerator(config.input_channels, config.output_channels, config.ngf)
        G_BA = UNetGenerator(config.input_channels, config.output_channels, config.ngf)
    elif config.architecture == "deep":
        G_AB = UNetGeneratorDeep(config.input_channels, config.output_channels, config.ngf)
        G_BA = UNetGeneratorDeep(config.input_channels, config.output_channels, config.ngf)
    else:
        raise ValueError(f"Nepoznata arhitektura: {config.architecture}")
    
    # Odabir arhitekture za diskriminatore
    if config.architecture == "standard":
        D_A = Discriminator(config.input_channels, config.ndf)
        D_B = Discriminator(config.input_channels, config.ndf)
    elif config.architecture == "deep":
        D_A = DiscriminatorDeep(config.input_channels, config.ndf)
        D_B = DiscriminatorDeep(config.input_channels, config.ndf)
    
    # Inicijalizacija težina
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

def get_data_loaders(config, test_split=True):
    """
    Stvaranje data loadera za treniranje, validaciju i opcionalno testiranje
    
    Args:
        config: Konfiguracija modela
        test_split: Ako True, stvara train/val/test split (70%/15%/15%)
                   Ako False, stvara samo train/val split (80%/20%)
    
    Returns:
        Ako test_split=True: (train_dataloader, val_dataloader, test_dataloader)
        Ako test_split=False: (train_dataloader, val_dataloader)
    """
    # Transformacije
    transform = get_transforms(config, use_padding=True)
    
    # Dataset
    dataset = MRIDataset(config.t1_dir, config.t2_dir, transform=transform)
    
    if test_split:
        # Izračunaj train/val/test split (70%/15%/15%)
        train_size = int(0.7 * len(dataset))
        val_size = int(0.15 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        
        # Koristi sklearn za stratificirani split ako je moguće (ovdje koristimo random split)
        indices = list(range(len(dataset)))
        train_indices, temp_indices = train_test_split(indices, test_size=val_size+test_size, random_state=42)
        val_indices, test_indices = train_test_split(temp_indices, test_size=test_size, random_state=42)
        
        # Stvaranje podskupova podataka
        train_subset = torch.utils.data.Subset(dataset, train_indices)
        val_subset = torch.utils.data.Subset(dataset, val_indices)
        test_subset = torch.utils.data.Subset(dataset, test_indices)
        
        print(f"Podjela dataseta: {len(train_subset)} treniranje, {len(val_subset)} validacija, {len(test_subset)} test")
    else:
        # Izračunaj validation split (80% train, 20% validation)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        
        # Koristi sklearn za split
        indices = list(range(len(dataset)))
        train_indices, val_indices = train_test_split(indices, test_size=val_size, random_state=42)
        
        # Stvaranje podskupova podataka
        train_subset = torch.utils.data.Subset(dataset, train_indices)
        val_subset = torch.utils.data.Subset(dataset, val_indices)
        
        print(f"Podjela dataseta: {len(train_subset)} treniranje, {len(val_subset)} validacija")
    
    # Stvaranje dataloadera
    train_dataloader = DataLoader(
        train_subset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True  # Za brži prijenos podataka na CUDA uređaj
    )
    
    val_dataloader = DataLoader(
        val_subset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    if test_split:
        test_dataloader = DataLoader(
            test_subset, 
            batch_size=config.batch_size, 
            shuffle=False, 
            num_workers=4,
            pin_memory=True
        )
        return train_dataloader, val_dataloader, test_dataloader
    else:
        return train_dataloader, val_dataloader
    

def get_stratified_data_loaders(config, special_ids, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Stvara dataloadere s stratificiranom podjelom koja osigurava da posebne slike
    budu ravnomjerno raspoređene među skupovima.
    
    Args:
        config: Konfiguracija modela
        special_ids: Lista IXI ID-ova koji predstavljaju "mutnije" slike
        train_ratio: Udio podataka za trening (default: 0.7)
        val_ratio: Udio podataka za validaciju (default: 0.15)
        test_ratio: Udio podataka za testiranje (default: 0.15)
    
    Returns:
        (train_dataloader, val_dataloader, test_dataloader)
    """
    # Provjera omjera
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-10, "Omjeri moraju davati zbroj 1.0"
    
    # Transformacije
    transform = get_transforms(config, use_padding=True)
    
    # Dataset
    dataset = MRIDataset(config.t1_dir, config.t2_dir, transform=transform)
    
    # Izdvoji indekse posebnih slika i regularnih slika
    special_indices = []
    regular_indices = []
    
    for i, data in enumerate(dataset):
        # Dobavi ID iz putanje datoteke
        file_path = data["A_path"]  # T1 putanja
        file_name = os.path.basename(file_path)
        
        # Pretpostavljam da je format imena "IXI123_T1_slice.png"
        ixi_id = file_name.split('_')[0].replace("IXI", "")
        
        if ixi_id in special_ids:
            special_indices.append(i)
        else:
            regular_indices.append(i)
    
    print(f"Pronađeno {len(special_indices)} posebnih slika i {len(regular_indices)} regularnih slika.")
    
    # Podjela posebnih slika
    special_train_size = int(len(special_indices) * train_ratio)
    special_val_size = int(len(special_indices) * val_ratio)
    special_test_size = len(special_indices) - special_train_size - special_val_size
    
    # Nasumična podjela posebnih slika
    random.shuffle(special_indices)
    
    special_train = special_indices[:special_train_size]
    special_val = special_indices[special_train_size:special_train_size + special_val_size]
    special_test = special_indices[special_train_size + special_val_size:]
    
    # Podjela regularnih slika
    regular_train_size = int(len(regular_indices) * train_ratio)
    regular_val_size = int(len(regular_indices) * val_ratio)
    regular_test_size = len(regular_indices) - regular_train_size - regular_val_size
    
    # Nasumična podjela regularnih slika
    random.shuffle(regular_indices)
    
    regular_train = regular_indices[:regular_train_size]
    regular_val = regular_indices[regular_train_size:regular_train_size + regular_val_size]
    regular_test = regular_indices[regular_train_size + regular_val_size:]
    
    # Kombiniraj indekse
    train_indices = special_train + regular_train
    val_indices = special_val + regular_val
    test_indices = special_test + regular_test
    
    # Ponovno izmiješaj kombinirane indekse
    random.shuffle(train_indices)
    random.shuffle(val_indices)
    random.shuffle(test_indices)
    
    print(f"Podjela posebnih slika: {len(special_train)} trening, {len(special_val)} validacija, {len(special_test)} test")
    print(f"Podjela regularnih slika: {len(regular_train)} trening, {len(regular_val)} validacija, {len(regular_test)} test")
    print(f"Ukupna podjela: {len(train_indices)} trening, {len(val_indices)} validacija, {len(test_indices)} test")
    
    # Stvaranje podskupova podataka
    train_subset = torch.utils.data.Subset(dataset, train_indices)
    val_subset = torch.utils.data.Subset(dataset, val_indices)
    test_subset = torch.utils.data.Subset(dataset, test_indices)
    
    # Stvaranje dataloadera
    train_dataloader = DataLoader(
        train_subset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    val_dataloader = DataLoader(
        val_subset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    test_dataloader = DataLoader(
        test_subset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    return train_dataloader, val_dataloader, test_dataloader

def sample_images(epoch, G_AB, G_BA, val_dataloader, config):
    """Generiranje uzorka slika i spremanje"""
    G_AB.eval()
    G_BA.eval()
    
    with torch.no_grad():
        batch = next(iter(val_dataloader))
        real_A = batch["A"].to(config.device)
        fake_B = G_AB(real_A)
        real_B = batch["B"].to(config.device)
        fake_A = G_BA(real_B)
        
        # Također izračunaj rekonstrukcije ciklusa
        rec_A = G_BA(fake_B)
        rec_B = G_AB(fake_A)
        
        # Konvertiraj u slike u rasponu [0, 1]
        real_A = (real_A * 0.5 + 0.5).cpu()
        fake_B = (fake_B * 0.5 + 0.5).cpu()
        rec_A = (rec_A * 0.5 + 0.5).cpu()
        real_B = (real_B * 0.5 + 0.5).cpu()
        fake_A = (fake_A * 0.5 + 0.5).cpu()
        rec_B = (rec_B * 0.5 + 0.5).cpu()
        
        # Stvori mrežu slika
        image_grid = make_grid([
            real_A[0], fake_B[0], rec_A[0],
            real_B[0], fake_A[0], rec_B[0]
        ], nrow=3, normalize=False)
        
        # Spremi mrežu slika
        plt.figure(figsize=(15, 10))
        plt.imshow(image_grid.permute(1, 2, 0).numpy(), cmap='gray')
        plt.axis('off')
        plt.title(f'Epoha {epoch}')
        plt.tight_layout()
        plt.savefig(f"{config.results_dir}/epoch_{epoch}.png")
        plt.close()

    G_AB.train()
    G_BA.train()
    
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

def test_model(G_AB, G_BA, test_dataloader, config):
    """
    Testiranje istreniranog modela na test setu
    
    Args:
        G_AB: Generator za T1->T2 translaciju
        G_BA: Generator za T2->T1 translaciju
        test_dataloader: DataLoader za test set
        config: Konfiguracija modela
    """
    # Postavi modele u eval način rada
    G_AB.eval()
    G_BA.eval()
    
    # Metrike
    metrics_t1_to_t2 = {"ssim": [], "mse": [], "psnr": []}
    metrics_t2_to_t1 = {"ssim": [], "mse": [], "psnr": []}
    metrics_cycle_t1 = {"ssim": [], "mse": [], "psnr": []}
    metrics_cycle_t2 = {"ssim": [], "mse": [], "psnr": []}
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluacija modela"):
            real_A = batch["A"].to(config.device)  # T1
            real_B = batch["B"].to(config.device)  # T2
            
            # Generiraj translacije
            fake_B = G_AB(real_A)  # T1 -> T2
            fake_A = G_BA(real_B)  # T2 -> T1
            
            # Generiraj rekonstrukcije
            rec_A = G_BA(fake_B)  # T1 -> T2 -> T1
            rec_B = G_AB(fake_A)  # T2 -> T1 -> T2
            
            # Normaliziraj za izračun metrika
            real_A_norm = (real_A * 0.5 + 0.5)
            real_B_norm = (real_B * 0.5 + 0.5)
            fake_A_norm = (fake_A * 0.5 + 0.5)
            fake_B_norm = (fake_B * 0.5 + 0.5)
            rec_A_norm = (rec_A * 0.5 + 0.5)
            rec_B_norm = (rec_B * 0.5 + 0.5)
            
            # Izračunaj metrike za svaku sliku u batchu
            for i in range(real_A.size(0)):
                # T1 -> T2 translacija
                metrics = calculate_metrics(real_B_norm[i].squeeze(), fake_B_norm[i].squeeze())
                metrics_t1_to_t2["ssim"].append(metrics["ssim"])
                metrics_t1_to_t2["mse"].append(metrics["mse"])
                metrics_t1_to_t2["psnr"].append(metrics["psnr"])
                
                # T2 -> T1 translacija
                metrics = calculate_metrics(real_A_norm[i].squeeze(), fake_A_norm[i].squeeze())
                metrics_t2_to_t1["ssim"].append(metrics["ssim"])
                metrics_t2_to_t1["mse"].append(metrics["mse"])
                metrics_t2_to_t1["psnr"].append(metrics["psnr"])
                
                # T1 -> T2 -> T1 ciklus
                metrics = calculate_metrics(real_A_norm[i].squeeze(), rec_A_norm[i].squeeze())
                metrics_cycle_t1["ssim"].append(metrics["ssim"])
                metrics_cycle_t1["mse"].append(metrics["mse"])
                metrics_cycle_t1["psnr"].append(metrics["psnr"])
                
                # T2 -> T1 -> T2 ciklus
                metrics = calculate_metrics(real_B_norm[i].squeeze(), rec_B_norm[i].squeeze())
                metrics_cycle_t2["ssim"].append(metrics["ssim"])
                metrics_cycle_t2["mse"].append(metrics["mse"])
                metrics_cycle_t2["psnr"].append(metrics["psnr"])

                metrics_crop = calculate_metrics_fixed_crop(real_B_norm[i].squeeze(), fake_B_norm[i].squeeze())
                metrics_t1_to_t2["ssim_crop"].append(metrics_crop["ssim"])
                metrics_t1_to_t2["mse_crop"].append(metrics_crop["mse"])
                metrics_t1_to_t2["psnr_crop"].append(metrics_crop["psnr"])

                metrics_crop = calculate_metrics_fixed_crop(real_A_norm[i].squeeze(), fake_A_norm[i].squeeze())
                metrics_t2_to_t1["ssim_crop"].append(metrics_crop["ssim"])
                metrics_t2_to_t1["mse_crop"].append(metrics_crop["mse"])
                metrics_t2_to_t1["psnr_crop"].append(metrics_crop["psnr"])

                metrics_crop = calculate_metrics_fixed_crop(real_A_norm[i].squeeze(), rec_A_norm[i].squeeze())
                metrics_cycle_t1["ssim_crop"].append(metrics_crop["ssim"])
                metrics_cycle_t1["mse_crop"].append(metrics_crop["mse"])
                metrics_cycle_t1["psnr_crop"].append(metrics_crop["psnr"])

                metrics_crop = calculate_metrics_fixed_crop(real_B_norm[i].squeeze(), rec_B_norm[i].squeeze())
                metrics_cycle_t2["ssim_crop"].append(metrics_crop["ssim"])
                metrics_cycle_t2["mse_crop"].append(metrics_crop["mse"])
                metrics_cycle_t2["psnr_crop"].append(metrics_crop["psnr"])

    
    # Izračunaj prosječne metrike
    avg_metrics = {
        "T1->T2": {
            "ssim": np.mean(metrics_t1_to_t2["ssim"]),
            "mse": np.mean(metrics_t1_to_t2["mse"]),
            "psnr": np.mean(metrics_t1_to_t2["psnr"])
        },
        "T2->T1": {
            "ssim": np.mean(metrics_t2_to_t1["ssim"]),
            "mse": np.mean(metrics_t2_to_t1["mse"]),
            "psnr": np.mean(metrics_t2_to_t1["psnr"])
        },
        "T1->T2->T1": {
            "ssim": np.mean(metrics_cycle_t1["ssim"]),
            "mse": np.mean(metrics_cycle_t1["mse"]),
            "psnr": np.mean(metrics_cycle_t1["psnr"])
        },
        "T2->T1->T2": {
            "ssim": np.mean(metrics_cycle_t2["ssim"]),
            "mse": np.mean(metrics_cycle_t2["mse"]),
            "psnr": np.mean(metrics_cycle_t2["psnr"])
        }
    }

    agv_metrics_crop = {
        "T1->T2": {
            "ssim": np.mean(metrics_t1_to_t2["ssim_crop"]),
            "mse": np.mean(metrics_t1_to_t2["mse_crop"]),
            "psnr": np.mean(metrics_t1_to_t2["psnr_crop"])
        },
        "T2->T1": {
            "ssim": np.mean(metrics_t2_to_t1["ssim_crop"]),
            "mse": np.mean(metrics_t2_to_t1["mse_crop"]),
            "psnr": np.mean(metrics_t2_to_t1["psnr_crop"])
        },
        "T1->T2->T1": {
            "ssim": np.mean(metrics_cycle_t1["ssim_crop"]),
            "mse": np.mean(metrics_cycle_t1["mse_crop"]),
            "psnr": np.mean(metrics_cycle_t1["psnr_crop"])
        },
        "T2->T1->T2": {
            "ssim": np.mean(metrics_cycle_t2["ssim_crop"]),
            "mse": np.mean(metrics_cycle_t2["mse_crop"]),
            "psnr": np.mean(metrics_cycle_t2["psnr_crop"])
        }
    }
    
    # Ispiši rezultate
    print("\n=== Rezultati evaluacije ===")
    print("T1 -> T2 translacija:")
    print(f"  SSIM: {avg_metrics['T1->T2']['ssim']:.4f}")
    print(f"  MSE: {avg_metrics['T1->T2']['mse']:.6f}")
    print(f"  PSNR: {avg_metrics['T1->T2']['psnr']:.2f} dB")
    
    print("\nT2 -> T1 translacija:")
    print(f"  SSIM: {avg_metrics['T2->T1']['ssim']:.4f}")
    print(f"  MSE: {avg_metrics['T2->T1']['mse']:.6f}")
    print(f"  PSNR: {avg_metrics['T2->T1']['psnr']:.2f} dB")
    
    print("\nT1 -> T2 -> T1 rekonstrukcija:")
    print(f"  SSIM: {avg_metrics['T1->T2->T1']['ssim']:.4f}")
    print(f"  MSE: {avg_metrics['T1->T2->T1']['mse']:.6f}")
    print(f"  PSNR: {avg_metrics['T1->T2->T1']['psnr']:.2f} dB")
    
    print("\nT2 -> T1 -> T2 rekonstrukcija:")
    print(f"  SSIM: {avg_metrics['T2->T1->T2']['ssim']:.4f}")
    print(f"  MSE: {avg_metrics['T2->T1->T2']['mse']:.6f}")
    print(f"  PSNR: {avg_metrics['T2->T1->T2']['psnr']:.2f} dB")

    print("\n=== Rezultati evaluacije (fiksni crop) ===")
    print("T1 -> T2 translacija:")
    print(f"  SSIM: {agv_metrics_crop['T1->T2']['ssim']:.4f}")
    print(f"  MSE: {agv_metrics_crop['T1->T2']['mse']:.6f}")
    print(f"  PSNR: {agv_metrics_crop['T1->T2']['psnr']:.2f} dB")
    print("\nT2 -> T1 translacija:")
    print(f"  SSIM: {agv_metrics_crop['T2->T1']['ssim']:.4f}")
    print(f"  MSE: {agv_metrics_crop['T2->T1']['mse']:.6f}")
    print(f"  PSNR: {agv_metrics_crop['T2->T1']['psnr']:.2f} dB")
    print("\nT1 -> T2 -> T1 rekonstrukcija:")
    print(f"  SSIM: {agv_metrics_crop['T1->T2->T1']['ssim']:.4f}")
    print(f"  MSE: {agv_metrics_crop['T1->T2->T1']['mse']:.6f}")
    print(f"  PSNR: {agv_metrics_crop['T1->T2->T1']['psnr']:.2f} dB")
    print("\nT2 -> T1 -> T2 rekonstrukcija:")
    print(f"  SSIM: {agv_metrics_crop['T2->T1->T2']['ssim']:.4f}")
    print(f"  MSE: {agv_metrics_crop['T2->T1->T2']['mse']:.6f}")
    print(f"  PSNR: {agv_metrics_crop['T2->T1->T2']['psnr']:.2f} dB")
    

def train_cyclegan(config, use_test_split=True, train_dataloader=None, val_dataloader=None, test_dataloader=None):
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
            if config.architecture == "standard":
                # Standardni diskriminator
                valid = torch.ones((real_A.size(0), 1, 15, 15), requires_grad=False).to(config.device)
                fake = torch.zeros((real_A.size(0), 1, 15, 15), requires_grad=False).to(config.device)
            else:
                # Dublji diskriminator
                valid = torch.ones((real_A.size(0), 1, 7, 7), requires_grad=False).to(config.device)
                fake = torch.zeros((real_A.size(0), 1, 7, 7), requires_grad=False).to(config.device)
            
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
                if config.architecture == "standard":
                    valid = torch.ones((real_A.size(0), 1, 15, 15), device=config.device)
                else:
                    valid = torch.ones((real_A.size(0), 1, 7, 7), device=config.device)
                    
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
