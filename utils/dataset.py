import os
import glob
import random
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from .config import CycleGANConfig

class MRIDataset(Dataset):
    def __init__(self, t1_dir, t2_dir, transform=None):
        self.transform = transform
        self.t1_file_paths = sorted(glob.glob(os.path.join(t1_dir, "*.png"))) #nadi sve png slike
        self.t2_file_paths = sorted(glob.glob(os.path.join(t2_dir, "*.png")))
        
        print(f"Pronađeno {len(self.t1_file_paths)} T1 slika i {len(self.t2_file_paths)} T2 slika")
        
        self.t1_ids = [os.path.basename(f).split('_')[0].replace("IXI", "") for f in self.t1_file_paths]
        self.t2_ids = [os.path.basename(f).split('_')[0].replace("IXI", "") for f in self.t2_file_paths]
        
        # pronalaženje zajedničkih ID-ova
        common_ids = list(set(self.t1_ids) & set(self.t2_ids))
        if not common_ids:
            raise ValueError("Nije pronađen nijedan par T1/T2 slika!")
            
        print(f"Pronađeno {len(common_ids)} sparenih T1/T2 slika")
        
        # mapiranje između ID-ova i putanja
        t1_id_to_path = {os.path.basename(f).split('_')[0].replace("IXI", ""): f for f in self.t1_file_paths}
        t2_id_to_path = {os.path.basename(f).split('_')[0].replace("IXI", ""): f for f in self.t2_file_paths}
        
        # zadrži samo sparenih slika
        self.t1_files = [t1_id_to_path[id] for id in common_ids]
        self.t2_files = [t2_id_to_path[id] for id in common_ids]
        
        # sortiraj po ID-u pacijenta
        sorted_indices = sorted(range(len(common_ids)), key=lambda i: common_ids[i])
        self.t1_files = [self.t1_files[i] for i in sorted_indices]
        self.t2_files = [self.t2_files[i] for i in sorted_indices]
        
    def __getitem__(self, index):
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

#opcionalno, koristili su drugim radovima (ima li smisla?)
class PadToSize:
    def __init__(self, size=256):
        self.size = size
        
    def __call__(self, img):
        return pad_to_size(img, self.size)

def get_transforms(config: CycleGANConfig, use_padding=True):
    if use_padding:
        return transforms.Compose([
            PadToSize(config.img_size),
            transforms.ToTensor(), #normalizacija na [0, 1]
            transforms.Normalize([0.5], [0.5])  # Normalizacija na [-1, 1]
        ])
    else:
        return transforms.Compose([
            transforms.Resize((config.img_size, config.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # Normalizacija na [-1, 1]
        ])
    
def get_data_loaders(config: CycleGANConfig, test_split=True):
    # Transformacije
    transform = get_transforms(config, use_padding=True)
    # Dataset
    dataset = MRIDataset(config.t1_dir, config.t2_dir, transform=transform)
    
    if test_split:
        # Izračunaj train/val/test split (70%/15%/15%)
        train_size = int(0.7 * len(dataset))
        val_size = int(0.15 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        
        indices = list(range(len(dataset)))
        train_indices, temp_indices = train_test_split(indices, test_size=val_size+test_size, random_state=42)
        val_indices, test_indices = train_test_split(temp_indices, test_size=test_size, random_state=42)

        train_subset = torch.utils.data.Subset(dataset, train_indices)
        val_subset = torch.utils.data.Subset(dataset, val_indices)
        test_subset = torch.utils.data.Subset(dataset, test_indices)
        
        print(f"Podjela dataseta: {len(train_subset)} treniranje, {len(val_subset)} validacija, {len(test_subset)} test")
    else:
        # Izračunaj validation split (80% train, 20% validation)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
    
        indices = list(range(len(dataset)))
        train_indices, val_indices = train_test_split(indices, test_size=val_size, random_state=42)
        
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
    

def get_stratified_data_loaders(config: CycleGANConfig, special_ids, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15): #kada imamo posebne slike koje želimo zadržati u svim skupovima
    transform = get_transforms(config, use_padding=True)

    dataset = MRIDataset(config.t1_dir, config.t2_dir, transform=transform)
    
    special_indices = []
    regular_indices = []
    
    for i, data in enumerate(dataset):
        file_path = data["A_path"]  # T1 putanja
        file_name = os.path.basename(file_path)
        
        # npr. "IXI123_T1_slice.png"
        ixi_id = file_name.split('_')[0].replace("IXI", "")
        
        if ixi_id in special_ids:
            special_indices.append(i)
        else:
            regular_indices.append(i)
    
    print(f"Pronađeno {len(special_indices)} posebnih slika i {len(regular_indices)} regularnih slika.")
    
    special_train_size = int(len(special_indices) * train_ratio)
    special_val_size = int(len(special_indices) * val_ratio)
    special_test_size = len(special_indices) - special_train_size - special_val_size

    random.shuffle(special_indices)
    
    special_train = special_indices[:special_train_size]
    special_val = special_indices[special_train_size:special_train_size + special_val_size]
    special_test = special_indices[special_train_size + special_val_size:]
    
    regular_train_size = int(len(regular_indices) * train_ratio)
    regular_val_size = int(len(regular_indices) * val_ratio)
    regular_test_size = len(regular_indices) - regular_train_size - regular_val_size
    
    random.shuffle(regular_indices)
    
    regular_train = regular_indices[:regular_train_size]
    regular_val = regular_indices[regular_train_size:regular_train_size + regular_val_size]
    regular_test = regular_indices[regular_train_size + regular_val_size:]
    
    # Kombiniraj indekse
    train_indices = special_train + regular_train
    val_indices = special_val + regular_val
    test_indices = special_test + regular_test

    random.shuffle(train_indices)
    random.shuffle(val_indices)
    random.shuffle(test_indices)
    
    print(f"Podjela posebnih slika: {len(special_train)} trening, {len(special_val)} validacija, {len(special_test)} test")
    print(f"Podjela regularnih slika: {len(regular_train)} trening, {len(regular_val)} validacija, {len(regular_test)} test")
    print(f"Ukupna podjela: {len(train_indices)} trening, {len(val_indices)} validacija, {len(test_indices)} test")
    
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