import os
import torch

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