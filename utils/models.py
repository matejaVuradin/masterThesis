import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

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
        
        # Decoder sa skip vezama
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
        # stop na 2x2, nema down8 i down9
        
        # residual blok za dubinu
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(features * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features * 8, features * 8, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(features * 8)
        )

        # jedna up operacija manje
        self.up1 = UNetUp(features * 8, features * 8, dropout=0.5)
        self.up2 = UNetUp(features * 16, features * 8, dropout=0.5)
        self.up3 = UNetUp(features * 16, features * 8, dropout=0.5)
        self.up4 = UNetUp(features * 16, features * 4)
        self.up5 = UNetUp(features * 8, features * 2)
        self.up6 = UNetUp(features * 4, features)
        
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
        d7 = self.down7(d6)  # [512, 2, 2]
        
        # Bottleneck - residual veza
        bottle = d7 + self.bottleneck(d7)
        
        # Decoder sa skip vezama
        u1 = self.up1(bottle, d6)
        u2 = self.up2(u1, d5)
        u3 = self.up3(u2, d4)
        u4 = self.up4(u3, d3)
        u5 = self.up5(u4, d2)
        u6 = self.up6(u5, d1)
        
        return self.final(u6)

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
            nn.Conv2d(features * 8, 1, 4, padding=1)
            # Izlaz: 1 x 15 x 15
        )

    def forward(self, img):
        return self.model(img)
    

class VGG19FeatureExtractor(nn.Module):
    def __init__(self, requires_grad=False):
        super(VGG19FeatureExtractor, self).__init__()
        vgg19 = models.vgg19(weights=models.VGG19_Weights.DEFAULT) #pre-trenirani VGG19 model
        
        # Koristimo samo značajke (features) a ne klasifikacijske slojeve
        self.vgg = nn.Sequential(*list(vgg19.features.children())[:20])  # prvih 20 slojeva
        
        # ne treniramo enkoder
        if not requires_grad:
            for param in self.vgg.parameters():
                param.requires_grad = False
                    
    def forward(self, x):
        features = [] #za skip veze
        
        # Prolazak kroz slojeve VGG19
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            
            # Spremimo značajke nakon određenih konvolucija
            if i == 3:   # Prvi blok (64 kanala)
                features.append(x)
            elif i == 8:  # Drugi blok (128 kanala)
                features.append(x)
            elif i == 15: # Treći blok (256 kanala)
                features.append(x)
            elif i == 19: # Četvrti blok (512 kanala)
                features.append(x)
        
        return features


class VGG19Generator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=64):
        super(VGG19Generator, self).__init__()
        
        # grayscale ulaz -> RGB za VGG19
        self.input_adapter = nn.Conv2d(in_channels, 3, kernel_size=1, stride=1, padding=0)
        
        self.encoder = VGG19FeatureExtractor(requires_grad=False)
        
        # sluzi kao learning adapter za skip veze
        self.adapt_512 = nn.Conv2d(512, 512, kernel_size=1)  # Najdublji sloj
        self.adapt_256 = nn.Conv2d(256, 256, kernel_size=1)  # Treći sloj
        self.adapt_128 = nn.Conv2d(128, 128, kernel_size=1)  # Drugi sloj
        self.adapt_64 = nn.Conv2d(64, 64, kernel_size=1)    # Prvi sloj
        
        # Decoder
        self.upconv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.norm1 = nn.InstanceNorm2d(256)
        
        # upconv1 i concat s adapt_256 imamo 256+256=512 kanala
        self.upconv2 = nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1)
        self.norm2 = nn.InstanceNorm2d(128)
        
        # upconv2 i concat s adapt_128 imamo 128+128=256 kanala
        self.upconv3 = nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1)
        self.norm3 = nn.InstanceNorm2d(64)
        
        # upconv3 i concat s adapt_64 imamo 64+64=128 kanala na 256x256
        self.final_conv = nn.Conv2d(128, 32, kernel_size=3, padding=1)
        self.norm4 = nn.InstanceNorm2d(32)
        
        self.out_conv = nn.Conv2d(32, out_channels, kernel_size=7, stride=1, padding=3)
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        x_rgb = self.input_adapter(x) # [1, 256, 256] -> [3, 256, 256]	
        vgg_features = self.encoder(x_rgb) #značajke iz VGG19
        
        # Značajke iz VGG19: vgg_features[0] = 64ch (256x256), vgg_features[1] = 128ch (128x128), 
        # vgg_features[2] = 256ch (64x64), vgg_features[3] = 512ch (32x32)
        
        feat_512 = self.adapt_512(vgg_features[3])  # 512ch, 32x32
        feat_256 = self.adapt_256(vgg_features[2])  # 256ch, 64x64
        feat_128 = self.adapt_128(vgg_features[1])  # 128ch, 128x128
        feat_64 = self.adapt_64(vgg_features[0])    # 64ch, 256x256
        
        # Dekoder s skip konekcijama
        x = self.relu(self.norm1(self.upconv1(feat_512)))  # 512ch -> 256ch, 32x32 -> 64x64
        x = torch.cat([x, feat_256], dim=1)  # 256ch + 256ch = 512ch, 64x64
        
        x = self.relu(self.norm2(self.upconv2(x)))  # 512ch -> 128ch, 64x64 -> 128x128
        x = torch.cat([x, feat_128], dim=1)  # 128ch + 128ch = 256ch, 128x128
        
        x = self.relu(self.norm3(self.upconv3(x)))  # 256ch -> 64ch, 128x128 -> 256x256
        x = torch.cat([x, feat_64], dim=1)  # 64ch + 64ch = 128ch, 256x256
        
        x = self.relu(self.norm4(self.final_conv(x)))  # 128ch -> 32ch, veličina ostaje 256x256
    
        x = self.tanh(self.out_conv(x))  # 32ch -> 1ch, veličina ostaje 256x256
        
        return x