import torch
import torch.nn as nn
import torch.nn.functional as F

# define the NN architecture
class ConvAutoencoder_GELU(nn.Module):
    def __init__(self, device, z_dim, include_noise):
        super().__init__()
        self.device = device
        self.include_noise = include_noise

        self.encoder_conv2D = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            nn.GELU()
        )

        ## Flatten Layer
        self.flatten = nn.Flatten(start_dim=1)
        
        ## Linear Section
        self.encoder_linear = nn.Sequential(
            nn.Linear(3 * 3 * 32, 128),
            nn.GELU(),
            nn.Linear(128, z_dim),
        )

        self.decoder_linear = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.GELU(),
            nn.Linear(128, 3 * 3 * 32),
            nn.GELU(),
        )

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, 3, 3))

        self.decoder_convt2d = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=0),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.GELU(),
            nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1),
        )


    def forward(self, x):
        ## Encode the image to latent space
        x = self.encoder_conv2D(x)
        x = self.flatten(x)
        x = self.encoder_linear(x)

        ## Add random gaussian noise to latent encoding
        if self.include_noise:
            noise = torch.randn(x.shape).to(self.device)
            x = x + noise

        ## Decode the latent encoding back to reconstructed image
        x = self.decoder_linear(x)
        x = self.unflatten(x)
        x = self.decoder_convt2d(x)
        x = torch.sigmoid(x)
                
        return x
    
class ConvAutoencoder_ReLU(nn.Module):
    def __init__(self, device, z_dim, include_noise):
        super().__init__()
        self.device = device
        self.include_noise = include_noise

        self.encoder_conv2D = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            nn.ReLU(True),
        )

        ## Flatten Layer
        self.flatten = nn.Flatten(start_dim=1)
        
        ## Linear Section
        self.encoder_linear = nn.Sequential(
            nn.Linear(3 * 3 * 32, 128),
            nn.ReLU(True),
            nn.Linear(128, z_dim),
        )

        self.decoder_linear = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 3 * 3 * 32),
            nn.ReLU(True),
        )

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, 3, 3))

        self.decoder_convt2d = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1),
        )


    def forward(self, x):
        ## Encode the image to latent space
        x = self.encoder_conv2D(x)
        x = self.flatten(x)
        x = self.encoder_linear(x)

        ## Add random gaussian noise to latent encoding
        if self.include_noise:
            noise = torch.randn(x.shape).to(self.device)
            x = x + noise

        ## Decode the latent encoding back to reconstructed image
        x = self.decoder_linear(x)
        x = self.unflatten(x)
        x = self.decoder_convt2d(x)
        x = torch.sigmoid(x)
                
        return x