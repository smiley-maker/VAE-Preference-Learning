from src.VAE.utils.imports import *

class TVAE_Model(nn.Module):
    def __init__(self, device, input_size, hidden_dim=256, latent_dim=20, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = device
        self.latent_dim = latent_dim
        self.input_size = input_size

        # Convolutional encoder
        self.encoder = nn.Sequential(
            # 1D data
            nn.Linear(input_size, 16),
            nn.LeakyReLU(0.2),
            nn.Linear(16, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 64), 
            nn.LeakyReLU(0.2)
        )

        # Calculate the size after convolutional layers
#        self.conv_output_size = self._get_conv_output_size(image_size)
#        self.flatten_size = 64 * self.conv_output_size * self.conv_output_size

        # Latent mean & variance
        self.mean_layer = nn.Linear(64, latent_dim)
        self.logvar_layer = nn.Linear(64, latent_dim)

        # Convolutional decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 16),
            nn.LeakyReLU(0.2),
            nn.Linear(16, input_size), # Output size 
            nn.Sigmoid()
        )

    def encode(self, x):
        # Encodes x using the encoder
        x = self.encoder(x)
#        x = x.view(x.size(0), -1)  # Flatten
        mean = self.mean_layer(x)
        logvar = self.logvar_layer(x)
        return mean, logvar
    
    def reparametrize(self, mean, logvar):
        std = torch.exp(logvar / 2)
        epsilon = torch.randn_like(std).to(self.device)
        return mean + std * epsilon
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
#        x = x.view(-1, self.input_size)
        mean, logvar = self.encode(x)
        z = self.reparametrize(mean, logvar)
        x_hat = self.decode(z)
        return x_hat, mean, logvar
