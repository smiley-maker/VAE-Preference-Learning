from src.VAE.utils.imports import *

class VAE_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Define the encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, latent_dim),
            nn.LeakyReLU(0.2)
        )

        # Latent mean & variance
        self.mean_layer = nn.Linear(latent_dim, 2)
        self.logvar_layer = nn.Linear(latent_dim, 2)

        # Define the decoder
        self.decoder = nn.Sequential(
            nn.Linear(2, latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def encode(self, x):
        # Encodes x using the encoder
        x = self.encoder(x)
        mean, logvar = self.mean_layer(x), self.logvar_layer(x)
        return mean, logvar
    
    def reparametrize(self, mean, var):
        epsilon = torch.randn_like(var)
        z = mean + var*epsilon
        return z
    
    def decode(self, x):
        # Decodes x using the decoder
        return self.decoder(x)
    
    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparametrize(mean, logvar)
        x_hat = self.decode(z)
        return x_hat, mean, logvar
