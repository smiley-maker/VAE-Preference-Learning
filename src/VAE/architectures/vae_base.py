import torch
import torch.nn as nn

class VAE_Model(nn.Module):
    def __init__(self, device, image_size, input_channels=3, hidden_dim=256, latent_dim=20, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = device
        self.latent_dim = latent_dim

        # Convolutional encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=5, stride=2, padding=2),  # Output: (16, image_size/2, image_size/2)
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),  # Output: (32, image_size/4, image_size/4)
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),  # Output: (64, image_size/8, image_size/8)
            nn.LeakyReLU(0.2)
        )

        # Calculate the size after convolutional layers
        self.conv_output_size = self._get_conv_output_size(image_size)
        self.flatten_size = 64 * self.conv_output_size * self.conv_output_size

        # Latent mean & variance
        self.mean_layer = nn.Linear(self.flatten_size, latent_dim)
        self.logvar_layer = nn.Linear(self.flatten_size, latent_dim)

        # Convolutional decoder
        self.decoder_fc = nn.Linear(latent_dim, self.flatten_size)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 16, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(16, input_channels, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.Sigmoid()
        )


    def _get_conv_output_size(self, image_size):
        # Helper function to determine the size after convolutional layers
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, image_size, image_size)
            dummy_output = self.encoder(dummy_input)
            output_size = dummy_output.size(2)  # Get spatial dimension (H or W)
        return output_size

    def encode(self, x):
        # Encodes x using the encoder
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # Flatten
        mean = self.mean_layer(x)
        logvar = self.logvar_layer(x)
        return mean, logvar
    
    def reparametrize(self, mean, logvar):
        std = torch.exp(logvar / 2)
        epsilon = torch.randn_like(std).to(self.device)
        return mean + std * epsilon
    
    def decode(self, z):
        # Decodes z using the decoder
        z = self.decoder_fc(z)
        z = z.view(z.size(0), 64, self.conv_output_size, self.conv_output_size)  # Reshape
        return self.decoder(z)
    
    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparametrize(mean, logvar)
        x_hat = self.decode(z)
        return x_hat, mean, logvar
