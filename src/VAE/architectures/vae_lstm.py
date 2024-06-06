from src.VAE.utils.imports import *


class MemoryVAE(nn.Module):
    def __init__(self, batch_size : int,
                  terrain_classes_count : int, 
                  trajectory_length : int, 
                  latent_dim : int,
                  hidden_size : int,
                  device,
                  *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Model Size Parameters
        self.M = terrain_classes_count # M Terrain Types
        self.B = batch_size # Batch Size B
        self.L = trajectory_length # Trajectory Length L
        self.hidden_size = hidden_size # Hidden Dimensions for LSTM
        self.latent_dim = latent_dim # VAE Latent Encoding Dimension(s)

        # Device
        self.device = device

        # Defining Encoder Layers
        self.encoder_lstm = nn.LSTM(
            input_size=self.L,#*self.B,
            hidden_size=self.hidden_size,
            batch_first=True,
#            bidirectional=False
        )

        self.encoder_meanvar = nn.Linear(
            in_features=self.hidden_size,#(self.B, 1, self.hidden_size),
            out_features=self.latent_dim*2#(self.B, self.latent_dim*2)
        )

        # Decoder Layers
        self.decoder_input = nn.Linear(
            in_features=self.latent_dim,#(self.B, self.latent_dim),
            out_features=self.hidden_size#(self.B, 1, self.hidden_size)
        )

        self.decoder_lstm = nn.LSTM(
            input_size=self.latent_dim,#(self.B, 1, self.hidden_size),
            hidden_size=self.hidden_size,
            batch_first=True
        )

        self.decoder_output = nn.Linear(
            in_features=self.hidden_size,#self.L*self.hidden_size,#(self.B, self.L, self.hidden_size),
            out_features=self.L#*self.M#(self.B, self.L, self.M)
        )

        # Utilities
        self.dropout = nn.Dropout(p=0.2)
    
    def encode(self, x):
        # Encodes x into a latent space vector represented by mu and sigma
        x = self.dropout(x)
        x, _ = self.encoder_lstm(x)
        combined_mu_sigma = self.encoder_meanvar(x)
        mu, logvar = torch.split(combined_mu_sigma, self.latent_dim, dim=1)

        return mu, logvar
    
    def reparametrize(self, mean, logvar):
        std = torch.exp(logvar / 2)
        epsilon = torch.randn_like(std).to(self.device)
        return mean + std*epsilon
    
    def decode(self, z):
#        z = self.decoder_input(z)
#        z = self.dropout(z)
        z, _ = self.decoder_lstm(z)
        z = self.decoder_output(z)
        return z
#        return nn.Softmax(z)
    
    def forward(self, x):
        print(x.shape)
        mean, logvar = self.encode(x)
        std_mean = self.reparametrize(mean, logvar)
        x_hat = self.decode(std_mean)
#        print(x_hat)
#        print(mean)
#        print(std_mean)
#        print(logvar)
        return x_hat, mean, logvar