from src.VAE.utils.imports import *

class TrainVAE:
    def __init__(
            self,
            model,
            optimizer, 
            epochs : int, 
            batch_size : int,
            data, 
            xdim : int,
            device, 
            save_images = False,
            extra_losses = None
        ) -> None:
        
        self.model = model
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.data = data
        self.xdim = xdim
        self.device = device
        self.extra_losses = extra_losses
        self.save_images = save_images
        self.writer = SummaryWriter("../runs/trajectory_experiment/LSTM-500-epochs")  # Create writer

    
    def vae_loss(self, x, x_hat, mean, logvar):
        reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction="sum")
        KLD = - 0.5 * torch.sum(1+ logvar - mean.pow(2) - logvar.exp())
        return reproduction_loss + KLD

    def sample_latent_space(self):
        """Samples random latent vectors from a standard normal distribution."""
        z = torch.randn(self.batch_size, self.model.latent_dim).to(self.device)
        return z
    
    def train_model(self):
#        print(self.data)
        for epoch in range(self.epochs):
            overall_loss = 0

            for batch_num, x in enumerate(self.data):
                x = x.to(self.device)
#                print(x.shape)
#                x = torch.flatten(x).to(self.device)

                self.optimizer.zero_grad()

                x_hat, mean, logvar = self.model(x)
                loss = self.vae_loss(x, x_hat, mean, logvar)

                if self.extra_losses != None:
                    for l in self.extra_losses:
                        loss += (l(x, x_hat))

                overall_loss += loss.item()

                loss.backward()
                self.optimizer.step()


            # Log loss to TensorBoard
            self.writer.add_scalar("Loss/train", overall_loss/(batch_num*self.batch_size), epoch)

            if self.save_images:
                with torch.no_grad():
                    # Sample from latent space (replace with your sampling function)
                    sampled_z = self.sample_latent_space()
                    sampled_images = self.model.decode(sampled_z)

                    # Normalize and reshape for TensorBoard (assuming image data)
                    grid = torchvision.utils.make_grid(
                        sampled_images.detach().cpu().view(-1, *self.xdim) / 2 + 0.5, normalize=True)

                    self.writer.add_image("LatentSpace/Samples", grid, epoch)

            
            print(f"Epoch {epoch + 1}: {overall_loss/(batch_num*self.batch_size)}")
        
        self.writer.close()
        return overall_loss