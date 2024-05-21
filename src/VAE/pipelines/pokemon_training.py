from src.VAE.utils.imports import *
from src.VAE.architectures.vae_base import VAE_Model
from src.VAE.data_handlers.pokemon_handler import PokemonDataset
from src.VAE.training_scripts.train_vae import TrainVAE

def show_image(x):
    img = x[0]
    img = img.detach().numpy()
    img = img.transpose((1, 2, 0))
    im = Image.fromarray((img*255).astype(np.uint8)).convert('RGB')
    return im

def show_images_grid(images, title='Sample Images Grid'):
  '''
  show input torch tensor of images [num_images, ch, w, h] in a grid
  '''
  plt.figure(figsize=(7, 7))
  grid = vutils.make_grid(images, nrow=images.shape[0]//2, padding=2, normalize=True)
  plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
  plt.title(title)
  plt.axis('off')
  plt.show()


def posterior_sampling(model, data_loader, n_samples=25):
  model.eval()
  images, _ = next(iter(data_loader))
  images = images[:n_samples]
  with torch.no_grad():
    _, _, encoding_dist = model(images.to(device))
    input_sample=encoding_dist.sample()
    recon_images = model.decode(input_sample)
    show_images_grid(images, title=f'input samples')
    show_images_grid(recon_images, title=f'generated posterior samples')

def visualize_latent_space(model, data_loader, device, num_samples=100):
    model.eval()
    latents = []
    with torch.no_grad():
        for i, (data, _) in enumerate(data_loader):
          if len(latents) > num_samples:
            break
          data = data.to(device)#.view(data.size(0), -1)
          mu, _ = model.encode(data)
          latents.append(mu)

    latents = torch.cat(latents, dim=0).cpu().numpy()
    tsne = TSNE(n_components=2, verbose=1)
    tsne_results = tsne.fit_transform(latents)
    fig = px.scatter(tsne_results, x=0, y=1)
    fig.update_layout(title='VAE Latent Space with TSNE',
                        width=600,
                        height=600)

    fig.show()

def generate_and_show_images(model, device, num_images=10, latent_dim=20):
    model.eval()
    with torch.no_grad():
        # Generate random latent vectors
        z = torch.randn(num_images, latent_dim).to(device)
        
        # Decode the latent vectors to generate images
        generated_images = model.decode(z)
        
        # Reshape and move the images to CPU for visualization
        generated_images = generated_images.view(num_images, 3, 120, 120).cpu()

        # Plot the images
        fig, axes = plt.subplots(1, num_images, figsize=(10, 10))
        for i, ax in enumerate(axes):
            img = generated_images[i].numpy().transpose(1, 2, 0)
#            ax.imsave(img*255, "test_one.png")
            ax.imshow((img * 255).astype(np.uint8))
            ax.axis('off')

#        plt.savefig("test_one.png")
        plt.show()

if __name__ == "__main__":
    # Create device
    device = torch.device("mps")
    print(f"Using {device}")

    # Load model
    model = VAE_Model(
        # image dimension is (3, 120, 120)
        image_size=120,
        input_channels=3,
        hidden_dim=512,
        latent_dim=100,
        device=device
    ).to(device)

    # Load data
    dataset = PokemonDataset()
    dataloader = DataLoader(dataset, batch_size=32, drop_last=True)

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    
    # Train model
    trainer = TrainVAE(
        model=model,
        optimizer=optimizer,
        epochs=500,
        batch_size=32,
        data=dataloader, 
        xdim=(3,120,120),
        device=device
    )

    trainer.train_model()

    visualize_latent_space(model, dataloader, device, num_samples=800)

    # Generate and show images
    generate_and_show_images(
        model=model,
        device=device,
        num_images=5,
        latent_dim=100  # Make sure this matches the latent_dim in the VAE_Model
    )