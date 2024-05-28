from src.VAE.utils.imports import *
from src.VAE.data_handlers.trajectory_segment_gatherer import Segment, SegmentCollection, TrajectoryDataset
from src.VAE.training_scripts.train_vae import TrainVAE
from src.VAE.architectures.tvae_base import TVAE_Model

def collate(batch):
    return torch.stack(batch)

if __name__ == "__main__":
    # Create device
    device = torch.device("mps")
    print(f"Using {device}")

    categories = ["Grass", "Road", "Sidewalk", "Water", "Trees"]

    grid_size = (120, 120)

    random_array = np.random.randint(0, len(categories), size=grid_size)
    costmap = np.vectorize(lambda x: categories[x])(random_array)

    segment_size = 10
    num_categories = len(categories)

    # Load model
    model = TVAE_Model(
#        input_size=(10, len(categories)),
        input_size=num_categories*segment_size*32 ,
        hidden_dim=512,
        latent_dim=100,
        device=device
    ).to(device)

    # Load data
    dataset = TrajectoryDataset(
        data=costmap,
        num_samples=321,
        segment_length=segment_size-1,
        terrain_types=categories
    )

    dataloader = DataLoader(dataset, batch_size=32, drop_last=True, collate_fn=collate)

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    
    # Train model
    trainer = TrainVAE(
        model=model,
        optimizer=optimizer,
        epochs=20,
        batch_size=32,
        data=dataloader, 
        xdim=100, # Maybe trajectory size? 
        device=device
    )

    print("Created trainer")

    trainer.train_model()