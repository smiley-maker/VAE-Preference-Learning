import aprel.basics
import aprel.querying
from src.VAE.utils.imports import *
from src.VAE.data_handlers.trajectory_segment_gatherer import TrajectoryDataset
from src.VAE.training_scripts.train_vae import TrainVAE
from src.VAE.architectures.tvae_base import TVAE_Model

def collate(batch):
    return torch.stack(batch)

def visualize_latent_space(model, data_loader, device, num_samples=100):
    model.eval()
    latents = []
    with torch.no_grad():
        for i, data in enumerate(data_loader):
          data = torch.flatten(data).to(device)

          if len(latents) > num_samples:
            break
          #data = data.to(device)#.view(data.size(0), -1)
          mu, _ = model.encode(data)
          latents.append(mu)

    latents = torch.cat(latents, dim=0).cpu().numpy()
    latents = latents.reshape(1, -1)
    tsne = TSNE(n_components=2, verbose=1)
    tsne_results = tsne.fit_transform(latents)
    fig = px.scatter(tsne_results, x=0, y=1)
    fig.update_layout(title='VAE Latent Space with TSNE',
                        width=600,
                        height=600)

    fig.show()


def cluster_latent_space(model, data_loader, device):
    model.eval()
    latents = []
    
    for data in data_loader:
       data = torch.flatten(data).to(device)
       mu, var = model.encode(data)
       std_mean = model.reparametrize(mu, var)
       latents.append(std_mean)

#    with torch.no_grad():
#        for i, data in enumerate(data_loader):
          # Flattens data so it's the appropriate input size for encoder. 
#          data = torch.flatten(data).to(device)
          
          # Encodes data into mu and log variance (latent space representation)
#          mu, var = model.encode(data)
          # Reparametrizes the latent space representation to be better aligned. 
#          standardized_mean = model.reparametrize(mu, var)
          # Stores the latent representation for this trajectory. 
#          latents.append(standardized_mean)

    latents = torch.cat(latents, dim=0).cpu().numpy()
#    latents = latents.reshape(1, -1)
#    latents = latents[:64]

    # Clustering
    km = KMeans(n_clusters=5, random_state=0, n_init="auto").fit(latents)
#    centers = km.cluster_centers_
    labels = km.labels_
    return labels


def feature_func(traj : Trajectory):
   # This function returns the terrain feature breakdown for a trajectory. 
   # Essentially count of each terrain type. 
   features = []
   terrain_types = np.unique(traj)
   for t in terrain_types:
    count = np.count_nonzero(traj == t)
    features.append(count/len(traj))
    
   return np.array(features)

if __name__ == "__main__":
    # Create device
    device = torch.device("mps")
    print(f"Using {device}")

    categories = ["Grass", "Road", "Sidewalk", "Water", "Trees"]

    grid_size = (900, 900)

    random_array = np.random.randint(0, len(categories), size=grid_size)
    costmap = np.vectorize(lambda x: categories[x])(random_array)

    segment_size = 15
    num_categories = len(categories)

    # Load model
    model = TVAE_Model(
#        input_size=(10, len(categories)),
        input_size=num_categories*segment_size*32 ,
        hidden_dim=512,
        latent_dim=num_categories*segment_size,
        device=device
    ).to(device)

    # Load data
    dataset = TrajectoryDataset(
        data=costmap,
        num_samples=65,
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
        epochs=5,
        batch_size=32,
        data=dataloader, 
        xdim=num_categories*segment_size, # Maybe trajectory size? 
        device=device
    )

    print("Created trainer, training....")
    trainer.train_model()

    # Cluster the latent space representations of the original dataset. 
    clusters = cluster_latent_space(model, dataset, device)

    # Now we can use APREL with the clustered trajectory data. 
    
    # Creates a fake environment
    env = NonGymEnvironment(feature_func)

    # Converting data set to TrajectorySet
    trajectories = []
    for d in dataset:
        if d != None:
            trajectories.append(
                Trajectory(env, list(d))
            )
        else:
           break

    
    trajectory_set = TrajectorySet(trajectories)
    print("Trajectory Set has Size::::::::::::")
    print(trajectory_set.size)

    # Creates a query optimizer
    # Will likely need to convert the dataset to work with Aprel's trajectory set class
    query_optimizer = QueryOptimizerDiscreteTrajectorySet(trajectory_set)

    # Creates a human user manager
    true_user = HumanUser(delay=0.5)

    # Creates a weight vector with randomized parameters
    params = {"weights": get_random_normalized_vector(num_categories)}

    # Creates a user model
    user_model = SoftmaxUser(params)

    # Initializes the belief network using the user model and parameters
    belief = SamplingBasedBelief(user_model, [], params)

    # Creates an example query using the first two trajectories. 
    query = PreferenceQuery(trajectory_set[:2])

    # Learning Loop
    print("starting learning loop.......")
    
    for query_no in range(10):
        queries, objective_values = query_optimizer.optimize('variational', belief, query, clusters=clusters)
        print('Objective Value: ' + str(objective_values[0]))
        
        responses = true_user.respond(queries[0])
        belief.update(Preference(queries[0], responses[0]))
        print('Estimated user parameters: ' + str(belief.mean))

