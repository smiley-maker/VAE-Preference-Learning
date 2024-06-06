from src.VAE.utils.imports import *
from src.VAE.data_handlers.trajectory_segment_gatherer import TrajectoryDataset
from src.VAE.training_scripts.train_vae import TrainVAE
from src.VAE.architectures.vae_lstm import MemoryVAE
from src.VAE.utils.moving_average import moving_average
from src.VAE.vizualization_scripts.ClusteredTSNE import cluster_latent_space

def collate(batch):
    return torch.stack(batch)


def feature_func(traj : Trajectory):
   # This function returns the terrain feature breakdown for a trajectory. 
   # Essentially count of each terrain type. 
    features = []
    categories = ["Grass", "Road", "Sidewalk", "Water", "Trees", "Rock", "Brush", "Sand"]
    mapping = {
            t : i for i,t in enumerate(categories)
        }
    
    traj = np.array(traj, dtype=int)

    for t in categories:
        count = np.count_nonzero(traj == mapping[t])
        features.append(count/len(traj))
    
    return np.array(features)

def run_experiment():
    pass

if __name__ == "__main__":
    # Create device
    device = torch.device("mps")
    print(f"Using {device}")

    categories = ["Grass", "Road", "Sidewalk", "Water", "Trees", "Rock", "Brush", "Sand"]
    mapping_index_to_terrain = {
            i : t for i,t in enumerate(categories)
        }


    grid_size = (900, 900)

    random_array = np.random.randint(0, len(categories), size=grid_size)
    costmap = np.vectorize(lambda x: categories[x])(random_array)

    segment_size = 16
    num_categories = len(categories)
    batch_size = 32

    # Load LSTM Model
    model = MemoryVAE(
       batch_size=batch_size,
       terrain_classes_count=num_categories,
       trajectory_length=segment_size,
       latent_dim=num_categories // 2,
       hidden_size=128, 
       device=device
    ).to(device)


    # Load data
    dataset = TrajectoryDataset(
        data=costmap,
        num_samples=65,
        segment_length=segment_size-1,
        terrain_types=categories
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=True, collate_fn=collate)

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
    clusters = cluster_latent_space(model, dataloader, device, mapping=mapping_index_to_terrain)

    # Now we can use APREL with the clustered trajectory data. 
    
    # Creates a fake environment
    env = NonGymEnvironment(feature_func)

    # Converting data set to TrajectorySet
    trajectories = []
    for d in dataset:
        if d != None:
            trajectories.append(
                Trajectory(env, list(d), num_bins=num_categories, bin_labels=categories)
            )
        else:
           break

    trajectory_set = TrajectorySet(trajectories)

    # Creates a query optimizer
    query_optimizer = QueryOptimizerDiscreteTrajectorySet(trajectory_set)

    # Creates a human user manager
    true_user = HumanUser(delay=0.5)

    # Creates a weight vector with randomized parameters
    random_params = {"weights": get_random_normalized_vector(num_categories)}
    variational_params = {"weights": get_random_normalized_vector(num_categories)}

    # Creates a user model
    random_user_model = SoftmaxUser(random_params)
    variational_user_model = SoftmaxUser(variational_params)

    # Initializes the belief network using the user model and parameters
    # Comparing random sampling to variational autoencoder based sampling. 
    random_belief = SamplingBasedBelief(random_user_model, [], random_params)
    variational_belief = SamplingBasedBelief(variational_user_model, [], variational_params)

    # Creates an example query using the first two trajectories. 
    query = PreferenceQuery(trajectory_set[:2])

    # Learning Loop
    print("starting learning loop.......")

    # Dictionaries to hold the adjusted weights over time for each terrain type. 
    vweights = {
        k : [] for k in categories
    }

    rweights = {
        k : [] for k in categories
    }
    
    for query_no in range(5):
        vqueries, _ = query_optimizer.optimize('variational', variational_belief, query, clusters=clusters)
        rqueries, _ = query_optimizer.optimize('random', random_belief, query, clusters=clusters)
        
        print("Variational Query ->")
        vresponses = true_user.respond(vqueries[0])
        print("Random Query ->")
        rresonpses = true_user.respond(rqueries[0])

        variational_belief.update(Preference(vqueries[0], vresponses[0]))
        random_belief.update(Preference(rqueries[0], rresonpses[0]))

        print(f"Estimated weights for variational belief: {variational_belief.mean}")
        print(f"Estimated weights for random belief: {random_belief.mean}")

        for i, vw in enumerate(variational_belief.mean["weights"]):
            vweights[mapping_index_to_terrain[i]].append(vw)
        
        for j, rw in enumerate(random_belief.mean["weights"]):
            rweights[mapping_index_to_terrain[j]].append(rw)

    
    # Plot weights over iterations
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    for terrain_type in categories:
        avg_vweights = moving_average(vweights[terrain_type])
        axes[0].plot(avg_vweights, label=terrain_type)
        avg_rweights = moving_average(rweights[terrain_type])
        axes[1].plot(avg_rweights, label=terrain_type)
    
    axes[0].set_title("Average Terrain Weights Over Time for VAE Sampling")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Terrain Weight")
    axes[0].legend()

    axes[1].set_title("Average Terrain Weights Over Time for Random Sampling")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Terrain Weight")
    axes[1].legend()

    plt.tight_layout()

    plt.show()