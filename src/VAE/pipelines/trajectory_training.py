from src.VAE.utils.imports import *
from src.VAE.utils.simulated_user import SimUser
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
    categories = ["Water", "Sand", "Rock", "Trees", "Sidewalk"]
#    categories = ["Grass", "Road", "Sidewalk", "Water", "Trees", "Rock", "Brush", "Sand"]
    mapping = {
            t : i for i,t in enumerate(categories)
        }
    
    traj = np.array(traj, dtype=int)

    for t in categories:
        count = np.count_nonzero(traj == mapping[t])
        features.append(count/len(traj))
    
    # For the simulated user we want to get the preferential ordering of the categories
    paired = list(zip(features, categories))

    # Sort the pairs based on the features (weights)
    paired_sorted = sorted(paired, key=lambda x: x[0], reverse=True)

    # Extract the sorted categories (labels)
    sorted_categories = [category for _, category in paired_sorted]
#    print(f"Features: {features}")
#    print(f"Sorted Categories: {sorted_categories}")
    
    return np.array(features), sorted_categories

def get_preferential_order(weights, semantic_categories):
    # For the simulated user we want to get the preferential ordering of the categories
    paired = list(zip(weights, semantic_categories))

    # Sort the pairs based on the features (weights)
    paired_sorted = sorted(paired, key=lambda x: x[0], reverse=True)

    # Extract the sorted categories (labels)
    sorted_categories = [category for _, category in paired_sorted]

    return sorted_categories

if __name__ == "__main__":
    # Create device
    device = torch.device("mps")
    print(f"Using {device}")

#    categories = ["Grass", "Road", "Sidewalk", "Water", "Trees", "Rock", "Brush", "Sand"]
    categories = ["Water", "Sand", "Rock", "Trees", "Sidewalk"]
    mapping_index_to_terrain = {
            i : t for i,t in enumerate(categories)
        }


    grid_size = (5000, 5000)

    p = np.ones(len(categories)) / len(categories)

 #   random_array = np.random.multinomial(grid_size, p, size=1)

    random_array = np.random.randint(0, len(categories), size=grid_size)
    costmap = np.vectorize(lambda x: categories[x])(random_array)

#    costmap = np.load("traversability_costmap.npy")
#    value_to_label = dict(zip(np.unique(costmap), categories[:len(np.unique(costmap))]))
#    costmap = np.vectorize(value_to_label.get)(costmap)
#    print(np.unique(costmap))

#    np.save("./semantic_map.npy", costmap)
#    costmap = np.load("semantic_map.npy")
#    print(costmap.shape)
#    print(f"Sand: {np.count_nonzero(costmap == 'Sand')/costmap.size}")
#    print(f"Water: {np.count_nonzero(costmap == 'Water')/costmap.size}")
#    print(f"Rock: {np.count_nonzero(costmap == 'Rock')/costmap.size}")
#    print(f"Trees: {np.count_nonzero(costmap == 'Trees')/costmap.size}")
#    print(f"Sidewalk: {np.count_nonzero(costmap == 'Sidewalk')/costmap.size}")

    segment_size = 64
    num_categories = len(categories)
    batch_size = 32

    # Load LSTM Model
    model = MemoryVAE(
       batch_size=batch_size,
       terrain_classes_count=num_categories,
       trajectory_length=segment_size,
       latent_dim=num_categories ,
       hidden_size=512, 
       device=device
    ).to(device)


    # Load data
    dataset = TrajectoryDataset(
        data=costmap,
        num_samples=321,
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
        epochs=500,
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

    # For demonstration, we will use a simulated user that selects the trajectory
    # most aligned with the provided reward function. 
#    rewards = [1., -0.8, 0.5, -1., -0.2, 0.0, 0.2, -0.5] # We can play more with these weights later.
    rewards = [-1, -0.5, 0.0, 0.5, 1.0]

    simuser = SimUser(rewards=rewards, labels=categories)
    
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
#    true_user = HumanUser(delay=0.5)

    # Creates a weight vector with randomized parameters
    initial_weights = get_random_normalized_vector(num_categories)
    random_params = {"weights": initial_weights}
    variational_params = {"weights": initial_weights}

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

    rvariance = []
    vvariance = []
    variance_difference = []

    valignment = []
    ralignment = []
#    vconvergence = []
#    rconvergence = []
    
    for query_no in range(10):
        print(f"RUNNING QUERY {query_no}")
        vqueries, _ = query_optimizer.optimize('variational_info', variational_belief, query, clusters=clusters)
        rqueries, _ = query_optimizer.optimize('random', random_belief, query, clusters=clusters)
#        print(vqueries[0].slate[0].features)

        avg_var_v = 0
        avg_var_r = 0

        for x in range(len(rewards)):
            avg_var_v += abs(vqueries[0].slate[0].features[0][x] - vqueries[0].slate[1].features[0][x])
            avg_var_r += abs(rqueries[0].slate[0].features[0][x] - rqueries[0].slate[1].features[0][x])
        
        avg_var_r = avg_var_r / len(rewards)
        avg_var_v = avg_var_v / len(rewards)
#        vvariance.append(avg_var_v)
#        rvariance.append(avg_var_r)
        variance_difference.append((avg_var_v - avg_var_r) / (avg_var_v + avg_var_r)) # Percent Difference

        valignment.append(
            simuser.check_distribution(get_preferential_order(variational_belief.mean["weights"], categories))
        )

        ralignment.append(
            simuser.check_distribution(get_preferential_order(random_belief.mean["weights"], categories))
        )


#        print("Variational Query ->")
        vresponses = simuser.respond(vqueries[0])
        print(vresponses)
#        vresponses = true_user.respond(vqueries[0])
#        print("Random Query ->")
        rresonpses = simuser.respond(rqueries[0])
#        rresonpses = true_user.respond(rqueries[0])

        # if query_no in [5, 10, 15]:
        #     print("Example histograms:")
        #     print("Showing VariQuery")
        #     vqueries[0].visualize()
        #     print("Showing Random")
        #     rqueries[0].visualize()

#        print("Getting responses....")

        variational_belief.update(Preference(vqueries[0], vresponses[0]))
        random_belief.update(Preference(rqueries[0], rresonpses[0]))

#        print(f"Estimated weights for variational belief: {variational_belief.mean}")
#        print(f"Estimated weights for random belief: {random_belief.mean}")

        
        for i, vw in enumerate(variational_belief.mean["weights"]):
            vweights[mapping_index_to_terrain[i]].append(vw)
        
        for j, rw in enumerate(random_belief.mean["weights"]):
            rweights[mapping_index_to_terrain[j]].append(rw)
        


    
    # Plot weights over iterations
    fig, axes = plt.subplots(2, 2, figsize=(12, 6))

    for terrain_type in categories:
        avg_vweights = moving_average(vweights[terrain_type])
        axes[0][0].plot(avg_vweights, label=terrain_type)
        avg_rweights = moving_average(rweights[terrain_type])
        axes[0][1].plot(avg_rweights, label=terrain_type)

    axes[0][0].set_title("Average Terrain Weights Over Time for VAE Sampling")
    axes[0][0].set_xlabel("Iteration")
    axes[0][0].set_ylabel("Terrain Weight")
    axes[0][0].legend()

    axes[0][1].set_title("Average Terrain Weights Over Time for Random Sampling")
    axes[0][1].set_xlabel("Iteration")
    axes[0][1].set_ylabel("Terrain Weight")
    axes[0][1].legend()


    axes[1][0].plot(ralignment, label="Random Queries", color="#5c87ff", linewidth=2)
    axes[1][0].plot(valignment, label="VAE-based Queries", color="#f75cff", linewidth=2)
    axes[1][0].set_title("Reward Alignment Over Time")
    axes[1][0].set_xlabel("Iteration")
    axes[1][0].set_ylabel("Reward Alignment")
    axes[1][0].legend()

    axes[1][1].plot(variance_difference, color="#000", linewidth=2)
    axes[1][1].set_title("Normalized Change in Variance Over Time")
    axes[1][1].set_xlabel("Iteration")
    axes[1][1].set_ylabel("Normalized Difference in Variance")

    plt.tight_layout()

    plt.show()

    print("Summary -------------------------------")

    print(f"The reward weights were: {rewards}")

    vbm = variational_belief.mean["weights"]
    rbm = random_belief.mean["weights"]

    print(f"The final weights for VAE Query Selection were: {vbm}")
    print(f"The final weights for Random Query Selection were: {rbm}")
    print(f"The difference from ground truth from the VAE Query Selection was: {[abs(vbm[k] - rewards[k]) for k in range(len(rewards))]}")
    print(f"The difference from ground truth from the Random Query Selection was: {[abs(rbm[k] - rewards[k]) for k in range(len(rewards))]}")