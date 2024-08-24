from src.VAE.utils.imports import *
from src.VAE.utils.simulated_user import SimUser
from src.VAE.data_handlers.trajectory_segment_gatherer import TrajectoryDataset
from src.VAE.training_scripts.train_vae import TrainVAE
from src.VAE.architectures.vae_lstm import MemoryVAE
from src.VAE.utils.moving_average import moving_average
from src.VAE.vizualization_scripts.ClusteredTSNE import cluster_latent_space, visualize_latent_space
import copy

def collate(batch):
#    batch.sort(key=lambda x: len(x), reverse=True)
#    batch_trajectories = [item for item in batch]
#    lengths = [len(seq) for seq in batch_trajectories]
#    padded_trajectories = pad_sequence(batch_trajectories, batch_first=True)
#    packed_trajectories = pack_padded_sequence(padded_trajectories, lengths, batch_first=True, enforce_sorted=True)
#    return packed_trajectories
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

def coord_to_cell(x,y):
    id_x = (y/0.2)
    id_y = (x/0.2)
    return (int(id_x), int(id_y))

if __name__ == "__main__":
    # Create device
    device = torch.device("mps")
    print(f"Using {device}")

#    categories = ["Grass", "Road", "Sidewalk", "Water", "Trees", "Rock", "Brush", "Sand"]
    categories = ["Water", "Sand", "Rock", "Trees", "Sidewalk"]
    mapping_index_to_terrain = {
            i : t for i,t in enumerate(categories)
        }


#    grid_size = (5000, 5000)

#    p = np.ones(len(categories)) / len(categories)

 #   random_array = np.random.multinomial(grid_size, p, size=1)

#    random_array = np.random.randint(0, len(categories), size=grid_size)
#    costmap = np.vectorize(lambda x: categories[x])(random_array)

#    costmap = np.load("traversability_costmap.npy")
#    value_to_label = dict(zip(np.unique(costmap), categories[:len(np.unique(costmap))]))
#    costmap = np.vectorize(value_to_label.get)(costmap)
#    print(np.unique(costmap))

#    np.save("./semantic_map.npy", costmap)
    costmap = np.load("semantic_map.npy")
    print(costmap.shape)
    print(f"Sand: {100*np.count_nonzero(costmap == 'Sand')/costmap.size}")
    print(f"Water: {100*np.count_nonzero(costmap == 'Water')/costmap.size}")
    print(f"Rock: {100*np.count_nonzero(costmap == 'Rock')/costmap.size}")
    print(f"Trees: {100*np.count_nonzero(costmap == 'Trees')/costmap.size}")
    print(f"Sidewalk: {100*np.count_nonzero(costmap == 'Sidewalk')/costmap.size}")

    segment_size = 50
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

    #model = torch.load("../results/models/model7500.pt")


    # Load data
    dataset = TrajectoryDataset(
        data=costmap,
        num_samples=321,
        segment_length=segment_size-1,
        terrain_types=categories,
        start_point = (210, 200),#(random.randint(0, len(costmap)), random.randint(0, len(costmap[0]))),
        end_point = (255, 160), #(random.randint(0, len(costmap)), random.randint(0, len(costmap[0]))),
        device=device
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=True, collate_fn=collate)

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    
    # Train model
    trainer = TrainVAE(
        model=model,
        optimizer=optimizer,
        epochs=5000,
        batch_size=32,
        data=dataloader, 
        xdim=num_categories*segment_size, # Maybe trajectory size? 
        device=device
    )

    print("Created trainer, training....")
    trainer.train_model()

#    torch.save(model, "../results/models/model40010000gru.pt")

    dataset = TrajectoryDataset(
        data=costmap,
        num_samples=50,
        segment_length=segment_size-1,
        terrain_types=categories,
        start_point = (210,200),
        end_point = (255, 160),
        device=device
    )


    dataset2 = copy.deepcopy(dataset)

    dataloader = DataLoader(dataset, batch_size=1, drop_last=False, collate_fn=collate)

    # Cluster the latent space representations of the original dataset. 
    clusters = cluster_latent_space(model, dataloader, device, mapping=mapping_index_to_terrain)

    # Check if any clusters are less represented. 
    cluster_counts = [np.count_nonzero(clusters == t) for t in range(len(categories))]
    # Maybe we could plot a distribution for this later
    print(cluster_counts)
    # Thresholds for cluster size and acceptable imbalance ratio
    max_cluster_size = 1.25*len(dataset2)/(num_categories) # Fully equal with 25% imbalance
    threshold = 0.15

    max_size = max(cluster_counts)
    min_size = min(cluster_counts)
    min_cluster = np.argmin(cluster_counts)

    print(min_size)
    print(max_size)
    print(min_cluster)

    print("Optimizing Trajectory Set.......")
    print("Original length: ")
    print(len(dataset))

    while (max_size - min_size)/min_size > threshold:
        # Obtain a new trajectory using decoder
        # - Determine which cluster belongs to the minimum
        # - Put a trajectory of all that terrain type into the model
        # - Get a decoded output to add to that cluster. 
        sample_traj = torch.tensor([min_cluster]*segment_size, dtype=torch.float, device=device)
        new_trajectory = model(torch.stack([sample_traj]))[0].round().long()
        dataset2.trajectories = torch.cat((dataset2.trajectories, new_trajectory))
        print(len(dataset2))
        clusters = np.append(clusters, min_cluster)
        cluster_counts = [np.count_nonzero(clusters == t) for t in range(len(categories))]
        max_size = max(cluster_counts)
        min_size = min(cluster_counts)
        min_cluster = np.argmin(cluster_counts)
        print((max_size - min_size)/min_size)

    num_labels = len(np.unique(clusters))



    # I think re-clustering after doing the balancing will be interesting. 
    # I'm curious to see if the same trajectories are clustered into the right
    # places and if the overall distribution would improve. 
    dataloader2 = DataLoader(dataset2, batch_size=1, drop_last=False, collate_fn=collate)
#    new_clusters = cluster_latent_space(
#        model, dataloader, device, mapping=mapping_index_to_terrain
#    )
    print(len(dataloader2))
    print(len(clusters))
    visualize_latent_space(model, dataloader2, device, mapping=mapping_index_to_terrain, labels=clusters)


    # Now we can use APREL with the clustered trajectory data. 

    # For demonstration, we will use a simulated user that selects the trajectory
    # most aligned with the provided reward function. 
#    rewards = [1., -0.8, 0.5, -1., -0.2, 0.0, 0.2, -0.5] # We can play more with these weights later.
    rewards = [-1, -0.5, 0.0, 0.5, 1.0]

    simuser = SimUser(rewards=rewards, labels=categories)
    
    # Creates a fake environment
    env = NonGymEnvironment(feature_func)

    print(len(dataset))

    # Converting data set to TrajectorySet
    trajectories = []
    for d in dataset:
        if d != None:
            trajectories.append(
                Trajectory(env, list(d.cpu().detach().numpy()), num_bins=num_categories, bin_labels=categories)
            )
        else:
            break
    
    trajectories2 = []
    for d in dataset2:
        if d != None:
            trajectories2.append(
                Trajectory(env, list(d.cpu().detach().numpy()), num_bins=num_categories, bin_labels=categories)
            )
        else:
            break

    print("Trajectories")
    print(len(trajectories))
    print(len(trajectories2))

    trajectory_set = TrajectorySet(trajectories)
    trajectory_set2 = TrajectorySet(trajectories2)

    # Creates a query optimizer
    rquery_optimizer = QueryOptimizerDiscreteTrajectorySet(trajectory_set)
    vquery_optimizer = QueryOptimizerDiscreteTrajectorySet(trajectory_set2)

    # Creates a human user manager
#    true_user = HumanUser(delay=0.5)

    # Creates a weight vector with randomized parameters
#    initial_weights = get_random_normalized_vector(num_categories)
    rinitial_weights = [0.0]*len(rewards)
    random_params = {"weights": rinitial_weights}
    vinitial_weights = [0.0]*len(rewards)
    variational_params = {"weights": vinitial_weights}

    # Creates a user model
    random_user_model = SoftmaxUser(random_params)
    variational_user_model = SoftmaxUser(variational_params)

    # Initializes the belief network using the user model and parameters
    # Comparing random sampling to variational autoencoder based sampling. 
    random_belief = SamplingBasedBelief(random_user_model, [], random_params)
    print(f"MaxEnt Weights: {random_belief.mean['weights']}")
    variational_belief = SamplingBasedBelief(variational_user_model, [], variational_params)
    print(f"Variational Weights: {variational_belief.mean['weights']}")

    # Creates an example query using the first two trajectories. 
    query = PreferenceQuery(trajectory_set[:2])

    # Learning Loop
#    print("starting learning loop.......")

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

    time_differences = {
        "Random": [],
        "Variational": []
    }

    query_no = 0
    max_iterations = 25
    
    print("Learning with Max Entropy Query Selection")
    random_convergence_time = []
    rbm = random_belief.mean["weights"]
    rconvergence = sum([abs(rbm[k] - rewards[k]) for k in range(len(rewards))])/(len(rewards)*2)
    random_convergence_time.append(rconvergence)
    while query_no <= max_iterations:
    #for query_no in range(25):
        query_no += 1
        print(f"RUNNING QUERY {query_no}")
        rqueries, _ = rquery_optimizer.optimize('mutual_information', random_belief, query, clusters=clusters)
        avg_var_r = 0

        for x in range(len(rewards)):
            avg_var_r += abs(rqueries[0].slate[0].features[0][x] - rqueries[0].slate[1].features[0][x])
        
        avg_var_r = avg_var_r / len(rewards)
        #avg_var_v = avg_var_v / len(rewards)
        ralignment.append(
            simuser.check_distribution(get_preferential_order(random_belief.mean["weights"], categories))
        )

        rresonpses = simuser.respond(rqueries[0])
        random_belief.update(Preference(rqueries[0], rresonpses[0]))
        for j, rw in enumerate(random_belief.mean["weights"]):
            rweights[mapping_index_to_terrain[j]].append(rw)
        
        rbm = random_belief.mean["weights"]
        rconvergence = sum([abs(rbm[k] - rewards[k]) for k in range(len(rewards))])/(len(rewards)*2)
        random_convergence_time.append(rconvergence)

        


    print("Learning with VAE Query Selection")
    query_no = 0
    variational_convergence_time = []
    vbm = variational_belief.mean["weights"]
    vconvergence = sum([abs(vbm[k] - rewards[k]) for k in range(len(rewards))])/(len(rewards)*2)
    variational_convergence_time.append(vconvergence)
    while query_no <= max_iterations:
#    for query_no in range(25):
        query_no += 1
        print(f"RUNNING QUERY {query_no}")
        # Time VAE method
#        vstart_time = time.time()
        vqueries, _ = vquery_optimizer.optimize('variational_info', variational_belief, query, clusters=clusters)
#        time_differences["Variational"].append(time.time() - vstart_time)
#        rstart_time = time.time()
        #rqueries, _ = query_optimizer.optimize('mutual_information', random_belief, query, clusters=clusters)
#        time_differences["Random"].append(time.time() - rstart_time)
#        print(vqueries[0].slate[0].features)

        avg_var_v = 0
#        avg_var_r = 0

        for x in range(len(rewards)):
            avg_var_v += abs(vqueries[0].slate[0].features[0][x] - vqueries[0].slate[1].features[0][x])
#            avg_var_r += abs(rqueries[0].slate[0].features[0][x] - rqueries[0].slate[1].features[0][x])
        
#        avg_var_r = avg_var_r / len(rewards)
        avg_var_v = avg_var_v / len(rewards)
#        vvariance.append(avg_var_v)
#        rvariance.append(avg_var_r)
#        variance_difference.append((avg_var_v - avg_var_r) / (avg_var_v + avg_var_r)) # Percent Difference

        valignment.append(
            simuser.check_distribution(get_preferential_order(variational_belief.mean["weights"], categories))
        )

#        ralignment.append(
#            simuser.check_distribution(get_preferential_order(random_belief.mean["weights"], categories))
#        )


#        print("Variational Query ->")
        vresponses = simuser.respond(vqueries[0])
        print(vresponses)
#        vresponses = true_user.respond(vqueries[0])
#        print("Random Query ->")
#        rresonpses = simuser.respond(rqueries[0])
#        rresonpses = true_user.respond(rqueries[0])

        # if query_no in [5, 10, 15]:
        #     print("Example histograms:")
        #     print("Showing VariQuery")
        #     vqueries[0].visualize()
        #     print("Showing Random")
        #     rqueries[0].visualize()

#        print("Getting responses....")

        variational_belief.update(Preference(vqueries[0], vresponses[0]))
        #random_belief.update(Preference(rqueries[0], rresonpses[0]))

#        print(f"Estimated weights for variational belief: {variational_belief.mean}")
#        print(f"Estimated weights for random belief: {random_belief.mean}")

        
        for i, vw in enumerate(variational_belief.mean["weights"]):
            vweights[mapping_index_to_terrain[i]].append(vw)
        
#        for j, rw in enumerate(random_belief.mean["weights"]):
#            rweights[mapping_index_to_terrain[j]].append(rw)
        vbm = variational_belief.mean["weights"]
        vconvergence = sum([abs(vbm[k] - rewards[k]) for k in range(len(rewards))])/(len(rewards)*2)
        variational_convergence_time.append(vconvergence)



    
    # Plot weights over iterations
    fig, axes = plt.subplots(2, 2, figsize=(12, 6))

    for terrain_type in categories:
        avg_vweights = moving_average(vweights[terrain_type])
        axes[0][0].plot(avg_vweights, label=terrain_type)
        avg_rweights = moving_average(rweights[terrain_type])
        axes[0][1].plot(avg_rweights, label=terrain_type)

    axes[0][0].set_title("Average Terrain Weights Over Time for VAE Sampling")
    axes[0][0].set_xlabel("Query Number")
    axes[0][0].set_ylabel("Terrain Weight")
    axes[0][0].legend()

    axes[0][1].set_title("Average Terrain Weights Over Time for Mutual Information")
    axes[0][1].set_xlabel("Query Number")
    axes[0][1].set_ylabel("Terrain Weight")
    axes[0][1].legend()


    axes[1][0].plot(ralignment, label="Mutual Information Queries", color="#5c87ff", linewidth=2)
    axes[1][0].plot(valignment, label="VAE-based Queries", color="#f75cff", linewidth=2)
    axes[1][0].set_title("Reward Alignment Over Time")
    axes[1][0].set_xlabel("Query Number")
    axes[1][0].set_ylabel("Reward Alignment")
    axes[1][0].legend()


    axes[1][1].plot(variational_convergence_time, label="Convergence with VAE-based Queries", color="#f75cff", linewidth=2)
    axes[1][1].plot(random_convergence_time, label="Convergence with Mutual Information Queries", color="#5c87ff", linewidth=2)
    axes[1][1].set_title("Convergence to Ground Truth Over Time")
    axes[1][1].set_xlabel("Query Number")
    axes[1][1].set_ylabel("MAE Loss")
    axes[1][1].legend()

    #axes[1][1].plot(variance_difference, color="#000", linewidth=2)
    #axes[1][1].set_title("Normalized Change in Variance Over Time")
    #axes[1][1].set_xlabel("Iteration")
    #axes[1][1].set_ylabel("Normalized Difference in Variance")

    #axes[2][0].plot(time_differences["Random"], label="Mutual Information", color="#5c87ff", linewidth=2)
    #axes[2][0].plot(time_differences["Variational"], label="Variational", color="#f75cff", linewidth=2)
    #axes[2][0].set_title("Query Time per Iteration")
    #axes[2][0].set_ylabel("Query Time")
    #axes[2][0].set_xlabel("Iteration")
    #axes[2][0].legend()

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
    print(f"The MAE loss at the end for VAE Query Selection was: {variational_convergence_time[-1]}")
    print(f"The MAE loss at the end for Maximum Entropy Query Selection was {random_convergence_time[-1]}")