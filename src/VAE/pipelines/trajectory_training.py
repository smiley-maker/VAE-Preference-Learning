from src.VAE.utils.imports import *
from src.VAE.utils.simulated_user import SimUser
from src.VAE.data_handlers.trajectory_segment_gatherer import TrajectoryDataset
from src.VAE.training_scripts.train_vae import TrainVAE
from src.VAE.architectures.vae_lstm import MemoryVAE
from src.VAE.vizualization_scripts.ClusteredTSNE import cluster_latent_space, visualize_latent_space
from src.VAE.pipelines.preference_learning import TerrainLearning
from src.VAE.vizualization_scripts.costmap_query_viz import visualize_double_costmap_query

class TerrainLearningPipeline:
    def __init__(self, 
                 terrain_map : np.array, 
                 terrain_types : list[str], 
                 ground_truth : list[float],
                 num_epochs : int,
                 trajectory_length : int,
                 batch_size : int,
                 start_loc : tuple[int, int],
                 end_loc : tuple[int, int],
                 num_training_samples : int,
                 num_initial_dataset : int,
                 cluster_balance_threshold : float,
                 use_planner : bool,
                 data_folder : str
                ) -> None:
        
        self.map = terrain_map
        self.terrain_types = terrain_types
        self.ground_truth = ground_truth
        self.num_epochs = num_epochs
        self.trajectory_length = trajectory_length
        self.batch_size = batch_size
        self.start_loc = start_loc
        self.end_loc = end_loc
        self.num_training = num_training_samples
        self.num_initial = num_initial_dataset
        self.threshold = cluster_balance_threshold
        self.use_planner = use_planner
        self.data_folder = data_folder

        # Constructs a pairwise relationship beetween rewards and terrain types.
        self.terrain_rewards_ground_truth = {
            t : r for t,r in zip(self.terrain_types, self.ground_truth)
        }

        # We can still get the preferential ordering using the simulated
        # user's function: 
        self.simuser = SimUser(
            list(self.terrain_rewards_ground_truth.values()), 
            list(self.terrain_rewards_ground_truth.keys())
        )

        self.reward_ordering = self.simuser.get_preferential_order(self.terrain_rewards_ground_truth)

        print(f"Ground Truth Reward Ordering: {self.reward_ordering}")

        # We need a mapping from terrain labels to indices as well. 
        self.terrain_to_index = {
            t : i for i, t in enumerate(self.terrain_types)
        }

        # And one going the other direction
        self.index_to_terrain = {
            i : t for i, t in enumerate(self.terrain_types)
        }

        # Device for tensors
        self.device = torch.device("mps")
        print(f"Using {self.device}")



    def collate(self, batch):
        """Collate function used to create a stacked batch for dataloader. 

        Args:
            batch (list[Tensor]): batch of n trajectories. 

        Returns:
            Tensor: Returns a stacked version of the batch compatible with PyTorch. 
        """
        return torch.stack(batch)


    def feature_func(self, traj : Trajectory) -> tuple[np.array, list[str]]:
        """
        Returns the terrain feature breakdown for a given trajectory. 

        Args:
            traj (Trajectory): Trajectory containing encoded terrains.

        Returns:
            tuple[np.array, list[str]]: The feature breakdown and the 
                                        categories sorted by frequency. 
        """

        features = {}

        # Convert the trajectory to a numpy array
        traj = np.array(traj, dtype=int)
        n = len(traj)

        # Count how many times each terrain type appears in the trajectory
        for t in self.terrain_types:
            count = np.count_nonzero(traj == self.terrain_to_index[t])
            features[t] = (count / n)
        
        # Get the preferential ordering to be used in preference elicitation. 
        sorted_terrains = self.simuser.get_preferential_order(features)

        # Return the reward weights and terrain importances. 
        return np.array(list(features.values())), sorted_terrains
    
    
    def find_max_terrain_region(self, terrain_type, region_width, region_height):
        """
        Finds the region in the costmap with the most occurrences of the specified terrain type.

        Args:
            terrain_type: The terrain type to search for.
            region_width: The width of the region to consider.
            region_height: The height of the region to consider.

        Returns:
            A tuple containing:
                - The starting row index of the region.
                - The starting column index of the region.
        """

        rows, cols = self.map.shape
        max_count = 0
        best_start = (0, 0)
        best_region = None

        for i in range(0, rows - region_height + 1, region_height//2):
            for j in range(0, cols - region_width + 1, region_width//2):
                region = self.map[j:j+region_width, i:i+region_height]
                count = np.count_nonzero(region == terrain_type)
                if count > max_count:
                    best_region = region
                    max_count = count
                    print(f"Updated max count for {terrain_type} to {max_count}")
                    best_start = (i, j)

        return best_region, best_start


    def translate_points_to_new_region(self, region_start_x_new, region_start_y_new):
        """
        Translates start and goal points from one region to another.

        Args:
            region_start_x_new: X-coordinate of the starting point of the new region.
            region_start_y_new: Y-coordinate of the starting point of the new region.

        Returns:
            A tuple containing the translated start and goal points.
        """

        offset_x = region_start_x_new - self.modified_dataset.min_x
        offset_y = region_start_y_new - self.modified_dataset.min_y

        if abs(offset_x) > len(self.map[0]) - region_start_x_new:
            offset_x = int((len(self.map[0])-region_start_x_new)*math.copysign(1, offset_x))
        if abs(offset_y) > len(self.map[1]) - region_start_y_new:
            offset_y = int((len(self.map[1])-region_start_x_new)*math.copysign(1, offset_y))


        print(f"Offset X: {offset_x}")
        print(f"Offset Y: {offset_y}")

        start_x_new = self.start_loc[0] + offset_x
        start_y_new = self.start_loc[1] + offset_y
        goal_x_new = self.end_loc[0] + offset_x
        goal_y_new = self.end_loc[1] + offset_y

        print(f"Start: ({start_x_new}, {start_y_new})")
        print(f"Goal: ({goal_x_new}, {goal_y_new})")

        return max(start_x_new, 0), max(start_y_new, 0), max(goal_x_new, 0), max(goal_y_new, 0)

    
    def find_best_locations(self):
        """
        We know the area size that we need for path planning. So, we can 
        search the costmap with overlapping grids to find the start and 
        end goals that give the most of each terrain type. 
        """

        w = abs(self.start_loc[0] - self.end_loc[0]) + 400
        h = abs(self.start_loc[1] - self.end_loc[1]) + 400

        print(f"Width: {w}, Height: {h}")

        self.best_locs = {
            t : self.find_max_terrain_region(t, w, h) for t in self.terrain_types
        }

        for t in self.terrain_types:
            try:
                assert self.best_locs[t][0][0] < self.trajectory_length*2
                assert self.best_locs[t][0][1] < self.trajectory_length*2
            except:
                print(f"Map for {t} is too large, with shape {self.best_locs[t][0].shape}")
        
        print(
            [(np.count_nonzero(self.best_locs[t][0] == t))/self.best_locs[t][0].size for t in self.terrain_types]
        )
        print(self.best_locs.keys())

    def train_model(self):
        self.model = MemoryVAE(
            batch_size=self.batch_size,
            terrain_classes_count=len(self.terrain_types),
            trajectory_length=self.trajectory_length,
            latent_dim=len(self.terrain_types),
            hidden_size=512,
            device=self.device
        ).to(self.device)

        # Load data
        training_dataset = TrajectoryDataset(
            data=self.map,
            num_samples=self.num_training,
            segment_length=self.trajectory_length - 1,
            terrain_types=self.terrain_types,
            device=self.device
        )

        dataloader = DataLoader(training_dataset, batch_size=self.batch_size, drop_last=True, collate_fn=self.collate)

        # Create optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=3e-4)
        
        # Train model
        trainer = TrainVAE(
            model=self.model,
            optimizer=optimizer,
            epochs=self.num_epochs,
            batch_size=self.batch_size,
            data=dataloader, 
            xdim=len(self.terrain_types)*self.trajectory_length, 
            device=self.device
        )

        print("Created trainer, training....")
        trainer.train_model()
    
    def optimization_step(self):
        # Create initial dataset for preference learning
        self.initial_dataset = TrajectoryDataset(
            data=self.map,
            num_samples=self.num_initial,
            segment_length=self.trajectory_length-1,
            terrain_types=self.terrain_types,
            device=self.device,
            start_point=self.start_loc,
            end_point=self.end_loc
        )

        dataloader = DataLoader(self.initial_dataset, batch_size=1, drop_last=False, collate_fn=self.collate)

        # Create a copy of the initial dataset to improve on
        self.modified_dataset = copy.deepcopy(self.initial_dataset)

        # TODO: Find the number of clusters based on the unique terrain types
        # in the initial trajectory set. 
        # We could concatenate all trajectory terrain lists into one large
        # list and then use np.unique()
        # Or we could use a set and loop through the unique terrains in each
        # trajectory. 
#        unique_terrains = set()
#        for traj in self.initial_dataset.trajectories:
#            for t in np.unique(traj.cpu().numpy()):
#                unique_terrains.add(t)

        # Cluster the latent space representations of the original dataset. 
        self.clusters = cluster_latent_space(
            self.model, 
            dataloader, 
            self.device, 
            mapping=self.index_to_terrain,
 #           unique_terrains=len(unique_terrains)
        )

        # Clusters need to be re-mapped to ensure they represent terrain types. 
        # Each index in self.clusters corresponds to a trajectory in the initial dataset. 
        # Find all indices corresponding to each unique label and average the trajectory
        # feature breakdowns to determine which terrain type the label represents. 

        options = copy.deepcopy(self.terrain_types)

        cluster_mapping = {}

        print(self.clusters)

        for idx in np.unique(self.clusters):
            # Gets all trajectory indices for cluster idx
            entries = [xv for xv in self.clusters if xv == idx]
            print(entries)
            terrain_dict = {
                t : 0 for t in self.terrain_types
            }

            for jdx in list(entries):
                _, sorted_terrains = self.feature_func(self.initial_dataset[jdx].cpu())
                print(sorted_terrains)
                # most prominent is last entry of sorted_terrains
                terrain_dict[sorted_terrains[-1]] += 1
            print(terrain_dict)
            
            choice = max(terrain_dict, key=terrain_dict.get)
            print(choice)
            while choice not in options and len(options) > 0:
                terrain_dict[choice] = -1
                choice = max(terrain_dict, key=terrain_dict.get)
                print(choice)
                print(terrain_dict)

            
            cluster_mapping[idx] = self.terrain_to_index[choice]
            options.remove(choice)
        
        print(cluster_mapping)

        # Map all options in self.clusters for simplicity
        self.clusters = np.array([cluster_mapping[c] for c in self.clusters])
            
        visualize_latent_space(self.model, dataloader, self.device, mapping=self.index_to_terrain, labels=self.clusters)

        # Check if any clusters are less represented. 
        print(self.clusters)
        terrain_indices = [self.terrain_to_index[x] for x in self.terrain_types]
        print(terrain_indices)
        cluster_counts = [np.count_nonzero(self.clusters == t) for t in terrain_indices]
        # Maybe we could plot a distribution for this later
        print(cluster_counts)
        # Thresholds for cluster size and acceptable imbalance ratio
        max_size = max(cluster_counts)
        min_size = min(cluster_counts)
        min_cluster = np.argmin(cluster_counts)
        max_cluster = np.argmax(cluster_counts)



        while (max_size - min_size)/max_size > self.threshold:
            print(f"Min Cluster: {min_cluster}, or {self.index_to_terrain[min_cluster]}")
            print(f"Max Cluster: {max_cluster}, or {self.index_to_terrain[max_cluster]}")
            new_weights = random.choices(range(10, 91), k=len(self.terrain_types)-2)
            random.shuffle(new_weights)
            new_weights.insert(min_cluster, 100)
            new_weights.insert(max_cluster, 0)
            print("New Weights:")
            print(new_weights)
            mapping = {
                t : w for t,w in zip(self.terrain_types, new_weights)
            }

            self.modified_dataset.smaller_costmap = self.best_locs[self.index_to_terrain[min_cluster]][0]
            print(np.unique(self.modified_dataset.smaller_costmap))

            new_traj = torch.tensor(self.modified_dataset.integer_label_encoding(self.modified_dataset.plan_path(new_weights, mapping)[0]), device=self.device)

            if len(new_traj) <= self.trajectory_length:
                print("Expanding path")
                new_traj = torch.cat((new_traj, new_traj[:self.trajectory_length - len(new_traj)]))
                
            else:
                print("Increasing Path")
                new_traj = new_traj[:self.trajectory_length]
            
            print(f"Adding trajectory of size {len(new_traj)} to {self.terrain_types[min_cluster]}")
            print(f"Trajectory Breakdown: {self.feature_func(new_traj.cpu())}")

            self.modified_dataset.trajectories = torch.cat((self.modified_dataset.trajectories, new_traj.unsqueeze(0)))
            self.clusters = np.append(self.clusters, min_cluster)
            cluster_counts = [np.count_nonzero(self.clusters == t) for t in [self.terrain_to_index[x] for x in self.terrain_types]]
            print(cluster_counts)
            max_size = max(cluster_counts)
            min_size = min(cluster_counts)
            min_cluster = np.argmin(cluster_counts)
            max_cluster = np.argmax(cluster_counts)
            print((max_size - min_size)/max_size)


        # TODO: Add functionality for decoded trajectories based on self.use_planner option.
        # while (max_size - min_size)/min_size > self.threshold:
        #         # Obtain a new trajectory using decoder
        #         # - Determine which cluster belongs to the minimum
        #         # - Put a trajectory of all that terrain type into the model
        #         # - Get a decoded output to add to that cluster. 
        #     sample_traj = torch.tensor([min_cluster]*self.trajectory_length, dtype=torch.float, device=self.device)
        #     new_trajectory = self.model(torch.stack([sample_traj]))[0].round().long()
        #     self.modified_dataset.trajectories = torch.cat((self.modified_dataset.trajectories, new_trajectory))
        #     print(len(self.modified_dataset))
        #     self.clusters = np.append(self.clusters, min_cluster)
        #     cluster_counts = [np.count_nonzero(self.clusters == t) for t in range(len(self.terrain_types))]
        #     max_size = max(cluster_counts)
        #     min_size = min(cluster_counts)
        #     min_cluster = np.argmin(cluster_counts)
        #     print((max_size - min_size)/min_size)
        
        dataloader2 = DataLoader(self.modified_dataset, batch_size=1, drop_last=False, collate_fn=self.collate)
        visualize_latent_space(self.model, dataloader2, self.device, mapping=self.index_to_terrain, labels=self.clusters)
    
    def run_preference_learning(self, learning_type : str, max_queries : int, threshold : float):
        """Runs preference learning with different parameters. There are 3 different learning
           strategies available:
            => fixed, which runs for a certain number of queries
            => alignment, which runs until the estimated ground truth order is aligned
                with the ground truth order
            => convergence, which runs until the error between the estimated weights and
                the ground truth is within some threshold. 

        Args:
            learning_type (str): Determines wether to run convergence, alignment, or fixed preference learning
            max_queries (int): Maximum allowed number of queries
            threshold (float): Determines when convergence tests are allowed to stop. 
        """
        
        # Creates a fake environment
        env = NonGymEnvironment(self.feature_func)

        # Converting data sets to TrajectorySet
        trajectories = []
        for d in self.initial_dataset:
            if d != None:
                trajectories.append(
                    Trajectory(env, list(d.cpu().detach().numpy()), num_bins=len(self.terrain_types), bin_labels=self.terrain_types)
                )
            else:
                break
        
        trajectories2 = []
        for d in self.modified_dataset:
            if d != None:
                trajectories2.append(
                    Trajectory(env, list(d.cpu().detach().numpy()), num_bins=len(self.terrain_types), bin_labels=self.terrain_types)
                )
            else:
                break

        print("Trajectories")
        print(len(trajectories))
        print(len(trajectories2))

        trajectory_set = TrajectorySet(trajectories)
        trajectory_set2 = TrajectorySet(trajectories2)


        pl_manager = TerrainLearning(
            datasets=[trajectory_set, trajectory_set2, trajectory_set2, trajectory_set2, trajectory_set2, trajectory_set2],
            simuser = self.simuser,
            ground_truth=self.ground_truth,
            algorithms=["mutual_information", "variational_info", "volume_removal", "thompson", "disagreement"],
            terrain_labels=self.terrain_types,
            initial_weights=random.sample(list(np.linspace(-1, 1, 201)), len(self.terrain_types)),
            no_queries=max_queries,
            plot_error=True,
            plot_conv=True,
            use_sim=True,
            clusters=self.clusters
        )

        print(self.ground_truth)

        error_save = self.data_folder + f"/convergence/{len(self.initial_dataset)}_initial_trajectories_{len(self.modified_dataset)}_after_{self.trajectory_length}_length_{self.num_epochs}_epochs_error.png"
        if learning_type == "convergence":
            x = pl_manager.run_learning_convergence(threshold)
            figure_save = self.data_folder + f"/convergence/{len(self.initial_dataset)}_initial_trajectories_{len(self.modified_dataset)}_after_{self.trajectory_length}_length_{self.num_epochs}_epochs.png"
            pl_manager.plot_together(figure_save, error_save)
            return x
        elif learning_type == "alignment":
            x = pl_manager.run_learning_alignment()
            figure_save = self.data_folder + f"/alignment/{len(self.initial_dataset)}_initial_trajectories_{len(self.modified_dataset)}_after_{self.trajectory_length}_length_{self.num_epochs}_epochs.png"
            pl_manager.plot_together(self.data_folder + f"/alignment/{len(self.initial_dataset)}_initial_trajectories_{len(self.modified_dataset)}_after_{self.trajectory_length}_length_{self.num_epochs}_epochs.png", error_save)
            return x
        else:
            pl_manager.run_learning_fixed()
            figure_save = self.data_folder + f"/fixed/{len(self.initial_dataset)}_initial_trajectories_{len(self.modified_dataset)}_after_{self.trajectory_length}_length_{self.num_epochs}_epochs_{max_queries}_queries.png"
            pl_manager.plot_more_algorithms_together(figure_save, error_save)
            return None


def get_semantic_map():
    """
    The map I was using was configured based on costs associated with specific 
    terrain types, so this function was used to create a new map with costs 
    translated into semantic class labels. 

    Returns:
        np.array: Returns the adjusted semantic map. 
    """
    t_costmap = np.load("traversability_costmap.npy")

    # Map current costs to new costs
    cost_to_cost = {
        0 : 60, # Maps Sand to 60 (second to highest)
        70 : 70, # Sidewalk stays the same
        58 : 30, # Trees to second lowest
        56 : 1, # Water to lowest
        50 : 40 # Rock to third lowest. 
    }

    t_costmap = np.vectorize(lambda x : cost_to_cost[x])(t_costmap)
    print(np.unique(t_costmap))

    # Map costs to terrain types:
    cost_to_terrain = {
        0 : "Sand",
        50: "Rock", 
        56: "Water",
        58: "Trees",
        70: "Sidewalk"
    }

    cost_to_terrain = {
        60 : "Sand",
        70 : "Sidewalk",
        30 : "Trees",
        1 : "Water",
        40 : "Rock"
    }

    # Obtain semantic map:
    s_costmap = np.vectorize(lambda x: cost_to_terrain[x])(t_costmap)

    # Fix issues with costmap: 
    mapping = {
        "Water" : "Sand",
        "Trees" : "Rock", 
        "Rock": "Water",
        "Sand": "Trees",
        "Sidewalk" : "Sidewalk"
    }

    for t in mapping.keys():
       print(f"{t}: {100*np.count_nonzero(s_costmap == t)/s_costmap.size}")


    costmap = np.vectorize(lambda x: mapping[x])(s_costmap)
    np.save("./semantic_map_reconfigured.npy", costmap)

    return costmap


if __name__ == "__main__":
    # Hyperparameters:
    EPOCHS = 11200
    TRAJECTORY_LENGTH = 200
    BATCH_SIZE = 32
    START_GOAL = (3200, 3200)
    END_GOAL = (3170, 3350)#(3100, 3300)
    TRAIN_SIZE = 321
    PL_SIZE = 33
    CLUSTER_THRESHOLD = 0.05
    DATA_FOLDER = "../results"

    # Load costmap containing semantic terrain types
    costmap = np.load("semantic_map_reconfigured.npy")


    terrain_types = ["Water", "Trees", "Rock", "Sand", "Sidewalk"]

#    for t in terrain_types:
#       print(f"{t}: {100*np.count_nonzero(costmap == t)/costmap.size}")
    
#    np.save("./semantic_map_reconfigured.npy", costmap)


    # Creating a fake costmap with a lot of terrains
    # terrain_types = ["Water", "Mud", "Trees", "Brush", "Gravel", "Rock", "Sand", "Grass", "Road", "Sidewalk"]
    # indx_to_terrain = {
    #     i : t for i,t in enumerate(terrain_types)
    # }
    # grid_size = (3000, 3000)
    # random_array = np.random.randint(0, len(terrain_types), size=grid_size)
    # costmap = np.vectorize(lambda x: terrain_types[x])(random_array)
    # for t in terrain_types:
    #     print(f"{t}: {100*np.count_nonzero(costmap == t)/costmap.size}")


    # Obtain a list of unique terrain types in the environment
#    terrain_types = list(np.unique(costmap))

    # Set example ground truth rewards

    rewards = {
        t : -1 + i*(2/(len(terrain_types)-1)) for i, t in enumerate(terrain_types)
    }

    print(f"Created Rewards: {list(rewards.values())}")

    tlp = TerrainLearningPipeline(
        terrain_map=costmap,
        terrain_types=terrain_types,
        ground_truth=list(rewards.values()),
        num_epochs=EPOCHS,
        trajectory_length=TRAJECTORY_LENGTH,
        batch_size=BATCH_SIZE,
        start_loc=START_GOAL,
        end_loc=END_GOAL,
        num_training_samples=TRAIN_SIZE,
        num_initial_dataset=PL_SIZE,
        cluster_balance_threshold=CLUSTER_THRESHOLD,
        use_planner=True,
        data_folder=DATA_FOLDER
    )
    

    tlp.find_best_locations()
    tlp.train_model()
    tlp.optimization_step()


    fixed_tests = [("fixed", 25, 0.5), ("fixed", 75, 0.5)]
    convergence_tests = [("convergence", 500, 0.75), ("convergence", 500, 0.75), ("convergence", 500, 0.75)]#, ("convergence", 500, 0.75), ("convergence", 500, 0.75), ("convergence", 500, 0.75), ("convergence", 500, 0.75), ("convergence", 500, 0.75), ("convergence", 500, 0.75), ("convergence", 500, 0.75), ("convergence", 500, 0.75)]
    alignment_tests = [("alignment", 200, 0.5)]*10
    alignments = [] # List of lists of tuple representing the algorithm and convergence time. [[(a1, c1), (a2, c2)]]

    for t in fixed_tests:
        print(f"Running {t[0]} for up to {t[1]} queries with a threshold of {t[2]}.")
        tlp.run_preference_learning(
            learning_type=t[0],
            max_queries=t[1],
            threshold=t[2]
        )
    

    
#    for t in alignment_tests:
#        print(f"Running {len(alignment_tests)} Alignment Tests!")
#        ctime = tlp.run_preference_learning(
#            learning_type=t[0],
#            max_queries=t[1],
#            threshold=t[2]
#        )

#        alignments.append(ctime)


    # # I want a plot of convergence results (maybe 5 repetitions on a histogram?)
    # # Maybe a plot of the cluster counts before and after the pre-optimization step. 
#    histogram_plot(
#        data=alignments,#[[("mutual_information", 215), ("variational_info", 33)], [("mutual_information", 230), ("variational_info", 375)]],
#        save_path=DATA_FOLDER + f"/alignment/{EPOCHS}_epochs_{TRAJECTORY_LENGTH}_length_{PL_SIZE}_trajectory_set.png",
#        plot_title=f"Alignment Over {len(alignments)} Trials",
#        plot_yaxis="Time to Ground Truth Alignment",
#        plot_xaxis="Trial Number",
#        plot_colors=["#FAC05E", "#5448C8"]
#    )

    
#    convergences = []
#    for t in convergence_tests:
#        print(f"Running {len(convergence_tests)} Convergence Tests!")
#        ctime = tlp.run_preference_learning(
#            learning_type=t[0],
#            max_queries=t[1],
#            threshold=t[2]
#        )

#        convergences.append(ctime)


    # I want a plot of convergence results (maybe 5 repetitions on a histogram?)
    # Maybe a plot of the cluster counts before and after the pre-optimization step. 
#    histogram_plot(
#        data=convergences,#[[("mutual_information", 215), ("variational_info", 33)], [("mutual_information", 230), ("variational_info", 375)]],
#        save_path=DATA_FOLDER + f"/convergence/{EPOCHS}_epochs_{TRAJECTORY_LENGTH}_length_{PL_SIZE}_trajectory_set.png",
#        plot_title=f"Convergence Over {len(convergences)} Trials",
#        plot_yaxis="Time to Ground Truth Convergence",
#        plot_xaxis="Trial Number",
#        plot_colors=["#FAC05E", "#5448C8"]
#    )


    # categories = [0, 25, 50, 75, 100]

    # water_map = tlp.best_locs["Sand"][0]
    # sidewalk_map = tlp.best_locs["Sidewalk"][0]
    # print(f"Unique Terrains in Sidewalk Map: {np.unique(sidewalk_map)}")
    # print(f"Unique Terrains in Sand Map: {np.unique(water_map)}")

    # print(water_map)
    # print(sidewalk_map)

#    water_categories = [100, 0, 25, 50, 75]
    # water_categories = {
    #     "Water": 0,
    #     "Trees": 25,
    #     "Rock": 50,
    #     "Sand": 100,
    #     "Sidewalk": 75
    # }
#    sw_categories = [0, 25, 50, 75, 100]
    # sw_categories = {
    #     "Sidewalk": 100,
    #     "Sand": 75,
    #     "Rock": 50,
    #     "Trees": 25,
    #     "Water": 0
    # }

#    water_label_to_value = dict(zip(np.unique(water_map), water_categories[:len(np.unique(water_map))])) 
    #water_map = np.vectorize(water_label_to_value.get)(water_map)
#    print(f"Unique values in water map: {np.unique(water_map)}")

#    sidewalk_label_to_value = dict(zip(np.unique(sidewalk_map), sw_categories[:len(np.unique(sidewalk_map))]))
#    sidewalk_map = np.vectorize(sidewalk_label_to_value.get)(sidewalk_map)
#    print(f"Unique values in sidewalk map: {np.unique(sidewalk_map)}")

    # Plan paths 
    # Water
    # tlp.modified_dataset.smaller_costmap = water_map
    # water_traj = torch.tensor(tlp.modified_dataset.integer_label_encoding(tlp.modified_dataset.plan_path(list(water_categories.values()), water_categories)[0]), device=tlp.device)

    # if len(water_traj) <= tlp.trajectory_length:
    #     print("Expanding sand path")
    #     water_traj = torch.cat((water_traj, water_traj[:tlp.trajectory_length - len(water_traj)]))
        
    # else:
    #     print("Shortening sand path")
    #     water_traj = water_traj[:tlp.trajectory_length]

    # tlp.modified_dataset.trajectories = torch.cat((tlp.modified_dataset.trajectories, water_traj.unsqueeze(0)))

    # # Sidewalk
    # tlp.modified_dataset.smaller_costmap = sidewalk_map
    # sidewalk_traj = torch.tensor(tlp.modified_dataset.integer_label_encoding(tlp.modified_dataset.plan_path(list(sw_categories.values()), sw_categories)[0]), device=tlp.device)
   
    # if len(sidewalk_traj) <= tlp.trajectory_length:
    #     print("Expanding sidewalk path")
    #     sidewalk_traj = torch.cat((sidewalk_traj, sidewalk_traj[:tlp.trajectory_length - len(sidewalk_traj)]))
        
    # else:
    #     print("Shortening sidewalk path")
    #     sidewalk_traj = sidewalk_traj[:tlp.trajectory_length]

    # tlp.modified_dataset.trajectories = torch.cat((tlp.modified_dataset.trajectories, sidewalk_traj.unsqueeze(0)))

    


#    value_to_label = dict(zip(np.unique(costmap), categories[:len(np.unique(costmap))]))#tlp.modified_dataset.smaller_costmap))]))
#    costmap = np.vectorize(value_to_label.get)(costmap)#tlp.modified_dataset.smaller_costmap)
    # print(np.unique(costmap))

    # visualize_query(
    #     costmap=costmap,
    #     trajectories=[],#[tlp.modified_dataset.trajectory_poses[0], tlp.modified_dataset.trajectory_poses[2]],
    #     colors=[]#["#FAC05E", "#5448C8"]
    # )

#    visualize_double_costmap_query(
#        [costmap, costmap],
#        [tlp.best_locs["Water"], tlp.best_locs["Sidewalk"]],
#        [sidewalk_map, water_map],
#        [tlp.modified_dataset.trajectory_poses[-1], tlp.modified_dataset.trajectory_poses[-2]],
#        colors=["#FAC05E", "#5448C8"],
#        mappings=[sw_categories, water_categories]
#    )