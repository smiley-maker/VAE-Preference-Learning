# Regular Imports:
from src.VAE.utils.imports import *

# Model Imports:
from src.VAE.architectures.vae_lstm import MemoryVAE
from src.VAE.training_scripts.train_vae import TrainVAE

# Data Handlers: 
from src.VAE.data_handlers.trajectory_segment_gatherer import TrajectoryDataset
from src.VAE.vizualization_scripts.ClusteredTSNE import cluster_latent_space, visualize_latent_space
from src.VAE.vizualization_scripts.histogram_plot import histogram_plot

# Preference Learning Imports:
from src.VAE.pipelines.preference_learning import TerrainLearning
from src.VAE.utils.simulated_user import SimUser

class AlpsTrainingPipeline:
    def __init__(
            self,
            terrain_map : np.array,
            terrain_types : list[str],
            ground_truth : dict[str:float],
            trajectory_length : int,
            start_position : tuple[int,int],
            end_position : tuple[int, int],
            num_training_samples : int,
            num_initial_dataset : int,
            num_epochs : int,
            batch_size : int,
            cluster_balance_threshold : float,
            use_planner : bool,
            use_sim : bool,
            data_folder : str
    ) -> None:
        
        """
        This class facilitates preference learning in the
        alps environment. 
        """

        self.terrain_map = terrain_map
        self.terrain_types = terrain_types
        self.ground_truth = ground_truth
        self.trajectory_length = trajectory_length
        self.start_pos = start_position
        self.end_pos = end_position
        self.num_training = num_training_samples
        self.num_initial = num_initial_dataset
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.threshold = cluster_balance_threshold
        self.use_planner = use_planner
        self.use_sim = use_sim
        self.data_folder = data_folder

        # Create a simulated user if desired. 
        if self.use_sim:
            self.simuser = SimUser(
                list(self.ground_truth.values()),
                list(self.ground_truth.keys())
            )

            # Obtain ground truth ordering
            self.reward_ordering = self.simuser.get_preferential_order(self.ground_truth)
            print(f"Ground Truth Reward Ranking: {self.reward_ordering}")

        # Create mappings from terrain types to indices, and vice versa.
        self.terrain_to_idx = {
            t : i for i,t in enumerate(self.terrain_types)
        }

        self.idx_to_terrain = {
            i : t for i,t in enumerate(self.terrain_types)
        }

        # Create a device for tensor management
        self.device = torch.device("mps")

    
    def collate(self, batch) -> torch.tensor:
        """Collate function use to create a stacked batch for data loading.

        Args:
            batch (tensor): batch of n trajectories.
        
        Returns:
            Tensor: stacked version of the batch compatible with PyTorch.
        """

        return torch.stack(batch)
    
    def feature_func(self, traj : Trajectory) -> tuple[np.array, list[str]]:
        """Returns the terrain feature breakdown for a given trajectory.

        Args:
            traj (Trajectory): trajectory containing encoded terrains.

        Returns:
            tuple[np.array, list[str]]: the feature breakdown and the
                                        terrain types sorted by frequency.
        """

        features = {}

        # convert the trajectory to a numpy array
        traj = np.array(traj, dtype=int)
        N = len(traj) # N terrain types

        # Counts the number of times each terrain type appears in the trajectory
        for t in self.terrain_types:
            count = np.count_nonzero(traj == self.terrain_to_idx[t])
            features[t] = 2*(count / N) -1 # Converts to percentage of trajectory and scales to be between -1 and 1
        
        # Obtain the preferential ranking for preference elicitation. 
        sorted_features = self.simuser.get_preferential_order(features)

        # Return the feature importances and terrain ranking
        return np.array(list(features.values())), sorted_features
    
    def find_max_terrain_region(
            self, 
            terrain_type : str,
            region_width : int,
            region_height : int
    ) -> tuple[np.array, tuple[int, int]]:
        """
        Locates the map region with the most occurrences of the 
        desired terrain type. 

        Args:
            terrain_type (str): the terrain type to search for
            region_width (int): desired width of costmap region
            region_height (int): desired height of costmap region

        Returns:
            tuple[np.array, tuple[int, int]]: a tuple containing:
                - the 2D costmap region of size (region_width, region_height)
                - a tuple the starting row and column index of the region
        """

        rows, cols = self.terrain_map.shape
        max_count = 0
        best_start = (0,0)
        best_region = None

        for i in range(0, rows - region_height + 1, region_height//3):
            for j in range(0, cols - region_width + 1, region_width//3):
                region = self.terrain_map[j:j+region_width, i:i+region_height]
                count = np.count_nonzero(region == terrain_type)
                if count > max_count:
                    best_region = region
                    max_count = count
                    best_start = (i,j)
                
        print(f"Found region for {terrain_type} with {np.count_nonzero(best_region == terrain_type)} occurrences.")
        return best_region, best_start
    
    def find_all_best_locations(self):
        """
        Identifies the regions of the map that consist of
        the most of each terrain type using an overlapping
        grid search (faster than checking every column/row).
        """

        # Calculate desired region width and height
        w = abs(self.start_pos[0] - self.end_pos[0]) + 600
        h = abs(self.start_pos[1] - self.end_pos[1]) + 600

        # Find the best location for each terrain type
        self.best_locs = {
            t : self.find_max_terrain_region(t, w, h) for t in self.terrain_types
        }
    
    def train_model(self):
        """
        Creates the training data and trains the VAE model.
        """

        # Create a training dataset consisting of trajectories
        # collected randomly from the environment with length L.
        training_dataset = TrajectoryDataset(
            data=self.terrain_map,
            num_samples=self.num_training,
            segment_length=self.trajectory_length - 1,
            terrain_types=self.terrain_types,
            device=self.device
        )

        dataloader = DataLoader(
            training_dataset,
            batch_size=self.batch_size,
            drop_last=True,
            collate_fn=self.collate
        )

        # Create the model class
        self.model = MemoryVAE(
            batch_size=self.batch_size,
            terrain_classes_count=len(self.terrain_types),
            trajectory_length=self.trajectory_length,
            latent_dim=len(self.terrain_types),
            hidden_size=512,
            device=self.device
        ).to(self.device)

        # Create optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=3e-4)

        # Instantiate a trainer to train the model
        trainer = TrainVAE(
            model=self.model,
            optimizer=optimizer,
            epochs=self.num_epochs,
            batch_size=self.batch_size,
            data=dataloader,
            xdim=len(self.terrain_types)*self.trajectory_length,
            device=self.device
        )

        print("Created trainer, training VAE.....")
        trainer.train_model()
    
    def optimization_step(self):
        """
        Optimizes an initial trajectory set based on the latent
        vectors using K-Means clustering to identify gaps. 
        """

        # Create an initial dataset for preference learning
        self.initial_dataset = TrajectoryDataset(
            data=self.terrain_map,
            num_samples=self.num_initial,
            segment_length=self.trajectory_length -1,
            terrain_types=self.terrain_types,
            device=self.device,
            start_point=self.start_pos,
            end_point=self.end_pos
        )

        dataloader = DataLoader(
            self.initial_dataset,
            batch_size=1,
            drop_last=False,
            collate_fn=self.collate
        )

        # Makes a copy of the initial dataset to modify. 
        self.modified_dataset = copy.deepcopy(self.initial_dataset)

        # Cluster the latent representations of the intiial trajectory set
        self.clusters = cluster_latent_space(
            model=self.model,
            data_loader=dataloader,
            device=self.device,
            mapping=self.idx_to_terrain
        )

        # Since we want to semantically tie clusters to terrain types, they 
        # must be remapped using a majority vote system. 
        # Plan to improve on this in the future with different clustering systems.
        
        # Maintains a list of semantic labels that haven't been assigned yet. 
        available_options = copy.deepcopy(self.terrain_types)

        cluster_mapping = {}

        for idx in np.unique(self.clusters):
            # Obtain all trajectories tagged with cluster idx
            trajectories_idx = [c for c in self.clusters if c == idx]

            # Create a dictionary to store the counts for each terrain type
            # A terrain type gets a count if it is found as the highest percentage
            # in one of the trajectories. 
            terrain_dict = {
                t : 0 for t in self.terrain_types
            }

            for jdx in list(trajectories_idx):
                # Obtain reward order from the feature function
                _, sorted_terrains = self.feature_func(
                    self.initial_dataset[jdx].cpu()
                )

                # Increment the count for most important terrain type
                terrain_dict[sorted_terrains[-1]] += 1
            
            print(f"Terrain counts for {self.idx_to_terrain[idx]}: {terrain_dict}")

            # Based on the majority vote, select the first available option
            # that maximizes the terrain counts. 
            choice = max(terrain_dict, key=terrain_dict.get)
            while choice not in available_options and len(available_options) > 0:
                terrain_dict[choice] = -1
                choice = max(terrain_dict, key=terrain_dict.get)
            
            cluster_mapping[idx] = self.terrain_to_idx[choice]
            available_options.remove(choice)
        
        print(f"Cluster to terrain mapping: {cluster_mapping}")

        # Now we can use this mapping to modify the clusters list to tie it with terrains. 
        self.tclusters = np.array([cluster_mapping[c] for c in self.clusters])

        # Visualize the clustered latent vectors in a 3D downsampled plot. 
        visualize_latent_space(
            model=self.model,
            data_loader=dataloader,
            device=self.device,
            mapping=self.idx_to_terrain,
            labels=self.tclusters
        )

        # Now that the clusters are mapped correctly to terrains, we can look for imbalance 
        # between them to identify less repreesented terrain types. 
        cluster_counts = [np.count_nonzero(
            self.tclusters == self.terrain_to_idx[t]
        ) for t in self.terrain_types]

        print(f"Cluster Counts: {cluster_counts}")

        # Determine the size difference between the maximum and minimum clusters
        max_size = max(cluster_counts)
        min_size = min(cluster_counts)
        min_cluster = np.argmin(cluster_counts)
        max_cluster = np.argmax(cluster_counts)

        # Continue until the clusters are balanced to within some threshold
        while ((max_size - min_size) / max_size) > self.threshold:
            print(f"Min Cluster: {self.idx_to_terrain[min_cluster]}")
            print(f"Max Cluster: {self.idx_to_terrain[max_cluster]}")

            # Assign new weights with the highest going to the min cluster and the lowest to the max
            new_weights = random.sample(range(10, 90), len(self.terrain_types)-2)
            random.shuffle(new_weights)
            new_weights.insert(min_cluster, 100)
            new_weights.insert(max_cluster, 0)

            terrain_weight_mapping = {
                t : w for t,w in zip(self.terrain_types, new_weights)
            }

            print(f"Terrain Weight Mapping: {terrain_weight_mapping}")

            # Find region of costmap with most of min terrain type
            self.modified_dataset.smaller_costmap = self.best_locs[self.idx_to_terrain[min_cluster]][0]
            print(f"Modified map to have nique values: {np.unique(self.modified_dataset.smaller_costmap)}")

            # Plan a new trajectory using these weights and map. 
            new_traj = torch.tensor(
                self.modified_dataset.integer_label_encoding(
                    self.modified_dataset.plan_path(
                        new_weights, terrain_weight_mapping
                    )[0]
                ), device=self.device
            )

            # Pad or shorten trajectory to fit expected length L for the encoder
            # Note trajectory states are still stored for visualization of the original
            if len(new_traj) <= self.trajectory_length:
                new_traj = torch.cat((new_traj, new_traj[:self.trajectory_length - len(new_traj)]))
            
            else:
                new_traj = new_traj[:self.trajectory_length]
            
            print(f"Adding trajectory of size {len(new_traj)} to {self.terrain_types[min_cluster]}")
            print(f"Trajectory Breakdown: {self.feature_func(new_traj.cpu())}")

            # Update the dataset to include the new trajectory
            # TODO: think about whether to add the trajectory to the intended cluster
            #       or to add it to the one corresponding to the highest feature
            self.modified_dataset.trajectories = torch.cat((self.modified_dataset.trajectories, new_traj.unsqueeze(0)))
            self.tclusters = np.append(self.tclusters, min_cluster)
            
            # Recalculate the cluster distribution
            cluster_counts = [np.count_nonzero(
                self.tclusters == self.terrain_to_idx[t]
            ) for t in self.terrain_types]

            # Recalculate min and max sizes
            max_size = max(cluster_counts)
            min_size = min(cluster_counts)
            max_cluster = np.argmax(cluster_counts)
            min_cluster = np.argmin(cluster_counts)

            print(f"Min:Max Ratio: {(max_size - min_size)/max_size}")
        
        print(f"Finished optimizing dataset! Added {len(self.initial_dataset) - len(self.modified_dataset)} trajectories.")
    

    def run_preference_learning(
            self, 
            learning_type : str, 
            max_queries : int, 
            threshold : float
        ) -> None | int:
        
        # Creates a fake environment to work with APREL
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

        

        trajectory_set = TrajectorySet(trajectories)
        trajectory_set2 = TrajectorySet(trajectories2)


        # Instantiates preference learning manager
        pl_manager = TerrainLearning(
            datasets=[trajectory_set, trajectory_set2],
            simuser = self.simuser,
            ground_truth=self.ground_truth.values(),
            algorithms=["mutual_information", "variational_info"],
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
            pl_manager.plot_together(figure_save, error_save)
            return None
    



if __name__ == "__main__":
    # Hyperparameters
    EPOCHS = 200
    TRAJECTORY_LENGTH = 200
    BATCH_SIZE = 32
    START_GOAL = (2500, 1120)
    END_GOAL = (2330, 1100)
    TRAIN_SIZE = 321
    PL_SIZE = 49
    CLUSTER_THRESHOLD = 0.05
    DATA_FOLDER = "../results"

    # Load the map, which is stored as a png file. 
    costmap = cv2.imread("husky.png")
    costmap = cv2.cvtColor(costmap, cv2.COLOR_BGR2GRAY)
    print(np.unique(costmap))

    terrain_types = [
        "Glacier",
        "Snowfield",
        "Permafrost",
        "RockFace",
        "Scree",
        "TalusSlope",
        "Moraine",
        "AlpineMeadow",
        "AlpineScrub",
        "AlpineTundra",
        "ConiferousForest",
        "DeciduousForest",
        "MixedForest",
        "DwarfForest",
        "ForestEdge",
        "AlpineLake",
        "GlacialTarn",
        "Wetland",
        "Meadow",
        "Bog",
        "Scrubland",
        "Grassland",
        "Heath",
        "BarrenLand",
        "Cave",
        "Cliff",
        "Ridge",
        "Valley",
        "Pass",
        "Crag",
        "Ravine"
    ]
    
    ground_truth = list((np.unique(costmap) / 255)*2 - 1) # Scales to -1 and 1
    reward_mapping = {
        t : g for t,g in zip(terrain_types, ground_truth)
    }

    value_mapping = {
        g : t for t,g in zip(terrain_types, np.unique(costmap))
    }

    print(value_mapping)

    costmap = np.vectorize(lambda x: value_mapping[x])(costmap)
    
    # Create learning pipeline
    atp = AlpsTrainingPipeline(
        terrain_map=costmap,
        terrain_types=terrain_types,
        ground_truth=reward_mapping,
        num_epochs=EPOCHS,
        trajectory_length=TRAJECTORY_LENGTH,
        batch_size=BATCH_SIZE,
        start_position=START_GOAL,
        end_position=END_GOAL,
        num_training_samples=TRAIN_SIZE,
        num_initial_dataset=PL_SIZE,
        cluster_balance_threshold=CLUSTER_THRESHOLD,
        use_planner=True,
        data_folder=DATA_FOLDER,
        use_sim=True
    )

    # Get best locations
    atp.find_all_best_locations()
    # Training...
    atp.train_model()
    # Optimization stage
    atp.optimization_step()

    # Tests ->
    convergence_tests = [("alignment", 75, 0.75)]*3

    convergences = []
    for ct in convergence_tests:
        convergence = atp.run_preference_learning(
            learning_type=ct[0],
            max_queries=ct[1],
            threshold=ct[2]
        )

        convergences.append(convergence)

    # Plot histogram
    histogram_plot(
        data=convergences,
        save_path=DATA_FOLDER + f"/alignment/{EPOCHS}_epochs/{TRAJECTORY_LENGTH}_length_{PL_SIZE}_trajectory_set.png",
        plot_title=f"Alignment Results Over {len(convergences)} Trials",
        plot_yaxis="Time to Ground Truth Alignment",
        plot_xaxis="Trial Number",
        plot_colors=["#FAC05E", "#5448C8"]
    )

