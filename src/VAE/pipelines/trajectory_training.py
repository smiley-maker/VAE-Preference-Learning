from src.VAE.utils.imports import *
from src.VAE.utils.simulated_user import SimUser
from src.VAE.data_handlers.trajectory_segment_gatherer import TrajectoryDataset
from src.VAE.training_scripts.train_vae import TrainVAE
from src.VAE.architectures.vae_lstm import MemoryVAE
from src.VAE.utils.moving_average import moving_average
from src.VAE.vizualization_scripts.ClusteredTSNE import cluster_latent_space, visualize_latent_space
from src.VAE.pipelines.preference_learning import TerrainLearning
import copy
import json
import math
import os
import networkx as nx
from src.VAE.vizualization_scripts.histogram_plot import histogram_plot
from src.VAE.vizualization_scripts.costmap_query_viz import visualize_query

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

        # Let's think about the best way to construct the pairwise relationship
        # between rewards and terrain types. I could use a dictionary, which 
        # seems like it would make sense to make sure they remain tied. 
        self.terrain_rewards_ground_truth = {
            t : r for t,r in zip(self.terrain_types, self.ground_truth)
        }

        # Then we can still get the preferential ordering using the simulated
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

        print(self.index_to_terrain)
        print(self.terrain_types)

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
            #new_weights = list(range(int(100/len(self.terrain_types)), 101-2*(int(100/len(self.terrain_types))), int(100/len(self.terrain_types))))
            print(f"Min Cluster: {min_cluster}, or {self.index_to_terrain[min_cluster]}")
            print(f"Max Cluster: {max_cluster}, or {self.index_to_terrain[max_cluster]}")
            new_weights = [25, 50, 75]
            random.shuffle(new_weights)
            new_weights.insert(min_cluster, 100)
            new_weights.insert(max_cluster, 0)
            print("New Weights:")
            print(new_weights)
            print("Supposed Shape:")
            print(self.best_locs[self.index_to_terrain[min_cluster]][0].shape)
            self.modified_dataset.smaller_costmap = self.best_locs[self.index_to_terrain[min_cluster]][0]
            print(self.modified_dataset.smaller_costmap.shape)

            new_traj = torch.tensor(self.modified_dataset.integer_label_encoding(self.modified_dataset.plan_path(new_weights)[0]), device=self.device)

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
#            cluster_counts = [np.count_nonzero(self.clusters == t) for t in range(len(self.terrain_types))]
            cluster_counts = [np.count_nonzero(self.clusters == t) for t in [self.terrain_to_index[x] for x in self.terrain_types]]
            print(cluster_counts)
            max_size = max(cluster_counts)
            min_size = min(cluster_counts)
            min_cluster = np.argmin(cluster_counts)
            max_cluster = np.argmax(cluster_counts)
            print((max_size - min_size)/max_size)



#         t = 0
#         while (max_size - min_size)/max_size > self.threshold and t < len(self.initial_dataset):
#             # Obtain a new trajectory using the path planner
#             # - Prioritize the terrain type which belongs to the minimum
# #            x = list(range(0, 101, int(100/len(self.terrain_types))))
# #            new_weights = list(range(25, 76, 25))
#             new_weights = list(range(int(100/len(self.terrain_types)), 101-2*(int(100/len(self.terrain_types))), int(100/len(self.terrain_types))))
#             random.shuffle(new_weights)
#             new_weights.insert(min_cluster, 100)
#             new_weights.insert(max_cluster, 0)
#             print(new_weights)

#             new_traj = torch.tensor(self.modified_dataset.integer_label_encoding(self.modified_dataset.plan_path(new_weights)[0]), device=self.device)
#             # Let's pad or clip the length 
#             if len(new_traj) <= self.trajectory_length:
#                 # Pad the new trajectory up to the desired length
#                 new_traj = torch.cat((new_traj, new_traj[:self.trajectory_length - len(new_traj) ]))
#             else:
#                 # Stop the trajectory at the segment length
#                 new_traj = new_traj[:self.trajectory_length]
            
#             print(f"Adding to {min_cluster}...")
#             print(f"Trajectory Breakdown: {self.feature_func(new_traj.cpu())}")
            
#             self.modified_dataset.trajectories = torch.cat((self.modified_dataset.trajectories, new_traj.unsqueeze(0)))
#             dataloader = DataLoader(self.modified_dataset, batch_size=1, drop_last=False, collate_fn=self.collate)
#             self.clusters = cluster_latent_space(
#                 self.model, 
#                 dataloader, 
#                 self.device, 
#                 mapping=self.index_to_terrain
#             )

#             cluster_counts = [np.count_nonzero(self.clusters == t) for t in range(len(self.terrain_types))]
#             max_size = max(cluster_counts)
#             min_size = min(cluster_counts)
#             min_cluster = np.argmin(cluster_counts)
#             max_cluster = np.argmax(cluster_counts)
#             print((max_size - min_size)/max_size)

#             t += 1
        
#        dataloader2 = DataLoader(self.modified_dataset, batch_size=1, drop_last=False, collate_fn=self.collate)
#        visualize_latent_space(self.model, dataloader2, self.device, mapping=self.index_to_terrain, labels=self.clusters)

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
    
    def run_preference_learning(self, learning_type, max_queries : int, threshold : float):
        
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
            datasets=[trajectory_set, trajectory_set2],
            simuser = self.simuser,
            ground_truth=self.ground_truth,
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
    # Hyperparameters:
    EPOCHS = 11250
    TRAJECTORY_LENGTH = 200
    BATCH_SIZE = 32
    START_GOAL = (3200, 3200)
    END_GOAL = (3170, 3350)#(3100, 3300)
    TRAIN_SIZE = 481
    PL_SIZE = 33
    CLUSTER_THRESHOLD = 0.15
    DATA_FOLDER = "../results"

    # Load costmap containing semantic terrain types
    costmap = np.load("semantic_map.npy")

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
    terrain_types = ["Water", "Rock", "Sand", "Trees", "Sidewalk"]

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
    
#    if os.path.exists("./saved_regions.json"):
        # Read the data
#        with open("./saved_regions.json", "r") as f:
#            tlp.best_locs = json.load(f)
#    else:
#        tlp.find_best_locations()
        # Save to file
#        with open("terrain_locations.json", "w") as f:
 #           json.dump(tlp.best_locs, f)


    tlp.find_best_locations()
    tlp.train_model()
    tlp.optimization_step()

   # tests = [("convergence", 1000, 0.5), ("convergence", 1000, 1.0), ("fixed", 25, 0.1), ("fixed", 50, 0.1), ("fixed", 100, 0.1)]
#    tests = [("alignment", 1000, 0.5), ("alignment", 1000, 0.5), ("alignment", 1000, 0.5)]
    fixed_tests = [("fixed", 25, 0.5), ("fixed", 50, 0.5), ("fixed", 75, 0.5)]
    convergence_tests = [("convergence", 500, 0.75), ("convergence", 500, 0.75), ("convergence", 500, 0.75)]#, ("convergence", 500, 0.75), ("convergence", 500, 0.75), ("convergence", 500, 0.75), ("convergence", 500, 0.75), ("convergence", 500, 0.75), ("convergence", 500, 0.75), ("convergence", 500, 0.75), ("convergence", 500, 0.75)]
    alignment_tests = [("alignment", 200, 0.5), ("alignment", 200, 0.5), ("alignment", 200, 0.5), ("alignment", 200, 0.5), ("alignment", 200, 0.5), ("alignment", 200, 0.5), ("alignment", 200, 0.5), ("alignment", 200, 0.5), ("alignment", 200, 0.5), ("alignment", 200, 0.5)]
    alignments = [] # List of lists of tuple representing the algorithm and convergence time. [[(a1, c1), (a2, c2)]]

    # for t in fixed_tests:
    #     print(f"Running {t[0]} for up to {t[1]} queries with a threshold of {t[2]}.")
    #     tlp.run_preference_learning(
    #         learning_type=t[0],
    #         max_queries=t[1],
    #         threshold=t[2]
    #     )
    

    
    # for t in alignment_tests:
    #     print(f"Running {len(alignment_tests)} Alignment Tests!")
    #     ctime = tlp.run_preference_learning(
    #         learning_type=t[0],
    #         max_queries=t[1],
    #         threshold=t[2]
    #     )

    #     alignments.append(ctime)


    # # I want a plot of convergence results (maybe 5 repetitions on a histogram?)
    # # Maybe a plot of the cluster counts before and after the pre-optimization step. 
    # histogram_plot(
    #     data=alignments,#[[("mutual_information", 215), ("variational_info", 33)], [("mutual_information", 230), ("variational_info", 375)]],
    #     save_path=DATA_FOLDER + f"/alignment/{EPOCHS}_epochs_{TRAJECTORY_LENGTH}_length_{PL_SIZE}_trajectory_set.png",
    #     plot_title=f"Alignment Over {len(alignments)} Trials",
    #     plot_yaxis="Time to Ground Truth Alignment",
    #     plot_xaxis="Trial Number",
    #     plot_colors=["#FAC05E", "#5448C8"]
    # )

    
    convergences = []
    for t in convergence_tests:
        print(f"Running {len(convergence_tests)} Convergence Tests!")
        ctime = tlp.run_preference_learning(
            learning_type=t[0],
            max_queries=t[1],
            threshold=t[2]
        )

        convergences.append(ctime)


    # I want a plot of convergence results (maybe 5 repetitions on a histogram?)
    # Maybe a plot of the cluster counts before and after the pre-optimization step. 
    histogram_plot(
        data=convergences,#[[("mutual_information", 215), ("variational_info", 33)], [("mutual_information", 230), ("variational_info", 375)]],
        save_path=DATA_FOLDER + f"/convergence/{EPOCHS}_epochs_{TRAJECTORY_LENGTH}_length_{PL_SIZE}_trajectory_set.png",
        plot_title=f"Convergence Over {len(convergences)} Trials",
        plot_yaxis="Time to Ground Truth Convergence",
        plot_xaxis="Trial Number",
        plot_colors=["#FAC05E", "#5448C8"]
    )


    # categories = [0, 25, 50, 75, 100]

    # value_to_label = dict(zip(np.unique(costmap), categories[:len(np.unique(costmap))]))#tlp.modified_dataset.smaller_costmap))]))
    # costmap = np.vectorize(value_to_label.get)(costmap)#tlp.modified_dataset.smaller_costmap)
    # print(np.unique(costmap))

    # visualize_query(
    #     costmap=costmap,
    #     trajectories=[],#[tlp.modified_dataset.trajectory_poses[0], tlp.modified_dataset.trajectory_poses[2]],
    #     colors=[]#["#FAC05E", "#5448C8"]
    # )
    



""" if __name__ == "__main__":
    # Create device
    device = torch.device("mps")
    print(f"Using {device}")

#    categories = ["Grass", "Road", "Sidewalk", "Water", "Trees", "Rock", "Brush", "Sand"]
    categories = ["Water", "Sand", "Rock", "Trees", "Sidewalk"]
    mapping_index_to_terrain = {
            i : t for i,t in enumerate(categories)
        }


#    grid_size = (2000, 2000)

    #p = np.ones(len(categories)) / len(categories)
#    random_array = np.random.multinomial(grid_size, p, size=1)

#    random_array = np.random.randint(0, len(categories), size=grid_size)
#    costmap = np.vectorize(lambda x: categories[x])(random_array)

#    costmap = np.load("traversability_costmap.npy")
#    value_to_label = dict(zip(np.unique(costmap), categories[:len(np.unique(costmap))]))
#    costmap = np.vectorize(value_to_label.get)(costmap)
#    print(np.unique(costmap))

#    np.save("./semantic_map.npy", costmap)
    costmap = np.load("semantic_map.npy")
#    costmap = condense_map(costmap, 4)
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


    start_loc = (812, 500)
    goal_loc = (850, 530)

    #model = torch.load("../results/models/model7500.pt")


    # Load data
    dataset = TrajectoryDataset(
        data=costmap,
        num_samples=321,
        segment_length=segment_size-1,
        terrain_types=categories,
#        start_point = start_loc,#(random.randint(0, len(costmap)), random.randint(0, len(costmap[0]))),
#        end_point = goal_loc, #(random.randint(0, len(costmap)), random.randint(0, len(costmap[0]))),
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
#    model = torch.load("../results/models/model7500epochs321samplesLSTM.pt")
#    model = torch.load("../results/models/model5000epochs321samplesLSTM.pt")

    dataset = TrajectoryDataset(
        data=costmap,
        num_samples=33,
        segment_length=segment_size-1,
        terrain_types=categories,
        start_point = start_loc,
        end_point = goal_loc,
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

    # TODO: Modify the loop below to use the networkx path planner
    # rather than the decoder. 
    smaller_costmap = dataset.smaller_costmap
    print(smaller_costmap.shape)
    while (max_size - min_size)/max_size > threshold:
        # Obtain a new trajectory using the path planner
        # - Prioritize the terrain type which belongs to the minimum
        # - Maybe assign weights decreasing with size increase. 
        
        new_weights = list(range(0, 76, 25))
        random.shuffle(new_weights)
        new_weights.insert(min_cluster, 100)

        print(f"New Weights: {new_weights}")

#        new_weights = [0.99 if i == min_cluster else random.uniform(0.0, 100.0) for i in range(len(categories))]
#                       random.randchoice(-1, 1, step=0.01) for i in range(len(categories))]
#        new_traj = plan_path(smaller_costmap, new_weights, start_loc, goal_loc, device=device)
        new_traj = torch.tensor(dataset2.integer_label_encoding(dataset2.plan_path(new_weights)), device=device)
        # Let's pad or clip the length 
        if len(new_traj) <= segment_size:
            # Pad the new trajectory up to the desired length
            print("Padding trajectory...")
            new_traj = torch.cat((new_traj, new_traj[:dataset2.segment_length - len(new_traj) + 1]))
            print(new_traj.size())
        else:
            print("Shortening trajectory")
            print(new_traj.size())
            new_traj = new_traj[:dataset2.segment_length+1]
            print(new_traj.size())
        dataset2.trajectories = torch.cat((dataset2.trajectories, new_traj.unsqueeze(0)))
        clusters = np.append(clusters, min_cluster)
        cluster_counts = [np.count_nonzero(clusters == t) for t in range(len(categories))]
        max_size = max(cluster_counts)
        min_size = min(cluster_counts)
        min_cluster = np.argmin(cluster_counts)
        print((max_size - min_size)/max_size)


#    while (max_size - min_size)/min_size > threshold:
        # Obtain a new trajectory using decoder
        # - Determine which cluster belongs to the minimum
        # - Put a trajectory of all that terrain type into the model
        # - Get a decoded output to add to that cluster. 
 #       sample_traj = torch.tensor([min_cluster]*segment_size, dtype=torch.float, device=device)
#        new_trajectory = model(torch.stack([sample_traj]))[0].round().long()
#        dataset2.trajectories = torch.cat((dataset2.trajectories, new_trajectory))
#        print(len(dataset2))
#        clusters = np.append(clusters, min_cluster)
#        cluster_counts = [np.count_nonzero(clusters == t) for t in range(len(categories))]
#        max_size = max(cluster_counts)
#        min_size = min(cluster_counts)
#        min_cluster = np.argmin(cluster_counts)
#        print((max_size - min_size)/min_size)

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

    pl_manager = TerrainLearning(
        [trajectory_set, trajectory_set2],
        rewards,
        ["mutual_information", "variational_info"],
        categories,
        random.sample(list(np.linspace(-1, 1, 201)), 5),
        25,
        True,
        True,
        True,
        clusters
    )

    pl_manager.run_learning_alignment()
    pl_manager.plot_together(f"../results/experiment_alignment_convergence_32_trajectories_6000_epochs_{segment_size}_segment_size.png")
#    pl_manager.plot_results("variational_info", f"../results/experiment_vae_{segment_size}_segment_size_{25}_queries_{len(dataset2)}_trajectories_5000_epochs.png")
#    pl_manager.plot_results("mutual_information", f"../results/experiment_mutual_{segment_size}_segment_size_{25}_queries_{len(dataset)}_trajectories_5000_epochs.png")
#    pl_manager.run_learning_convergence(0.75)
#    pl_manager.plot_results("variational_info", f"../results/experiment_vae_{segment_size}_segment_size_convergence_0.75_{len(dataset2)}_trajectories_4000_epochs_convergence.png")
#    pl_manager.plot_results("mutual_information", f"../results/experiment_mutual_{segment_size}_segment_size_convergence_0.75_{len(dataset)}_trajectories_4000_epochs_convergence.png")


    # Creates a query optimizer
     mquery_optimizer = QueryOptimizerDiscreteTrajectorySet(trajectory_set)
    vquery_optimizer = QueryOptimizerDiscreteTrajectorySet(trajectory_set2)

    # Creates a human user manager
#    true_user = HumanUser(delay=0.5)

    # Creates a weight vector with randomized parameters
#    initial_weights = get_random_normalized_vector(num_categories)
    minitial_weights = [0.0]*len(rewards)
    mutual_params = {"weights": minitial_weights}
    vinitial_weights = [0.0]*len(rewards)
    variational_params = {"weights": vinitial_weights}

    # Creates a user model
    mutual_user_model = SoftmaxUser(mutual_params)
    variational_user_model = SoftmaxUser(variational_params)

    # Initializes the belief network using the user model and parameters
    # Comparing random sampling to variational autoencoder based sampling. 
    mutual_belief = SamplingBasedBelief(mutual_user_model, [], mutual_params)
    variational_belief = SamplingBasedBelief(variational_user_model, [], variational_params)

    # Creates an example query using the first two trajectories. 
    query = PreferenceQuery(trajectory_set[:2])

    # Dictionaries to hold the adjusted weights over time for each terrain type. 
    vweights = {
        k : [] for k in categories
    }

    mweights = {
        k : [] for k in categories
    }

    mvariance = []
    vvariance = []
    variance_difference = []

    valignment = []
    malignment = []

    time_differences = {
        "Mutual": [],
        "Variational": []
    }

    max_iterations = 49

        
    print(f"Learning with Max Entropy Query Selection")
    mutual_convergence_time = []
    mbm = mutual_belief.mean["weights"]
    mconvergence = sum([abs(mbm[k] - rewards[k]) for k in range(len(rewards))])/(len(rewards)*2)
    mutual_convergence_time.append(mconvergence)
    query_no = 0
    while query_no <= max_iterations:
        query_no += 1
        print(f"RUNNING QUERY {query_no}")
        mqueries, _ = mquery_optimizer.optimize('mutual_information', mutual_belief, query, clusters=clusters)
        avg_var_m = 0

        for x in range(len(rewards)):
            avg_var_m += abs(mqueries[0].slate[0].features[0][x] - mqueries[0].slate[1].features[0][x])
        
        avg_var_m = avg_var_m / len(rewards)
        #avg_var_v = avg_var_v / len(rewards)
        malignment.append(
            simuser.check_distribution(get_preferential_order(mutual_belief.mean["weights"], categories))
        )

        mresonpses = simuser.respond(mqueries[0])
        mutual_belief.update(Preference(mqueries[0], mresonpses[0]))
        for j, mw in enumerate(mutual_belief.mean["weights"]):
            mweights[mapping_index_to_terrain[j]].append(mw)
        
        mbm = mutual_belief.mean["weights"]
        mconvergence = sum([abs(mbm[k] - rewards[k]) for k in range(len(rewards))])/(len(rewards)*2)
        mutual_convergence_time.append(mconvergence)

    print(f"Learning with VAE Query Selection")
    query_no = 0
    variational_convergence_time = []
    vbm = variational_belief.mean["weights"]
    vconvergence = sum([abs(vbm[k] - rewards[k]) for k in range(len(rewards))])/(len(rewards)*2)
    variational_convergence_time.append(vconvergence)
    while query_no <= max_iterations:
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

        for x in range(len(rewards)):
            avg_var_v += abs(vqueries[0].slate[0].features[0][x] - vqueries[0].slate[1].features[0][x])
#            avg_var_r += abs(rqueries[0].slate[0].features[0][x] - rqueries[0].slate[1].features[0][x])
        
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
        vresponses = simuser.respond(vqueries[0])
        print(vresponses)
#        vresponses = true_user.respond(vqueries[0])


        variational_belief.update(Preference(vqueries[0], vresponses[0]))
        
        for i, vw in enumerate(variational_belief.mean["weights"]):
            vweights[mapping_index_to_terrain[i]].append(vw)
        
        vbm = variational_belief.mean["weights"]
        vconvergence = sum([abs(vbm[k] - rewards[k]) for k in range(len(rewards))])/(len(rewards)*2)
        variational_convergence_time.append(vconvergence)

    
    # Plot weights over iterations
    fig, axes = plt.subplots(2, 3, figsize=(15, 12))

    for terrain_type in categories:
        avg_vweights = moving_average(vweights[terrain_type])
        axes[0][0].plot(avg_vweights, label=terrain_type)
        axes[0][0].fill_between(range(len(avg_vweights)), avg_vweights, alpha=0.1)
        avg_mweights = moving_average(mweights[terrain_type])
        axes[0][1].plot(avg_mweights, label=terrain_type)
        axes[0][1].fill_between(range(len(avg_mweights)), avg_mweights, alpha=0.1)

    axes[0][0].set_title("Average Terrain Weights Over Time for VAE Sampling")
    axes[0][0].set_xlabel("Query Number")
    axes[0][0].set_ylabel("Terrain Weight")
    axes[0][0].legend()

    axes[0][1].set_title("Average Terrain Weights Over Time for Mutual Information")
    axes[0][1].set_xlabel("Query Number")
    axes[0][1].set_ylabel("Terrain Weight")
    axes[0][1].legend()


    axes[1][0].plot(malignment, label="Mutual Information Queries", color="#5c87ff", linewidth=2)
    axes[1][0].plot(valignment, label="VAE-based Queries", color="#f75cff", linewidth=2)
    axes[1][0].set_title("Reward Alignment Over Time")
    axes[1][0].set_xlabel("Query Number")
    axes[1][0].set_ylabel("Reward Alignment")
    axes[1][0].legend()


    axes[1][1].plot(variational_convergence_time, label="Convergence with VAE-based Queries", color="#f75cff", linewidth=2)
    axes[1][1].plot(mutual_convergence_time, label="Convergence with Mutual Information Queries", color="#5c87ff", linewidth=2)
    axes[1][1].set_title("Convergence to Ground Truth Over Time")
    axes[1][1].set_xlabel("Query Number")
    axes[1][1].set_ylabel("MAE Loss")
    axes[1][1].legend()

    vconvergences = [abs(vbm[k] - rewards[k]) for k in range(len(rewards))]
    mconvergences = [abs(mbm[k] - rewards[k]) for k in range(len(rewards))]
    axes[0][2].errorbar(categories, vbm, yerr=vconvergences, fmt='o', capsize=5, label="Reward Error for VAE-based Querying")
    axes[0][2].set_title('Ground Truth vs. Predicted Terrain Weights')
    axes[0][2].set_ylabel("Reward Convergence")
    axes[0][2].set_xlabel("Terrain Types")

    axes[1][2].errorbar(categories, mbm, yerr=mconvergences, fmt='o', capsize=5, label="Reward Error for Mutual Information Querying")
    axes[1][2].set_title('Ground Truth vs. Predicted Terrain Weights')
    axes[1][2].set_ylabel("Reward Convergence")
    axes[1][2].set_xlabel("Terrain Types")


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

#    plt.show()
    plt.savefig("../results/testingfig3.png")

    print("Summary -------------------------------")

    print(f"The reward weights were: {rewards}")

    vbm = variational_belief.mean["weights"]
    rbm = mutual_belief.mean["weights"]

    print(f"The final weights for VAE Query Selection were: {vbm}")
    print(f"The final weights for Random Query Selection were: {rbm}")
    print(f"The difference from ground truth from the VAE Query Selection was: {[abs(vbm[k] - rewards[k]) for k in range(len(rewards))]}")
    print(f"The difference from ground truth from the Random Query Selection was: {[abs(mbm[k] - rewards[k]) for k in range(len(rewards))]}")
    print(f"The MAE loss at the end for VAE Query Selection was: {variational_convergence_time[-1]}")
    print(f"The MAE loss at the end for Maximum Entropy Query Selection was {mutual_convergence_time[-1]}") 
"""