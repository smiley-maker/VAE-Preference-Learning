#from src.VAE.vizualization_scripts.ClusteredTSNE import cluster_latent_space, visualize_latent_space
#from src.VAE.data_handlers.trajectory_segment_gatherer import TrajectoryDataset
#from src.VAE.training_scripts.train_vae import TrainVAE
from src.VAE.utils.moving_average import moving_average
#from src.VAE.architectures.vae_lstm import MemoryVAE
from src.VAE.utils.simulated_user import SimUser
from src.VAE.utils.imports import *


class TerrainLearning:
    def __init__(self,
                 datasets : TrajectorySet | list[TrajectorySet],
                 simuser : SimUser,
                 ground_truth : list[float],
                 algorithms : list[str], 
                 terrain_labels : list[str],
                 initial_weights : list[float],
                 no_queries : int | str,
                 plot_error : bool,
                 plot_conv : bool,
                 use_sim : bool,
                 clusters : list[int] | None,
                ) -> None:
        """
        This class helps to facilitate algorithm comparisons and testing, 
        and general preference learning experimentation. 
        """

        self.datasets = datasets
        self.clusters = clusters
        self.ground_truth = ground_truth
        self.initial_weights = initial_weights
        self.algorithms = algorithms
        self.no_queries = no_queries
        self.plot_error = plot_error
        self.plot_conv = plot_conv
        self.terrain_labels = terrain_labels
        self.use_sim = use_sim
        self.simuser = simuser

        self.experiments = {
            k: {
                # Experiments to track
                "rewards": {t : [] for t in self.terrain_labels},
                "alignment": [],# if self.plot_conv else None,
                "error": []# if self.plot_error else None,
            }
            for k in self.algorithms
        }

#        if self.use_sim:
#            self.simuser = SimUser(self.ground_truth, self.terrain_labels)
#
        try:
            assert len(self.datasets) == len(self.algorithms)
        except:
            print("The number of algorithms and datasets must be the same. ")
            print(f"You provided {len(algorithms)} algorithms and {len(datasets)} datasets.")
        
        try: 
            if "variational_info" in self.algorithms:
                assert len(clusters) != 0
        except:
            print("When using variational info, clustered data must be provided.")


    def run_learning_fixed(self):
        # Run for any algorithms mnetioned. 
        for idx, algorithm in enumerate(self.algorithms):
            # Create a query optimizer using the dataset. 
            query_optimizer = QueryOptimizerDiscreteTrajectorySet(self.datasets[idx])
            # Initialize the belief network with provided weights
            params = {"weights": self.initial_weights}
            # Add initial weights to the reward plots
            for j, t in enumerate(self.terrain_labels):
                self.experiments[algorithm]["rewards"][t].append(self.initial_weights[j])
                        
            estimated_rewards = [r[-1] for r in self.experiments[algorithm]["rewards"].values()]
            convergence = np.sqrt(sum([(r - g)**2 for r,g in zip(estimated_rewards, self.ground_truth)]))
            self.experiments[algorithm]["error"].append(convergence)

            # Calculate initial reward alignment
            self.experiments[algorithm]["alignment"].append(
                self.simuser.check_distribution(self.simuser.get_preferential_order({t : w for t, w in zip(self.terrain_labels, self.initial_weights)}))
            )

            # Create a user model
            user_model = SoftmaxUser(params)
            # Create the belief network
            belief = SamplingBasedBelief(user_model, [], params)
            # Create an example query to use as template
            example_query = PreferenceQuery(self.datasets[idx][:2]) # Using first two trajectories
            # Learning Loop
            jdx = 0
            print(f"Running {algorithm} for {self.no_queries} queries.")
            while jdx <= self.no_queries:
                # Store terrain weights
                for j, t in enumerate(self.terrain_labels):
                    self.experiments[algorithm]["rewards"][t].append(belief.mean["weights"][j])
                
                # Calculate convergence as mse between ground truth and current rewards
                estimated_rewards = [r[-1] for r in self.experiments[algorithm]["rewards"].values()]
                convergence = np.sqrt(sum([(r - g)**2 for r,g in zip(estimated_rewards, self.ground_truth)]))
#                convergence = np.sqrt(sum([(abs(r - g) / g)**2 for r,g in zip(estimated_rewards, self.ground_truth)])) * 100
                self.experiments[algorithm]["error"].append(convergence)

                # Calculate reward alignment
                self.experiments[algorithm]["alignment"].append(
                    self.simuser.check_distribution(self.simuser.get_preferential_order({t : w for t, w in zip(self.terrain_labels, belief.mean["weights"])}))
                )

                print(f"Error: {self.experiments[algorithm]['error'][-1]}")                


                jdx += 1
                # Optimization
                queries, _ = query_optimizer.optimize(algorithm, belief, example_query, clusters=self.clusters)
                # Get responses
                if self.use_sim:
                    responses = self.simuser.respond(queries[0])
                    # Update belief distribution
                    belief.update(Preference(queries[0], responses[0]))


    def run_learning_convergence(self, threshold):
        # Run for any algorithms mnetioned. 
        convergence_times = []
        for idx, algorithm in enumerate(self.algorithms):
            print(f"Running {algorithm} until convergence to within {threshold*100}%")
            print("#"*100)
            # Create a query optimizer using the dataset. 
            query_optimizer = QueryOptimizerDiscreteTrajectorySet(self.datasets[idx])
            # Initialize the belief network with provided weights
            params = {"weights": self.initial_weights}
            # Add initial weights to the reward plots
            for j, t in enumerate(self.terrain_labels):
                self.experiments[algorithm]["rewards"][t].append(self.initial_weights[j])
                        
            estimated_rewards = [r[-1] for r in self.experiments[algorithm]["rewards"].values()]
            convergence = np.sqrt(sum([(r - g)**2 for r,g in zip(estimated_rewards, self.ground_truth)]))
            self.experiments[algorithm]["error"].append(convergence)

            # Create a user model
            user_model = SoftmaxUser(params)
            # Create the belief network
            belief = SamplingBasedBelief(user_model, [], params)
            # Create an example query to use as template
            example_query = PreferenceQuery(self.datasets[idx][:2]) # Using first two trajectories
            # Learning Loop
            jdx = 0
            while self.experiments[algorithm]["error"][-1] > threshold and jdx <= self.no_queries:
                # Store terrain weights
                for j, t in enumerate(self.terrain_labels):
                    self.experiments[algorithm]["rewards"][t].append(belief.mean["weights"][j])
                
                # Calculate convergence as mse between ground truth and current rewards
                estimated_rewards = [r[-1] for r in self.experiments[algorithm]["rewards"].values()]
                convergence = np.sqrt(sum([(r - g)**2 for r,g in zip(estimated_rewards, self.ground_truth)]))
#                convergence = np.sqrt(sum([(abs(r - g) / g)**2 for r,g in zip(estimated_rewards, self.ground_truth)])) * 100
                self.experiments[algorithm]["error"].append(convergence)

                # Calculate reward alignment
                self.experiments[algorithm]["alignment"].append(
                    self.simuser.check_distribution(self.simuser.get_preferential_order({t : w for t, w in zip(self.terrain_labels, belief.mean["weights"])}))
                )

                print(f"Error: {self.experiments[algorithm]['error'][-1]}")
                
                jdx += 1
                # Optimization
                queries, _ = query_optimizer.optimize(algorithm, belief, example_query, clusters=self.clusters)
                # Get responses
                if self.use_sim:
                    responses = self.simuser.respond(queries[0])
                    # Update belief distribution
                    belief.update(Preference(queries[0], responses[0]))
            
            print(f"The {algorithm} model converged in {jdx} queries.")
            convergence_times.append((algorithm, jdx))
        
        return convergence_times
    
    def run_learning_alignment(self):
        # Run for any algorithms mnetioned. 
        convergence_times = []
        for idx, algorithm in enumerate(self.algorithms):
            print(f"Running {algorithm} until all terrains are aligned to ground truth ordering.")
            print("#"*100)
            # Create a query optimizer using the dataset. 
            query_optimizer = QueryOptimizerDiscreteTrajectorySet(self.datasets[idx])
            # Initialize the belief network with provided weights
            params = {"weights": self.initial_weights}
            # Add initial weights to the reward plots
            for j, t in enumerate(self.terrain_labels):
                self.experiments[algorithm]["rewards"][t].append(self.initial_weights[j])
                        
            estimated_rewards = [r[-1] for r in self.experiments[algorithm]["rewards"].values()]
            convergence = np.sqrt(sum([(r - g)**2 for r,g in zip(estimated_rewards, self.ground_truth)]))
            self.experiments[algorithm]["error"].append(convergence)

            # Create a user model
            user_model = SoftmaxUser(params)
            # Create the belief network
            belief = SamplingBasedBelief(user_model, [], params)
            # Create an example query to use as template
            example_query = PreferenceQuery(self.datasets[idx][:2]) # Using first two trajectories
            
            # Calculate initial alignment
            self.experiments[algorithm]["alignment"].append(
                self.simuser.check_distribution(self.simuser.get_preferential_order({t : w for t, w in zip(self.terrain_labels, belief.mean["weights"])}))
            )

            # Learning Loop
            jdx = 0
            while self.experiments[algorithm]["alignment"][-1] != 0 and jdx <= self.no_queries:
                # Store terrain weights
                for j, t in enumerate(self.terrain_labels):
                    self.experiments[algorithm]["rewards"][t].append(belief.mean["weights"][j])
                
                # Calculate convergence as mse between ground truth and current rewards
                estimated_rewards = [r[-1] for r in self.experiments[algorithm]["rewards"].values()]
                convergence = np.sqrt(sum([(r - g)**2 for r,g in zip(estimated_rewards, self.ground_truth)]))
#                convergence = np.sqrt(sum([(abs(r - g) / g)**2 for r,g in zip(estimated_rewards, self.ground_truth)])) * 100
                self.experiments[algorithm]["error"].append(convergence)

                # Calculate reward alignment
                self.experiments[algorithm]["alignment"].append(
                    self.simuser.check_distribution(self.simuser.get_preferential_order({t : w for t, w in zip(self.terrain_labels, belief.mean["weights"])}))
                )

                print(f"Alignment: {self.experiments[algorithm]['alignment'][-1]}")
                
                jdx += 1
                # Optimization
                queries, _ = query_optimizer.optimize(algorithm, belief, example_query, clusters=self.clusters)
                # Get responses
                if self.use_sim:
                    responses = self.simuser.respond(queries[0])
                    # Update belief distribution
                    belief.update(Preference(queries[0], responses[0]))
            
            print(f"The {algorithm} model converged in {jdx} queries.")
            convergence_times.append((algorithm, jdx))
        
        return convergence_times
    

    def plot_results(self, algorithm : str, fig_file : str):
        # Plots the results for a given algorithm

        # Creating a color palette
        # Water, Rock, Sand, Trees, Sidewalk
        colors = ["#3b6994", "#594010", "#b8a225", "#76b08e", "#9b88bf"]
        color_dict = {
            t : c for t,c in zip(self.terrain_labels, colors)
        }

        # Create figure
        fig, axes = plt.subplots(2, 1, figsize=(12, 9))

        fig.subplots_adjust(wspace=0.2, hspace=0.5)

        # Define font sizes
        SIZE_DEFAULT = 12
        SIZE_LARGE = 16
        SIZE_TITLE = 24
        plt.rc("font", family="sans")
        plt.rc("font", weight="normal")
        plt.rc("font", size=SIZE_DEFAULT)
        plt.rc("axes", titlesize=SIZE_TITLE)
        plt.rc("axes", labelsize=SIZE_LARGE)
        plt.rc("xtick", labelsize=SIZE_DEFAULT)
        plt.rc("ytick", labelsize=SIZE_DEFAULT)
        
        # Plot moving average of terrain weights
        for terrain_type in self.terrain_labels:
            avg_weights = moving_average(self.experiments[algorithm]["rewards"][terrain_type])
            axes[0].plot(avg_weights, label=terrain_type, color=color_dict[terrain_type], linewidth=2)
        
        # Removing the box
        axes[0].spines["right"].set_visible(False)
        axes[0].spines["left"].set_visible(False)
        axes[0].spines["top"].set_visible(False)
        axes[0].yaxis.set_ticks_position("left")
        axes[0].xaxis.set_ticks_position("bottom")
        axes[0].spines["left"].set_bounds(-1, 1)

        axes[0].set_title("Moving Average of Terrain Weights")
        axes[0].set_xlabel("Query Number")
        axes[0].set_ylabel("Weight")
        axes[0].legend()

        
        # Plot alignment
        axes[1].plot(self.experiments[algorithm]["error"], color="#3c3e4a")
        axes[1].spines["right"].set_visible(False)
        axes[1].spines["left"].set_visible(False)
        axes[1].spines["top"].set_visible(False)
        axes[1].yaxis.set_ticks_position("left")
        axes[1].xaxis.set_ticks_position("bottom")
        axes[1].set_title("Convergence to Ground Truth Terrain Ordering")
        axes[1].set_xlabel("Query Number")
        axes[1].set_ylabel("Terrain Weight")

        # Plot convergence error bars
        print(self.experiments[algorithm]["error"][-1])
        print(self.terrain_labels)
#        axes[1][0].errorbar(self.terrain_labels, self.experiments[algorithm]["error"][-1])
#        axes[1][0].set_title("Ground Truth vs. Predicted Terrain Weights")
#        axes[1][0].set_ylabel("Final Reward Error")
#        axes[1][0].set_xlabel("Terrain Types")

        plt.savefig(fig_file)

        plt.close() # Not sure if we need this here. 

        print("#################################\n#################################\n#################################")
        print("RESULTS")
        
        final_weights = [self.experiments[algorithm]["rewards"][t][-1] for t in self.terrain_labels]
        print(f"The final weights for {algorithm} were: {final_weights}")
        final_convergence = self.experiments[algorithm]["error"][-1]
        print(f"The difference from the ground truth weights was: {final_convergence}")
        final_alignment = self.experiments[algorithm]["alignment"][-1]
        print(f"The alignment between the ground truth and learned weights was: {final_alignment}")
    

    def plot_together(self, fig_file : str, error_file : str):
        # Creating a color palette
        # Water, Sand, Rock, Trees, Sidewalk
#        colors = ["#3b6994", "#b8a225", "#594010", "#76b08e", "#9b88bf"]
        colors = ["#2C0E37", "#FFBC85", "#87313F", "#074F57", "#BFCDD9"]
        #colors = ["#0E103D", "#6A3E37", "#455E53", "#BC8034", "#A1A5A1", "#6C5A49", "#C8AD55", "#9BC59D", "#363537", "#FAC9B8"]
        print(self.terrain_labels)
        color_dict = {
            t : c for t,c in zip(self.terrain_labels, colors)
        }

        alignment_colors = ["#FAC05E", "#5448C8"]

        titles = {
            "variational_info" : "VAE Querying Terrain Weights",
            "mutual_information" : "Mutual Information Terrain Weights",
            "volume_removal" : "Volume Removal Terrain Weights"
        }

        # Create figure
        # The number of horizontal plots should be the same as the number of
        # algorithms in the experiments dictionary. The number of vertical plots
        # will be 2 (one row for the weights and one for the convergence and alignment)
        fig, axes = plt.subplots(2, len(self.experiments.keys()), figsize=(16, 9))

        fig.subplots_adjust(wspace=0.75, hspace=0.75)

        # Define font sizes
        SIZE_DEFAULT = 12
        SIZE_LARGE = 14
        SIZE_TITLE = 16
        plt.rc("font", family="sans")
        plt.rc("font", weight="normal")
        plt.rc("font", size=SIZE_DEFAULT)
        plt.rc("axes", titlesize=SIZE_TITLE)
        plt.rc("axes", labelsize=SIZE_LARGE)
        plt.rc("xtick", labelsize=SIZE_DEFAULT)
        plt.rc("ytick", labelsize=SIZE_DEFAULT)

        # Plot terrain weights:
        for idx, algorithm in enumerate(self.experiments):
            final_weights = {t : 0 for t in self.terrain_labels}
            for terrain_type in self.terrain_labels:
                avg_weights = moving_average(self.experiments[algorithm]["rewards"][terrain_type])
                final_weights[terrain_type] = self.experiments[algorithm]["rewards"][terrain_type][-1]
                axes[0][idx].plot(avg_weights, label=terrain_type, color=color_dict[terrain_type], linewidth=2)
                # axes[0][idx].text(
                #     len(avg_weights) * 1.01,
                #     idx * (16/5),
                #     terrain_type,
                #     color=color_dict[terrain_type],
                #     fontweight="bold",
                #     horizontalalignment="left",
                #     verticalalignment="center"
                # )

            axes[0][idx].spines["right"].set_visible(False)
            axes[0][idx].spines["left"].set_visible(False)
            axes[0][idx].spines["top"].set_visible(False)
            axes[0][idx].yaxis.set_ticks_position("left")
            axes[0][idx].xaxis.set_ticks_position("bottom")
            axes[0][idx].spines["left"].set_bounds(-1, 1)

            axes[0][idx].set_title(titles[algorithm])
            axes[0][idx].set_xlabel("Query Number")
            axes[0][idx].set_ylabel("Weight")

            #sorted_terrain_labels = sorted(final_weights, key=lambda x: final_weights[x], reverse=True)
            # Display the terrain labels in order of their final weights
            # for i, terrain_type in enumerate(sorted_terrain_labels):
            #     axes[0][idx].text(
            #         len(self.experiments[algorithm]["rewards"]["Water"]) * 1.01,
            #         i * 2,#(16/5),
            #         terrain_type,
            #         color=color_dict[terrain_type],
            #         fontweight="bold",
            #         horizontalalignment="left",
            #         verticalalignment="center"
            #     )
            axes[0][idx].legend()


            # Plot alignment
            axes[1][0].plot(self.experiments[algorithm]["alignment"], color=alignment_colors[idx], linewidth=2, label=algorithm)
            axes[1][0].spines["right"].set_visible(False)
            axes[1][0].spines["left"].set_visible(False)
            axes[1][0].spines["top"].set_visible(False)
            axes[1][0].yaxis.set_ticks_position("left")
            axes[1][0].xaxis.set_ticks_position("bottom")
            axes[1][0].spines["left"].set_bounds(0, 5)

            # Plot convergence
            axes[1][1].plot(self.experiments[algorithm]["error"], color=alignment_colors[idx], linewidth=2, label=algorithm)
            axes[1][1].spines["right"].set_visible(False)
            axes[1][1].spines["left"].set_visible(False)
            axes[1][1].spines["top"].set_visible(False)
            axes[1][1].yaxis.set_ticks_position("left")
            axes[1][1].xaxis.set_ticks_position("bottom")

        
        axes[1][0].set_title("Alignment to Ground Truth Weights")
        axes[1][0].set_xlabel("Query Number")
        axes[1][0].set_ylabel("# of Terrains Out of Order")
        axes[1][0].legend()

        axes[1][1].set_title("Convergence to Ground Truth Weights")
        axes[1][1].set_xlabel("Query Number")
        axes[1][1].set_ylabel("MSE")
        axes[1][1].legend()

        plt.savefig(fig_file)

        plt.close() # Not sure if we need this here. 


#        final_error = [abs(t[-1] - g) for t, g in zip(self.experiments["variational_info"]["rewards"].values(), self.ground_truth)]
#        print(f"VAE Final Errors: {final_error}")
        # axes[0].errorbar(np.arange(len(self.terrain_labels)), [t[-1] for t in self.experiments["variational_info"]["rewards"]], yerr=final_error, fmt="o", capsize=5, label="Reward Error for VAE-Based Querying", color=alignment_colors[1])        
        # axes[0].set_title('Ground Truth vs. Predicted Terrain Weights for VAE Querying')
        # axes[0].set_ylabel("Reward Convergence")
        # axes[0].set_xlabel("Terrain Types")
        # axes[0].set_xticks(np.arange(len(self.terrain_labels)), self.terrain_labels)

#        final_error = [abs(t[-1] - g) for t, g in zip(self.experiments["mutual_information"]["rewards"].values(), self.ground_truth)]
#        print(f"Mutual Information Final Error: {final_error}")
        # axes[1].errorbar(np.arange(len(self.terrain_labels)), [t[-1] for t in self.experiments["mutual_information"]["rewards"]], yerr=final_error, fmt="o", capsize=5, label="Reward Error for Mutual Information Querying", color=alignment_colors[0])        
        # axes[1].set_title('Ground Truth vs. Predicted Terrain Weights for Mutual Information Querying')
        # axes[1].set_ylabel("Reward Convergence")
        # axes[1].set_xlabel("Terrain Types")
        # axes[1].set_xticks(np.arange(len(self.terrain_labels)), self.terrain_labels)

#        plt.savefig(error_file)

#        plt.close()


        print("#################################\n#################################\n#################################")
        print("RESULTS")
        
        for algorithm in self.experiments.keys():
            final_weights = [self.experiments[algorithm]["rewards"][t][-1] for t in self.terrain_labels]
            print(f"The final weights for {algorithm} were: {final_weights}")
            final_convergence = self.experiments[algorithm]["error"][-1]
            print(f"The difference from the ground truth weights was: {final_convergence}")
            final_alignment = self.experiments[algorithm]["alignment"][-1]
            print(f"The alignment between the ground truth and learned weights was: {final_alignment}")
            final_error = [abs(t[-1] - g) for t, g in zip(self.experiments[algorithm]["rewards"].values(), self.ground_truth)]
            print(f"The final errors from the ground truth rewards were: {final_error}")
            print("#-#-#"*20)
