# VAE-based Terrain Preference-Learning Over Pairwise Trajectory Queries


Many navigational problems in robotics necessitate a well defined cost map related to the environment. Traditional techniques in creating these involve manual specification of terrain costs based on some context known to the human. However, this becomes intractable with large numbers of terrain types. Preference learning offers a unique way of tackling this type of problem by inferring a reward function through trajectory queries. However, offline preference learning suffers from the variability of the initial dataset, which limits the amount of information that can be gained from query responses and introduces a higher degree of cognitive burden on the human. In this paper, we propose to utilize recent advancements in preference learning surrounding the use of generative models, specifically variational autoencoders, as they utilize a lower dimensional latent space useful for clustering and inferring similarity or dissimilarity, to combat analogous or insufficient trajectory sets towards robotic navigation through learned terrain weights.

## Initial Results

<img src="./src/VAE/results/medium algorithm convergence comparison 75 queries.png" alt="Convergence comparison for several algorithms over 75 queries; gives histogram of final convergence values for each algorithm, showing ours performed significantly better over the others" />

<img src="./src/VAE/results/medium convergence algorithm comparison 75 queries.png" alt="Convergence comparison for several algorithms over 75 queries; gives the convergence over time for all algorithms" />

<img src="./src/VAE/results/fixed/32_initial_trajectories_72_after_200_length_11200_epochs_75_queries.png" alt="75 Queries Mutual Information vs Our Approach" />


<img src="./src/VAE/results/alignment/10000_epochs_150_length_33_trajectory_set.png" alt="Alignment Comparisons Over 3 Trials" />

<img src="./src/VAE/results/convergence/11200_epochs_200_length_33_trajectory_set.png" alt="Convergence Comparison Over 11 Trials" />
