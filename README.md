# VAE-based Terrain Preference-Learning Over Pairwise Trajectory Queries


Terrain preference learning from trajectory queries allows complex reward structures to be obtained for robot navigation without the need for manual specification. However, traditional offline preference learning approaches suffer from ambiguous trajectory pairs stemming from inadequacy in the initial dataset, which causes longer learning times and may lead to less accurate results. Several approaches have been introduced to tackle this common problem including creating preference learning models robust to volatility in weights from ambiguous choices, enhancing the query selection process towards mitigating dubious trajectory choices, and modifying the original dataset with highly variant samples. Inspired by recent work in the application of deep learning towards improving query selection, this paper introduces a joint dataset and query optimization procedure utilizing variational autoencoders. Our efforts leverage both the encoder and decoder models to identify underrepresented terrain types and supplement the trajectory set with targeted samples created using the decoder. We jointly optimize a clustered latent space towards creating balanced clusters that can be used to obtain diverse trajectory pairs.  

Currently in submission for HRI 2025. Presented at the InterAI workshop, and in a late breaking report session, at RO-MAN 2024. See the link below for the workshop paper. 

https://drive.google.com/file/d/1m5Fjig8bQNhsfFhzz2JFDxARZe4t-9dd/view


## Initial Results

<img src="./src/VAE/results/medium algorithm convergence comparison 75 queries.png" alt="Convergence comparison for several algorithms over 75 queries; gives histogram of final convergence values for each algorithm, showing ours performed significantly better over the others" />

<img src="./src/VAE/results/medium convergence algorithm comparison 75 queries.png" alt="Convergence comparison for several algorithms over 75 queries; gives the convergence over time for all algorithms" />

<img src="./src/VAE/results/fixed/32_initial_trajectories_72_after_200_length_11200_epochs_75_queries.png" alt="75 Queries Mutual Information vs Our Approach" />


<img src="./src/VAE/results/alignment/10000_epochs_150_length_33_trajectory_set.png" alt="Alignment Comparisons Over 3 Trials" />

<img src="./src/VAE/results/convergence/11200_epochs_200_length_33_trajectory_set.png" alt="Convergence Comparison Over 11 Trials" />
