# VAE-based Terrain Preference-Learning Over Pairwise Trajectory Queries

## Introduction

Terrain preference learning from trajectory queries allows complex reward structures to be obtained for robot navigation without the need for manual specification. However, traditional offline preference learning approaches suffer from ambiguous trajectory pairs stemming from inadequacy in the initial dataset, which causes longer learning times and may lead to less accurate results. Several approaches have been introduced to tackle this common problem including creating preference learning models robust to volatility in weights from ambiguous choices, enhancing the query selection process towards mitigating dubious trajectory choices, and modifying the original dataset with highly variant samples. Inspired by recent work in the application of deep learning towards improving query selection, this paper introduces a joint dataset and query optimization procedure utilizing variational autoencoders. Our efforts leverage both the encoder and decoder models to identify underrepresented terrain types and supplement the trajectory set with targeted samples created using the decoder. We jointly optimize a clustered latent space towards creating balanced clusters that can be used to obtain diverse trajectory pairs.  

Currently in submission for HRI 2025. Presented at the InterAI workshop, and in a late breaking report session, at RO-MAN 2024. See the link below for the workshop paper. 

https://drive.google.com/file/d/1m5Fjig8bQNhsfFhzz2JFDxARZe4t-9dd/view

## Background

<img src="./src/VAE/results/vae infographic.png" alt="Infographic describing what a variational autoencoder is" />

<img src="./src/VAE/results/robot navigation infographic.png" alt="Infographic describing how robots can navigate autonomously, including through LiDAR vision, cost-maps, and a combination" />

To represent the terrain costs for robot navigation, we utilized costmaps, which are grid-based representations of the environment where each cell stores a cost value associated with traversing that area. Costmaps are typically generated from LIDAR data and, in our approach, we learned the terrain costs represented in the costmaps using human preferences.

<img src="./src/VAE/results/pbirl infographic.png" alt="Infographic describing preference-based inverse reinforcement learning" />

I used this approach to infer terrain weights based on user preferences over robot trajectory/path samples by creating a feature function that simply reports the number of occurrences of each terrain in the path type divided by the total length. 

## Results

Using a simulated oracle to answer preference queries, I rigorously tested my approach against state-of-the-art baseline query acquisition functions. I primarily focused a suite of experiments comparing our method with mutual information based queries, which prioritizes the potential ability of a selected query to resolve uncertainty in the belief distribution, because our method is built on top of this. 

First I conducted three fixed length experiments, where I tracked the learned weights over time for a certain number of queries. I did this with 25, 50, and 75 queries, respectively. The figure above shows the learned weights for both methods. Mutual information querying resulted in the less represented terrain types (those that did not frequently appear in the trajectories shown to the user) converged to approximately 0, indicating that they were not distinguishable from one another. On the other hand, our method showed that these terrain weights actually converged to a representative value, and to the correct ground truth order. The figure below also shows the alignment and convergence of the learned weights when compared to the ground truth costs over time. The convergence plot demonstrates that our approach got closer to the ground truth faster than the mutual information querying approach. Similarly, the alignment plot shows that our approach not only became fully aligned much faster than the comparative method,, but also remained aligned for longer during preference elicitation. 

<img src="./src/VAE/results/fixed/32_initial_trajectories_72_after_200_length_11200_epochs_75_queries.png" alt="75 Queries Mutual Information vs Our Approach" />

We also conducted convergence and alignment experiments where the learning continued until the mean absolute error was within 0.75 of the ground truth and where ground truth alignment was reach (0 terrain types out of order), respectively. For the first suite of experiments, we conducted 11 tests. In 8/11 tests our method showed faster convergence than PbIRL with mutual information queries, and 1 test was a tie. In other words, discounting the tie, our method converged faster 80% of the time. There were also several tests were our method converged significantly faster. We tested alignment three times, where our approach became aligned much more quickly in all runs. Overall, we show statistically significant results over mutual information based querying. 

<img src="./src/VAE/results/alignment/10000_epochs_150_length_33_trajectory_set.png" alt="Alignment Comparisons Over 3 Trials" /> 

<img src="./src/VAE/results/convergence/11200_epochs_200_length_33_trajectory_set.png" alt="Convergence Comparison Over 11 Trials" />

**Thus our approach shows statistically significant improvements over mutual information.**

Finally, we conducted an experiment comparing our method with not just mutual information, but three other query acquisition functions as well. These include volume removal, Thompson sampling, and disagreement based selection. Our algorithm showed faster convergence against all four other algorithms. 

<img src="./src/VAE/results/medium algorithm convergence comparison 75 queries.png" alt="Convergence comparison for several algorithms over 75 queries; gives histogram of final convergence values for each algorithm, showing ours performed significantly better over the others" />

<img src="./src/VAE/results/medium convergence algorithm comparison 75 queries.png" alt="Convergence comparison for several algorithms over 75 queries; gives the convergence over time for all algorithms" />
