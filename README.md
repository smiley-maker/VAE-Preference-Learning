# Maximizing Query Diversity for Terrain Cost Preference Learning in Robot Navigation

**Official PyTorch implementation of the paper: "Maximizing Query Diversity for Terrain Cost Preference Learning in Robot Navigation"**
*Published in RO-MAN, 2025*

[![IEEE Xplore](https://img.shields.io/badge/IEEE_Xplore-Full_Paper-00629B?style=flat-square&logo=ieee&logoColor=white)](https://ieeexplore.ieee.org/document/11217576)
[![Conference](https://img.shields.io/badge/RO--MAN_2025-Eindhoven-pink?style=flat-square)](https://www.ro-man2025.org/)
<img src="https://img.shields.io/badge/pytorch-orange" />

## Abstract

> Effective robot navigation in real-world environments requires an understanding of terrain properties, as different terrain types impact factors such as speed, safety, and wear on the platform. Preference-based learning offers a compelling framework in which terrain costs can be inferred through simple trajectory queries to the user. However, existing query selection methods often suffer from redundant selection due to limited trajectory diversity, as well as query ambiguity, where the user must choose between trajectories with minimal distinguishable differences. These issues lead to inefficient learning and suboptimal terrain cost estimation. In this paper, we introduce a joint optimization framework that increases learning efficiency by improving both the diversity of the trajectory set and the query selection strategy. We used a variational autoencoder (VAE) to encode and group trajectories based on their terrain characteristics. Clusters were used to identify less represented terrain types so that new trajectories can be added to the corresponding cluster to ensure a balanced and representative query set. Additionally, we employ a cluster-aware query selection mechanism that prioritizes diverse trajectory pairs pulled from distinct clusters to maximize information gain. Experimental results demonstrate that our approach significantly reduces the number of queries required to converge to the ground-truth terrain cost assignment, outperforming state-of-the-art query selection techniques.

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
