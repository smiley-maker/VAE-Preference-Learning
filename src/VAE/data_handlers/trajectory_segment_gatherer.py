"""
This class will gather trajectory samples/segments from a given
2D costmap. There are two implemented modes for this: randomly
acquiring a trajectory with a fixed length from the map, and using
a graph-based path planner with specific start and goal locations. 
For the latter method, samples are either padded using earlier
sections of the trajectory or truncated to be the correct length. 
Finally, all trajectories are encoded using integer labels to 
represent terrain classes. 

@author: Jordan Sinclair
"""

from src.VAE.utils.imports import *
import matplotlib.cm as cm
from matplotlib import colormaps
from matplotlib.patches import Rectangle
import networkx as nx
from src.VAE.vizualization_scripts.costmap_query_viz import visualize_query

class TrajectoryDataset(Dataset):
    def __init__(
            self,
            data : np.ndarray, # Costmap
            num_samples : int, # Number of segments
            segment_length : int, # Length of each segment
            terrain_types : list[str], # Terrain types in costmap
            device : torch.DeviceObjType,
            start_point : tuple[int] = None, # Trajectory starting position
            end_point : tuple[int] = None, # Trajectory ending position
            rewards : list[int] = None
    ) -> None:
        super().__init__()

        self.data = data 
        self.num_samples = num_samples
        self.segment_length = segment_length
        self.terrain_types = terrain_types
        self.start_point = start_point
        self.end_point = end_point
        self.device = device
        self.rewards = rewards
        if self.start_point and self.end_point:
            self.slice_costmap()
            print(self.smaller_costmap.shape)

        self.mapping = {
            t : i for i,t in enumerate(self.terrain_types)
        }

        self.trajectories = []
        self.trajectory_poses = []
        self.get_trajectories()
    

    def slice_costmap(self):

        # Calculate the minimum and maximum x and y coordinates for the segment
        min_x = min(self.start_point[0], self.end_point[0]) - 200
        max_x = max(self.start_point[0], self.end_point[0]) + 200
        min_y = min(self.start_point[1], self.end_point[1]) - 200
        max_y = max(self.start_point[1], self.end_point[1]) + 200

        # Ensure the coordinates stay within the bounds of the costmap
        self.min_x = max(min_x, 0)
        self.min_y = max(min_y, 0)
        self.max_x = min(max_x, self.data.shape[0] - 1)
        self.max_y = min(max_y, self.data.shape[1] - 1)

        # Slice the costmap
        self.smaller_costmap = self.data[self.min_x:self.max_x+1, self.min_y:self.max_y+1]

        # Update start_point and end_point relative to the slice
        self.start_point = (200, 200)
        self.end_point = (self.max_x - self.min_x - 200, self.max_y - self.min_y - 200)


    def plan_path(self, costs, mapping):
        """Plans a path between a start and end cell in a 2D cost-map using networkx.

        Args:
            sem_map: A 2D numpy array representing the terrains in the environment.
            costs: A list containing what values to map each terrain type to.
            start: A tuple representing the coordinates of the starting cell.
            end: A tuple representing the coordinates of the ending cell.
            device: pytorch device.

        Returns:
            A list of terrain types from the semantic map based on the path found. 
        """

        def heuristic(node1, node2):
            return cost_map[node1[0], node1[1]] + cost_map[node2[0], node2[1]]


        # Remap semantic map to costs using mapping
        cost_map = np.vectorize(mapping.get)(self.smaller_costmap) 
 
        # Create a directed graph
        G = nx.DiGraph()

        # Add nodes and edges to the graph
        for i in range(cost_map.shape[0]):
            for j in range(cost_map.shape[1]):
                G.add_node((i, j), weight=cost_map[i, j])
                if i > 0:
                    G.add_edge((i, j), (i - 1, j), weight=cost_map[i - 1, j])
                if i < cost_map.shape[0] - 1:
                    G.add_edge((i, j), (i + 1, j), weight=cost_map[i + 1, j])
                if j > 0:
                    G.add_edge((i, j), (i, j - 1), weight=cost_map[i, j - 1])
                if j < cost_map.shape[1] - 1:
                    G.add_edge((i, j), (i, j + 1), weight=cost_map[i, j + 1])

        # Find the shortest path using Dijkstra's algorithm
        path = nx.astar_path(G, self.start_point, self.end_point, heuristic=heuristic, weight='weight')

        terrains = [self.smaller_costmap[x[0]][x[1]] for x in path]

        print(f"Unique Terrains in path of length {len(terrains)}:")
        print(np.unique(terrains))
        return terrains, path
    

    def sample_trajectory(self) -> list[str]:
        trajectory = []

        # Gets a random starting location. The last starting value is 
        # the segment length away from the end of the cost map. 
        if self.start_point == None or self.end_point == None:
            random_start_x = random.randrange(0, self.data.shape[0] - self.segment_length)
            random_start_y = random.randrange(0, self.data.shape[1] - self.segment_length)
            cell = self.data[random_start_x][random_start_y]
        else:
            random_start_x = self.start_point[0]
            random_start_y = self.start_point[1]
            cell = self.data[self.start_point[0]][self.start_point[1]]
        

        trajectory.append(cell)
        # Now we want to add as many connected cells as needed. 
        if self.start_point != None and self.end_point != None:
            x = random.choices(range(0, 101), k=len(self.terrain_types))
            random.shuffle(x)
            print(f"Random Weights: {x}")
            mapping = {
                t : w for t,w in zip(self.terrain_types, x)
            }
            trajectory, path = self.plan_path(x, mapping)#random.sample(x, 5))
            self.trajectory_poses.append(path)

        else:
            for i in range(self.segment_length):
                surroundings = [
                    # We can move in the positive x direction
                    (self.data[random_start_x + 1][random_start_y], random_start_x + 1, random_start_y),
                    # We can move in the negative x direction
                    (self.data[random_start_x - 1][random_start_y], random_start_x - 1, random_start_y),
                    # We can move in the positive y direction
                    (self.data[random_start_x][random_start_y + 1], random_start_x, random_start_y + 1),
                    # We can move in the negative y direction
                    (self.data[random_start_x][random_start_y - 1], random_start_x, random_start_y - 1),
                    # We can move diagonally to the upper left
                    (self.data[random_start_x - 1][random_start_y + 1], random_start_x - 1, random_start_y + 1),
                    # We can move diagonally to the lower left
                    (self.data[random_start_x - 1][random_start_y - 1], random_start_x - 1, random_start_y - 1),
                    # We can move diagonally to the upper right
                    (self.data[random_start_x + 1][random_start_y + 1], random_start_x + 1, random_start_y + 1),
                    # We can move diagonally to the lower right
                    (self.data[random_start_x + 1][random_start_y - 1], random_start_x + 1, random_start_y - 1)
                ] # - random start because we don't want to go backwards. 

                next_cell = random.choice(surroundings)

                trajectory.append(next_cell[0])

                random_start = next_cell
                random_start_x = next_cell[1]
                random_start_y = next_cell[2]
            

        return trajectory
    
    def encode_trajectory(self, traj : list[list[int]]):
        trajectory = []

        for cell in traj:
            for terrain in range(len(self.terrain_types)):
                if self.terrain_types[terrain] == cell:
                    ohe = [0]*len(self.terrain_types)
                    ohe[terrain] = 1
                    trajectory.append(ohe)
                
        return trajectory
    
    def integer_label_encoding(self, traj : list[str]) -> list[int]:
        return [self.mapping[cell.item()] for cell in traj]
    
    def get_trajectories(self):
        i = 1
        while i < self.num_samples:
            trajectory = self.sample_trajectory()
            trajectory = self.integer_label_encoding(trajectory)
            if trajectory not in self.trajectories:
                padding_length = self.segment_length - len(trajectory) + 1
                # Pad the trajectory with itself up to the padding length. 
                # If the trajectory length is less than the padding length, we
                # repeat as long as necessary. 

                if padding_length < 0: 
                    # We need to shorten the trajectory
                    trajectory = trajectory[:self.segment_length+1]
                    padding_length = 0

                if padding_length > 0:
                    # Pad the trajectory with itself (reflection)
                    while padding_length > 0:
                        trajectory.extend(trajectory[:padding_length])
                        padding_length = self.segment_length - len(trajectory) + 1
                
                self.trajectories.append(trajectory)#torch.tensor(trajectory, dtype=torch.float32, device=self.device))
                i = i + 1


        
        trajectories_mod = torch.stack([torch.tensor(t, dtype=torch.float, device=self.device) for t in self.trajectories])
        self.trajectories = trajectories_mod
    
    def __visualize__(self, idx : int):
        # Plots the trajectory as a sequence of terrain types with different colors. 
        # Need to define a color map for one-hot encoded trajectory
        colormap_name = 'tab10'  # You can choose any colormap here
        colormap = colormaps.get_cmap(colormap_name)
        cmap = colormap(np.arange(len(self.terrain_types)) / (len(self.terrain_types) - 1))

        terrain_colors = {self.terrain_types[i]: cmap[i] for i in range(len(self.terrain_types))}
        x = 0
        y = 0
        width = 50
        height = 50
        data = self.trajectories[idx]
        fig, ax = plt.subplots()
        recs = []
        for terrain_onehot in data:
            terrain_type_index = np.argmax(terrain_onehot)
            terrain_type_name = self.terrain_types[terrain_type_index]
            terrain_color = terrain_colors[terrain_type_name]
            # Plot rectangle with position, width, height and color
            recs.append(Rectangle((x, y), width, height, color=terrain_color))
#            ax.add_patch(Rectangle((x, y), width, height, color=terrain_color))
            x += width
            y += height
        
        for r in recs:
            print(r)
            ax.add_patch(r)

        plt.show()    

    def __getitem__(self, idx : int):
        try: 
            assert idx <= len(self.trajectories)
#            print(f"Trajectory has length {len(self.trajectories[idx])}")
            return self.trajectories[idx]
        except:
            print(f"Index {idx} out of range for trajectory set of length {len(self.trajectories)}.")
            return
    
    def __len__(self):
        return len(self.trajectories)
    
    def append(self, traj):
#        self.trajectories.append(traj)
        self.trajectories.add(traj)




    
    def __histogram__(self, idx : int):
        # How to show a trajectory to user for querying? 
        #plt.plot(self.segment)  # Won't work with one hot encoding. Maybe display cells in costmap

        # categorical data bar plot
#        plt.bar(self.terrains)
        # Count the frequency of each terrain type
        terrain_counts = {
            k : 0 for k in self.terrain_types
        }
        for terrain in self.trajectories[idx]:
            if terrain in terrain_counts:
                terrain_counts[terrain] += 1
            else:
                terrain_counts[terrain] = 1

        # Extracting terrain types and their counts
        terrain_freq = list(terrain_counts.values())

        # Create the count plot
#        print(self.terrain_types)
#        print(terrain_freq)
        plt.bar(self.terrain_types, terrain_freq)

        plt.title("Terrain Counts in Trajectory")
        plt.xlabel("Terrain Type")
        plt.ylabel("Count")

        plt.show()

        plt.close()
    
