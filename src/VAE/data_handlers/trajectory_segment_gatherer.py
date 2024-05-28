"""
This class will gather trajectory samples/segments from a given
2D costmap. All segments will be of the same length L and will
be from any location in the map. 

It is also verified that the trajectory segments don't have the
same start and end positions (essentially the same trajectory),
but I might consider refining that to the same start or same end
positon such that we don't allow slightly different trajectory 
samples. 

@author: Jordan Sinclair
"""

from src.VAE.utils.imports import *

class TrajectoryDataset(Dataset):
    def __init__(
            self,
            data : np.ndarray, # Costmap
            num_samples : int, # Number of segments
            segment_length : int, # Length of each segment
            terrain_types : list[str] # Terrain types in costmap
    ) -> None:
        super().__init__()

        self.data = data 
        self.num_samples = num_samples
        self.segment_length = segment_length
        self.terrain_types = terrain_types
        self.trajectories = []
        self.get_trajectories()

        
    def sample_trajectory(self) -> list[str]:
        trajectory = []

        # Gets a random starting location. The last starting value is 
        # the segment length away from the end of the cost map. 
        random_start_x = random.randrange(0, self.data.shape[0] - self.segment_length)
        random_start_y = random.randrange(0, self.data.shape[1] - self.segment_length)
        cell = self.data[random_start_x][random_start_y]

        trajectory.append(cell)
        # Now we want to add as many connected cells as needed. 
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
#                    trajectory.append([0]*(terrain-1) + [1] + [0]*(len(self.terrain_types) - (terrain + 1)))
        
        # Pytorch requires tensors rather than lists:
#        trajectory = torch.tensor(trajectory, dtype=torch.float)
        
        return trajectory

    
    def get_trajectories(self):
        i = 1        
        while i < self.num_samples:
            trajectory = self.sample_trajectory()
            trajectory = self.encode_trajectory(trajectory)
            if trajectory not in self.trajectories:
                self.trajectories.append(trajectory)
                i = i + 1
        
        trajectories_mod = torch.stack([torch.tensor(t, dtype=torch.float) for t in self.trajectories])
        self.trajectories = trajectories_mod
        print(self.trajectories.shape)
    
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




class SegmentCollection(Dataset):
    def __init__(self, data : np.ndarray, num_samples : int, segment_length : int, terrain_types : list[str]) -> None:
        # Data is a 2D costmap that we will select segments from. 
        super().__init__()
        self.data = data
        self.num_samples = num_samples
        self.segment_length = segment_length
        self.terrain_types = terrain_types
        self.trajectories = []
        self.trajectories = self.get_trajectories()
        self.segments = self.encode_terrains()


    def sample_data(self) -> list[int]: 
        # Collects a single segment
        # A list of costs associated with the path cells.  

        segment = []

        # Gets a random starting cell.
        random_start_x = random.randrange(1, self.data.shape[0]-self.segment_length)
        random_start_y = random.randrange(1, self.data.shape[1]-self.segment_length) 
        random_start = self.data[random_start_x][random_start_y]
        segment.append(random_start)

        # Now we want to add as many connected cells as needed. 
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

            segment.append(next_cell[0])

            random_start = next_cell
            random_start_x = next_cell[1]
            random_start_y = next_cell[2]

            return segment        
    
    def get_trajectories(self):
        """
        Returns a trajectory segment set with self.num_samples unique sequences
        of length self.segment_length. 
        """
        i = 1
        trajectories = []
        while i < self.num_samples:
            trajectory = self.sample_data()
            trajectory = self.encode_terrains()
            if trajectory not in trajectories:
                trajectories.append(trajectory)
                i = i + 1

        return trajectories # unique trajectories. 
    
    def encode_terrains(self):
        # self.terrains is a list of terrain types
        # to encode using one hot encoding.

        # Each class of terrain gets a one hot encoding.
        # grass -> [1, 0, 0, 0]
        # trees -> [0, 1, 0, 0]
        # ...

        segments = [] 

        for i in range(len(self.trajectories)):
            for j in range(len(self.trajectories[i])):
                segment = []
                for k in range(len(self.terrain_types)):
                    if self.terrain_types[k] == self.trajectories[i][j]:
                        segment.append([0]*(j-1) + [1] + [0]*(len(self.terrain_types) - (j + 1)))
                
                segments.append(segment)

        return segments
    
    def __getitem__(self, idx : int):
        try:
            assert idx <= len(self.trajectories)
            return self.trajectories[idx]
        except: 
            print(f"Index out of range for trajectory set of length {len(self.trajectories)}.")
            return
    
    def __len__(self):
        return len(self.trajectories)


class Segment(Dataset):
    def __init__(self, terrains : np.ndarray, terrain_types : list[str]) -> None:
        # Segment is a trajectory (a list of costs/terrain types)
        # One thing to note is that it's really going to have to be
        # the terrain type, not a cost, because the cost will change. 
        # Maybe either labels or one hot encoded vectors. 
        super().__init__()
        self.terrains = terrains
        self.terrain_types = terrain_types
        self.segment = self.encode_terrains()
    
    def __getitem__(self):
        return self.segment
    
    def __visualize__(self):
        # How to show a trajectory to user for querying? 
        #plt.plot(self.segment)  # Won't work with one hot encoding. Maybe display cells in costmap

        # categorical data bar plot
#        plt.bar(self.terrains)
        # Count the frequency of each terrain type
        terrain_counts = {
            k : 0 for k in self.terrain_types
        }
        for terrain in self.terrains:
            if terrain in terrain_counts:
                terrain_counts[terrain] += 1
            else:
                terrain_counts[terrain] = 1

        # Extracting terrain types and their counts
        terrain_freq = list(terrain_counts.values())

        # Create the count plot
        print(self.terrain_types)
        print(terrain_freq)
        plt.bar(self.terrain_types, terrain_freq)

        plt.title("Terrain Counts in Trajectory")
        plt.xlabel("Terrain Type")
        plt.ylabel("Count")

        plt.show()

        plt.close()
    
    def encode_terrains(self):
        # self.terrains is a list of terrain types
        # to encode using one hot encoding.

        # Each class of terrain gets a one hot encoding.
        # grass -> [1, 0, 0, 0]
        # trees -> [0, 1, 0, 0]
        # ...

        segment = []

        for i in range(len(self.terrains)):
            for j in range(len(self.terrain_types)):
                if self.terrain_types[j] == self.terrains[i]:
                    segment.append([0]*(j-1) + [1] + [0]*(len(self.terrain_types) - (j + 1))) # One hot encoding ?
        
        return segment
    
    def __len__(self):
        return len(self.segment)
    


if __name__ == "__main__":
    categories = ["Grass", "Road", "Sidewalk", "Water", "Trees"]

    grid_size = (120, 120)

    random_array = np.random.randint(0, len(categories), size=grid_size)
    data = np.vectorize(lambda x: categories[x])(random_array)

    collection = SegmentCollection(data, 10, 20, categories)
    collection[0].__visualize__()