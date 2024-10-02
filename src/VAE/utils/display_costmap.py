import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap

# Define the terrain types and corresponding colors
terrain_types = ["Water", "Trees", "Rock", "Sand", "Sidewalk"]
colors = ["#5D8CB1", "#5C8441", "#766440", "#D0C5B4", "#8897A0"]

if __name__ == "__main__":
    costmap = np.load("../pipelines/semantic_map_reconfigured.npy")
    print("Loaded Costmap")
    print(np.unique(costmap))
    costmap = costmap.reshape((6400, 6400))
    costs = {
        t : v for t,v in zip(terrain_types, [0, 25, 50, 75, 100])
    }
    print(costs)

    costmap = np.vectorize(costs.get)(costmap)
    print(np.unique(costmap))

    # Create a custom colormap
    cmap = ListedColormap(colors)
    # Plot the costmap with the custom colormap
    plt.imshow(costmap, cmap=cmap)

    # Create legend patches and labels
    patches = [ Patch(color=color) for color in colors ]
    labels = terrain_types

    # Add legend with labels
    plt.legend(patches, labels)
#    plt.colorbar()

    plt.xticks([])
    plt.yticks([])

    plt.show()