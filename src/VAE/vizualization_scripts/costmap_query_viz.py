from src.VAE.utils.imports import *
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap

def visualize_double_costmap_query(
        costmaps: list[np.array],
        trajectories: list[list[tuple[int, int]]],
        colors: list[str],
        mappings : list[dict]
):
    """Visualizes two costmaps with their corresponding trajectories.

    Args:
        costmaps (list[np.array]): List of costmap regions to show.
        trajectories (list[list[tuple[int, int]]]): List of trajectories.
        colors (list[str]): List of colors to display the trajectories.
    """

    print(costmaps)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    terrain_colors = {
        0: "#5D8CB1",
        25: "#5C8441",
        50: "#766440",
        75: "#D0C5B4",
        100: "#8897A0"
    }

    terrain_to_cost = {
        "Water": 0,
        "Trees": 25,
        "Rock": 50,
        "Sand": 75,
        "Sidewalk": 100
    }

    terrain_values = list(terrain_to_cost.values())
    terrain_colors = [terrain_colors[val] for val in terrain_values]

    cmap = LinearSegmentedColormap.from_list(
        "terrain_map", terrain_colors, N=len(terrain_colors)
    )

#    terrain_colors = ["#5D8CB1", "#5C8441", "#766440", "#D0C5B4", "#8897A0"]
#    cmap = ListedColormap(terrain_colors)
#    terrain_values = list(terrain_to_cost.values())  # Get numeric values in desired order
#    terrain_colors_list = [terrain_colors[val] for val in terrain_values]  # Colors in desired order

#    cmap = ListedColormap(terrain_colors_list)
#    cmap = ListedColormap([terrain_colors[terrain] for terrain in terrain_colors.keys()])
#    print(cmap.colors)
#    titles = ["Sidewalk Trajectory", "Rock Trajectory"]

    # Create legend patches and labels
    print(terrain_colors)
    patches = [ Patch(color=color) for color in terrain_colors]#.items() ]
    labels = list(terrain_to_cost.keys())


    for i in range(len(trajectories)):
        # Plot costmap
        # water_map = np.vectorize(water_categories.get)(water_map)
        c = np.vectorize(terrain_to_cost.get)(costmaps[i]) # Maps to numeric weights
        ax[i].imshow(c, cmap=cmap, origin="lower")
#        ax[i].set_title(titles[i])  # Add titles

        # Plot corresponding trajectory
        x_coords, y_coords = zip(*trajectories[i])
        ax[i].plot(x_coords, y_coords, color=colors[i], marker="o", label="Trajectory")

        ax[i].spines["right"].set_visible(False)
        ax[i].spines["left"].set_visible(False)
        ax[i].spines["bottom"].set_visible(False)
        ax[i].spines["top"].set_visible(False)
        ax[i].set_xticks([])
        ax[i].set_yticks([])

        ax[i].legend(patches, labels)




    plt.show()
    plt.close()



def visualize_query(costmap: np.array, trajectories: list[list[tuple[int, int]]], colors: list[str]):
    """This function plots two trajectories on top of a costmap plot.

    Args:
        costmap (np.array): 2D Numpy array with cost values associated with distinct terrain types.
        trajectories (list[list[tuple[int, int]]]): Two trajectories, each containing tuples to
                                                    represent x, y locations along the trajectory.
        colors (list[str]): List of colors to display the trajectories with.
    """

    # Plot the costmap
    fig, ax = plt.subplots()

    plt.imshow(costmap, cmap='gray', origin='lower')
#    plt.colorbar(label='Cost')

    # Plot each trajectory
    for trajectory, color in zip(trajectories, colors):
        x_coords, y_coords = zip(*trajectory)  # Unpack x and y coordinates
        plt.plot(x_coords, y_coords, color=color, marker='o')


    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # Display the plot
    plt.show()

