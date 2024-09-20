from src.VAE.utils.imports import *

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

