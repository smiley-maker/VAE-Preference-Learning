from src.VAE.utils.imports import *

if __name__ == "__main__":
    costmap = np.load("../pipelines/traversability_costmap.npy")
    costmap = costmap.reshape((6400, 6400))
    plt.imshow(costmap, cmap="Paired")
    plt.show()