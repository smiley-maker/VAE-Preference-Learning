from src.VAE.utils.imports import *


def histogram_plot(data: list[list[tuple[str, float]]], save_path: str, plot_title: str, plot_yaxis: str, plot_xaxis: str, plot_colors: list[str] = None):
    """
    Creates a grouped bar chart showing the convergence times for different algorithms.

    Args:
        data: A list of lists, where each inner list represents a trial and contains tuples for the algorithm name and convergence time.
        save_path: The path to save the plot.
        plot_title: The title of the plot.
        plot_yaxis: The label for the y-axis.
        plot_xaxis: The label for the x-axis.
        plot_colors: A list of colors to use for each algorithm. If not provided, default colors will be used.

    Returns:
        None
    """

    n_groups = len(data)
    algorithm_values = {}

    for d in data:
        for algorithm, convergence in d:
            if algorithm not in algorithm_values.keys():
                algorithm_values[algorithm] = []
            algorithm_values[algorithm].append(convergence)


    indx = np.arange(n_groups)
    bar_width = 0.35

    fig, ax = plt.subplots()

    # Plot the bars
    for idx, (algorithm, values) in enumerate(algorithm_values.items()):
        ax.bar(indx + idx * bar_width, values, bar_width, label=algorithm, color=plot_colors[idx])

#    for idx, algorithm in enumerate(algorithm_values.keys()):
#        ax.bar(indx, algorithm_values[algorithm], bar_width, label=algorithm, color=plot_colors[idx])
#    bar1 = ax.bar(index, mutual_info_values, bar_width, label="Mutual Information", color='#5f48e2')
#    bar2 = ax.bar(index + bar_width, variational_info_values, bar_width, label="Variational Info", color='#fbbd50')

#    if not plot_colors:
#        plot_colors = ["blue", "orange", "green", "red", "purple"]  # Default colors

#    plt.figure(figsize=(10, 6))

    # Extract unique algorithm names
#    algorithm_names = set([algorithm[0] for trial in data for algorithm in trial])

    # Create a bar chart for each algorithm
#    for algorithm in algorithm_names:
#        convergence_times = [algorithm[1] for trial in data if trial[0][0] == algorithm]  # Extract convergence times for current algorithm
#        if convergence_times:  # Check if list is not empty
#            plt.bar([i + 0.2 for i in range(len(data))], convergence_times, label=algorithm, width=0.2, color=plot_colors[list(algorithm_names).index(algorithm)])

    plt.title(plot_title)
    plt.xlabel(plot_xaxis)
    plt.ylabel(plot_yaxis)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")
    plt.xticks([i + bar_width/2 for i in range(len(data))], range(1, len(data) + 1))  # Adjust x-axis labels
    plt.legend()
    plt.savefig(save_path)
    plt.show()