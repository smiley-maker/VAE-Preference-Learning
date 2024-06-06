from src.VAE.utils.imports import *

def cluster_latent_space(model, data_loader, device):
    model.eval()

    with torch.no_grad():
        latent_representations = []
        for data in data_loader:
            data = data.to(device)
            mean, logvar = model.encode(data)
            std_mean = model.reparametrize(mean, logvar)
            latent_representations.append(std_mean.cpu().numpy())  # Move to numpy for K-Means

        # Concatenate all latent representations into a single array
        latent_array = np.concatenate(latent_representations, axis=0)

    # Clustering
    km = KMeans(n_clusters=8, random_state=0, n_init="auto").fit(latent_array)
#    centers = km.cluster_centers_
    labels = km.labels_

    # TSNE Visualization
    tsne = TSNE(n_components=3, verbose=1)
    tsne_results = tsne.fit_transform(latent_array)

    tab10 = plt.get_cmap('Pastel1')
    cluster_colors = ListedColormap(tab10(np.linspace(0, 1, len(np.unique(labels)))))
    
    # Create the scatter plot with color based on labels
    fig = px.scatter_3d(
        tsne_results, x=0, y=1, z=2,
        color=labels,
        color_discrete_sequence=cluster_colors(np.arange(len(np.unique(labels))))  # Map labels to colors
    )

    fig.update_layout(title='VAE Latent Space with TSNE',
                        width=600,
                        height=600)

    fig.show()

    return labels