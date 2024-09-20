from src.VAE.utils.imports import *

def cluster_latent_space(model, data_loader, device, mapping : dict = None, unique_terrains : int = None):
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
    # I think maybe we could infer the number of clusters based on the unique
    # terrains represented in the initial trajectory set. 
    if unique_terrains:
        km = KMeans(n_clusters=unique_terrains, random_state=0, n_init="auto").fit(latent_array)
    else:
        km = KMeans(n_clusters=len(mapping.keys()), random_state=0, n_init="auto").fit(latent_array)
#    centers = km.cluster_centers_
    labels = km.labels_


    return labels



def visualize_latent_space(model, data_loader, device, mapping, labels):
    print("VISUALIZING")
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

    print(len(latent_array))
    # TSNE Visualization
    tsne = TSNE(n_components=3, verbose=1, perplexity=len(labels)/4)
    tsne_results = tsne.fit_transform(latent_array)

    num_labels = len(np.unique(labels))

    cluster_labels = [mapping[i] for i in labels]
    print(labels)
    print(cluster_labels)

    tab10 = plt.get_cmap('Pastel2')
    colors = tab10(np.linspace(0, 1, num_labels))
    color_scale = [[i / (num_labels - 1), f'rgb({r * 255}, {g * 255}, {b * 255})'] for i, (r, g, b, a) in enumerate(colors)]

    # Create the scatter plot with color based on labels
    fig = px.scatter_3d(
        tsne_results, x=0, y=1, z=2,
        color=cluster_labels,
        color_continuous_scale=color_scale,
    )

    # Update the colorbar to show terrain type names
    fig.update_layout(coloraxis_colorbar=dict(
        title="Terrain Type",
        tickvals=np.arange(num_labels),
        ticktext=np.unique(cluster_labels)
    ))

    fig.update_layout(title='VAE Latent Space with TSNE',
                        width=600,
                        height=600)

    fig.show()