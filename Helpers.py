import os
import numpy as np
import matplotlib.pyplot as plt
from Lloyds import update_z
import pandas as pd


def balanced_subsample(df, label_column, total_samples):
    """
    Sub-sample x datapoints from a DataFrame such that each unique label 
    in the label column has the same count in the sampled dataset.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    label_column (str): The name of the column containing labels.
    total_samples (int): Total number of samples desired in the subsample.

    Returns:
    pd.DataFrame: A DataFrame containing the balanced subsample.
    """
    unique_labels = df[label_column].unique()
    samples_per_label = 100

    subsampled_frames = []

    for label in unique_labels:
        label_subset = df[df[label_column] == label]
        subsampled_frames.append(label_subset.sample(samples_per_label, random_state=42))

    balanced_df = pd.concat(subsampled_frames).sample(frac=1, random_state=42).reset_index(drop=True)
    return balanced_df


save_img_path = os.path.join(os.getcwd(), "output_images")

## creating a function to save image
def save_figure(fig, filename, directory = save_img_path, dpi=300):
    """
    Save a Matplotlib figure to a specified directory with a given filename as a PNG file.
    
    Parameters:
    - fig: Matplotlib figure object to save.
    - directory: Path to the directory where the figure should be saved.
    - filename: Name of the file (without extension).
    - dpi: Resolution of the saved figure (default is 300).
    """
    if not os.path.exists(directory):
        os.makedirs(directory)  # Create the directory if it doesn't exist
    
    filepath = os.path.join(directory, f"{filename}.png")
    fig.savefig(filepath, dpi=dpi, format='png', bbox_inches='tight')


# creating a helper function to plot clusters
def plot_clusters(X, z, mu, ax=None):
    """
    Plot clusters on a given axis or create a new figure.

    Parameters:
    - X: ndarray of shape (n_samples, n_features), the data points
    - z: array-like, cluster assignments for each data point
    - mu: ndarray of shape (n_clusters, n_features), cluster centroids
    - ax: Matplotlib axis object, optional. If None, a new figure is created.

    Returns:
    - ax: The axis containing the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    scatter = ax.scatter(X[:, 0], X[:, 1], c=z, cmap='viridis', s=30, edgecolor='k')

    # Add circles for cluster means
    for cluster_id, mean in enumerate(mu):
        circle = plt.Circle(mean, radius=0.5, edgecolor='black', facecolor='red', lw=2)
        ax.add_artist(circle)  # Add the circle to the axis
        ax.text(mean[0], mean[1], "X", color='black', fontsize=12, ha='center', va='center')

    # Add labels, title, and other settings
    ax.set_title("Fig Title")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.axis("equal")  # Keep axis scales equal
    ax.grid(True, linestyle='--', alpha=0.5)
    
    return ax

# creating a helper function to plot vornoi regions
def plot_vornoi(X, mu, ax = None):
    """
    Plot vornoi regions for given mu and X
    """
    xlims = (np.min(X[:, 0])*1.5, np.max(X[:, 0])*1.5)  ## X-axis range
    ylims = (np.min(X[:, 1])*1.5, np.max(X[:, 1])*1.5)  ## Y-axis range

    lin_res = 1000   # variable for setting the linear resolution
    x_cords = np.linspace(xlims[0], xlims[1], num = lin_res)
    y_cords = np.linspace(ylims[0], ylims[1], num = lin_res)
    xx, yy = np.meshgrid(x_cords, y_cords)
    xy = np.c_[xx.ravel(), yy.ravel()]
    z = update_z(xy, mu).reshape(lin_res, lin_res)

    if ax is None:
        fig, ax = plt.subplots(figsize = (10, 8))
    ax.contourf(xx, yy, z, cmap='Set3', alpha = 0.8)
    for cluster_id, mean in enumerate(mu):
        circle = plt.Circle(mean, radius=0.5, edgecolor='black', facecolor='red', lw=2)
        ax.add_artist(circle)
        ax.text(mean[0], mean[1], "X", color='black', fontsize=12, ha='center', va='center')
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
    return ax