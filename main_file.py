import pandas as pd
import os
from PCA import PCA
import matplotlib.pyplot as plt
import io
import numpy as np
import seaborn as sns
from Lloyds import lloyds
import io
from Helpers import save_figure, plot_clusters, plot_vornoi, balanced_subsample

# setting the plots style
sns.set_style("whitegrid")

# Loading the MNIST dataset into pandas dataframe
df_train = pd.read_parquet(os.path.join(os.getcwd(), "mnist_parquet", "train-00000-of-00001.parquet"))
df_test = pd.read_parquet(os.path.join(os.getcwd(), "mnist_parquet", "test-00000-of-00001.parquet"))


# Since the images are stored in bytes format, converting the bytes format into matrix format
## Also flattening the 28*28 image matrix to 784 sized vector

df_train["image_matrix"] = df_train["image"].apply(lambda x: plt.imread(io.BytesIO(x["bytes"])).ravel())
df_test["image_matrix"] = df_test["image"].apply(lambda x: plt.imread(io.BytesIO(x["bytes"])).ravel())

# Using the function balanced_subsample to sample 1000 images from the dataset with 100 samples for each digit
df_train = balanced_subsample(df_train.loc[:, ["label", "image_matrix"]], "label", 1000)


## Visualizing some train images
inds = [2,3,4,7,9,13]

fig, ax = plt.subplots(nrows = 1, ncols=6, figsize = (10,5))

for axis, ind in zip(ax.ravel(), inds):
    axis.imshow(df_train.loc[ind, "image_matrix"].reshape(28,28), cmap = "grey")
    axis.set_title(df_train.loc[ind, "label"], size = 12)
    axis.axis("off")
plt.suptitle("Visualizing digit sample images", y = 0.7, size = 14)
plt.show()
save_img_path = os.path.join(os.getcwd(), "output_images")
save_figure(fig, "Visualizing digit sample images")

# Running PCA on the MNIST dataset
X = np.array(list(df_train["image_matrix"]))
y = np.array(list(df_train["label"]))
pca = PCA()
eigenvalues, pca_comp = pca.fit(X)

#Visualizing the principal components
pca_list = [0, 1, 2, 9, 19, 29, 99, 199, 299, 781, 782, 783]
fig, ax = plt.subplots(nrows=4, ncols=3, figsize=(10, 10))

for axis, p in zip(ax.ravel(), pca_list):
    sns.heatmap(np.abs(pca_comp[:, p].reshape(28, 28)), ax=axis)
    axis.set_title(f"Principal Component: {p+1}", size = 12)
    axis.axis("off")

plt.suptitle("Visualizing Principal Components", size = 18, y = 1)
plt.tight_layout()
plt.show()
save_figure(fig, "Visualizing Principal Components")


## Calculating the fraction of variance explained by each principal component
f_variance = eigenvalues/np.sum(eigenvalues, axis = 0)

## Finding the 95% variance explained component
f_cumsum = np.cumsum(f_variance)
x = np.argwhere(f_cumsum>0.95).ravel()[0]  ## The component upto which 95% variance is explained

## Tabulating the variance explained by components
f_variance_p = np.round(f_variance*100, 4)
variance_exp = list(zip(np.arange(1, len(f_variance_p)+1), f_variance_p))
variance_exp = [{"Principal_Component": i, "Variance": j} for i,j in variance_exp]
df_variance = pd.DataFrame(columns = ["Principal_Component", "Variance"], data = variance_exp)
df_variance = df_variance.set_index("Principal_Component")
print("% Variance explained by top 5 Principal Components")
print(df_variance.head())
print("% Variance explained by bottom 5 Principal Components")
print(df_variance.tail())

# Plotting the explained variance as a scatter plot
fig = plt.figure(figsize = (12, 6))
plt.scatter(np.arange(1, len(f_variance)+1), f_variance*100)
plt.ylabel("% Varaince explained", size = 12)
plt.xlabel("Principal Components", size = 12)
plt.title("Variance explained by each principal component", size = 14)
plt.axvline(x, label = f"95% variance explained by first {x} components", color = "red", linestyle = "--")
plt.legend(fontsize = 12)
plt.show()
save_figure(fig, "Variance explained by each principal component")


# Reconstructing the dataset using different dimensional representations
inds = [0,2,4,5,7,13] # samples to plot
X_ = X[:20, :]  # taking a subsample of X to prevent un-necessary transformations of samples

d = [1, 25, 50, 100, 131, 200, 300]  # dimensions used to reconstruct
    
for dim in d:
    fig, ax = plt.subplots(nrows = 2, ncols = len(inds), figsize = (10, 3))
    X_transformed = pca.transform(X_, dim)
    X_reconstructed = pca.reconstruct(X_transformed)
    for i, x_zip in enumerate(zip(ax.ravel(), inds*2)):
        axis, ind = x_zip
        if i < len(inds):
            axis.imshow(X[ind, :].reshape(28, 28), cmap="grey")
        else:
            axis.imshow(X_reconstructed[ind, :].reshape(28, 28), cmap="grey")
        axis.axis("off")
        
    plt.suptitle(f"Original Images vs. Reconstructed Images using {dim} dimensions ", size = 12, y = 1)
    plt.tight_layout()
    plt.show()
    save_figure(fig, f"Original Images vs. Reconstructed Images using {dim} dimensions ")


## Loading the dataset for K-means
df = pd.read_csv("cm_dataset_2.csv", header = None)
X = np.array(df.loc[: , :])

## Running k-means with random initialization 5 times
runs = 5
z_list = []
mu_list = []
errors_list = []

for i in range(runs):
    z, mu, errors = lloyds(X, k = 2)
    z_list.append(z)
    mu_list.append(mu)
    errors_list.append(errors)

# Plotting the error function vs iteration curves
fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (10, 3))
for i in range(runs):
    ax.plot(np.arange(1, len(errors_list[i])+1), errors_list[i], 
            label = f"Run {i+1}: Final Error = {np.round(errors_list[i][-1], 2)}",
           linestyle = "--", linewidth = 3)
    ax.set_title("Error Function value with each iteration for different initializations", size = 10)
    ax.legend()
    ax.set_xlabel("Iterations", size = 10)
    ax.set_ylabel("Error Value", size = 10)
    ax.tick_params(axis='both', which='major', labelsize=10)
plt.show()
save_figure(fig, "Error Function value with each iteration for different initializations")

    
# Visualizing the clusters
fig, ax = plt.subplots(nrows = 5, ncols = 1, figsize = (10, 5*runs), gridspec_kw={'hspace': 0.2})
ax = ax.ravel()[:runs]

for i, axis in zip(range(runs), ax):
    plot_clusters(X, z_list[i], mu_list[i], ax = axis)
    axis.set_title(f"K-Means run {i+1} | Final Error Value = {np.round(errors_list[i][-1], 2)}", size = 8)
    axis.set_xlabel("Feature 0", size = 8)
    axis.set_ylabel("Feature 1", size = 8)
    axis.tick_params(axis='both', which='major', labelsize=8)
plt.suptitle("Clusters obtained for each run", y = 0.9, size = 10)
# plt.tight_layout()
plt.show()
save_figure(fig, "Clusters obtained for each run")


# Running K-means for different K-values
# Taking the labels at 0, 100, 150, 200, and 400 to be the initialization points
ids = [0, 100, 150, 200, 400]
init_means = X[ids, :]
fig, ax = plt.subplots(nrows = 2, ncols = 2, figsize = (10,10))
for i, (axis, K) in enumerate(zip(ax.ravel(), [2,3,4,5])):
    z, mu, errors = lloyds(X, k = K, init = init_means[:K, :])
    plot_vornoi(X, mu, ax = axis)
    axis.set_title(f"K-Means with K = {K}", size = 10)
    axis.set_xlabel("Feature 0", size = 10)
    axis.set_ylabel("Feature 1", size = 10)
plt.suptitle("VORNOI REGIONS for different k values", y = 0.99, size = 13)
plt.tight_layout()
plt.show()
save_figure(fig, "VORNOI REGIONS for different k values")

# Plotting the clusters for the above initialization points and k values
ids = [0, 100, 150, 200, 400]
init_means = X[ids, :]
fig, ax = plt.subplots(nrows = 2, ncols = 2, figsize = (10,10))
for i, (axis, K) in enumerate(zip(ax.ravel(), [2,3,4,5])):
    z, mu, errors = lloyds(X, k = K, init = init_means[:K, :])
    plot_clusters(X, z, mu, ax = axis)
    axis.set_title(f"K-Means with K = {K}", size = 9)
    axis.set_xlabel("Feature 0", size = 9)
    axis.set_ylabel("Feature 1", size = 9)
plt.suptitle("Getting clusters for different k values", y = 1, size = 10)
plt.tight_layout()
plt.show()
save_figure(fig, "Getting clusters for different k values")