Task 1: Principal Component Analysis (PCA) on MNIST Dataset
1. Objective: 
   - Perform PCA on a subset of the MNIST dataset (1000 images, 100 per digit class).
   - Visualize principal components, reconstruct images using reduced dimensions, and determine the optimal number of dimensions for downstream tasks like digit classification.

2. Steps and Results:
   - Data Preprocessing: 
     - MNIST images (28x28 pixels) were flattened into vectors of size 784.
     - A balanced subsample of 1000 images was created.
   - PCA Implementation:
     - PCA was implemented from scratch using covariance matrix, eigenvalues, and eigenvectors.
     - The first few principal components captured the most variance, while later components primarily captured noise.
   - Variance Explained:
     - The dataset lies in a subspace of 573 dimensions, with the top 131 principal components explaining 95% of the variance.
   - Reconstruction:
     - Images were reconstructed using varying numbers of principal components (e.g., 1, 25, 50, 100, 131, 200, and 300).
     - At 131 dimensions, images were clear enough for downstream tasks with minimal artifacts. At 200 dimensions, images were almost indistinguishable from the originals.
   - Optimal Dimensionality:
     - For downstream tasks like classification, 200 dimensions was recommended as it provides a good trade-off between clarity and computational efficiency.

---

Task 2: Lloyd's Algorithm for K-Means Clustering
1. Objective:
   - Implement Lloyd's algorithm for clustering a dataset of 1000 points in $$\mathbb{R}^2$$.
   - Analyze clustering results for different initializations and values of $$k$$ ($$k = \{2,3,4,5\}$$).

2. Steps and Results:
   - Implementation:
     - Lloyd's algorithm was implemented from scratch to minimize the sum of squared Euclidean distances between points and their cluster centers.
     - The algorithm iteratively updated cluster assignments and centers until convergence.
   - Random Initialization with $$k=2$$:
     - Five runs with different random initializations showed that the error function decreased with iterations but converged to slightly different final errors due to initialization sensitivity.
   - Clusters Visualization:
     - Clusters were plotted for each run. The results varied slightly depending on initialization.
   - Voronoi Regions for $$k = \{2,3,4,5\}$$:
     - Voronoi diagrams were plotted to visualize cluster boundaries for fixed initializations. These regions demonstrated linear boundaries between clusters.
   - Limitations of Lloyd's Algorithm:
     - The dataset had non-convex clusters (e.g., a circular cluster surrounded by a semi-circular arc), which Lloyd's algorithm failed to separate correctly due to its reliance on linear boundaries.
   - Recommendation:
     - Kernelized K-Means was suggested as a better alternative since it can handle non-linear separability by mapping data to higher-dimensional spaces where clusters become linearly separable.

---

Key Takeaways
1. PCA is effective for dimensionality reduction on high-dimensional datasets like MNIST. Selecting an appropriate number of principal components (e.g., 200) balances variance retention and computational efficiency.
2. Lloyd's algorithm works well for linearly separable clusters but struggles with non-convex datasets. Kernelized K-Means or other clustering methods should be used for such cases.

---

Source Code
- The report includes detailed Python implementations for PCA and Lloyd's algorithm along with helper functions for visualization (e.g., plotting clusters and Voronoi regions). These implementations demonstrate how to build these algorithms from scratch without relying on external libraries like Scikit-learn.




Instructions to run the code:

1. All the source code for all the algortihms and functions are kept in separate .py files. All these .py files should be maintained in the same directory.

2. The "main_file.py" file is the main file containing code to carry out all the tasks given in the 
assignment. Therefore, you only need to run this "main_file.py" file. It will automatically import all the other necessary functions.

3. Keep the datasets inside a folder named "datasets". This folder should
be in the same directory as that of all the other .py files.
