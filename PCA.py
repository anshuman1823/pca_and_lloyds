import numpy as np

class PCA():
    """
    Implement PCA with n_comp principal components on X (n x d) dataset with n samples and d features
    """
    def __init__(self, n_comp = None):
        self.n_comp = n_comp
        self.fit_done = False

    def _center(self, X):
        """
        Centers the dataset X
        """
        return X - np.mean(X, axis = 0)
    
    def fit(self, X):
        """
        Fits the PCA on passed X dataset. Returns eigenvalues and eigenvectors.
        All eigenvalues and eigenvectors are returned if n_comp = None (default), 
        else top `n_comp` eigenvalues and eigenvectors are returned
        """
        self.X = X
        if self.n_comp is None:
            self.n_comp = self.X.shape[1]   # setting number of principal components as all components if not passed
        self.X = self._center(self.X) # centering X
        self.n = self.X.shape[0]    # no. of samples
        cov = 1/self.n * self.X.T @ self.X  # calculating the covariance matrix
        e, v = np.linalg.eigh(cov)   # returns eigenvalues and normalized eigenvectors of the covariance matrix in increasing order
        self.fit_done = True
        self.eigenvalues = e[::-1][:self.n_comp]
        self.eigenvectors = v[:, ::-1][:, :self.n_comp]
        return self.eigenvalues, self.eigenvectors
    
    def transform(self, X, n_comp):
        """
        Projects the passed X (n x d) dataset with n samples and d features on the first n_comp PCA components
        Returns the projected dataset
        """
        if not self.fit_done:
            raise ValueError("fit PCA first before transforming")
        return X @ self.eigenvectors[:, :n_comp]
    
    def reconstruct(self, X):
        """
        Reconstruct dataset transformed using PCA to original dimensions
        """
        n_comp = X.shape[1]
        return X @ self.eigenvectors[:, :n_comp].T
