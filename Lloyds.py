import numpy as np
from copy import deepcopy

# implement Lloyds

def update_z(X, mu):
    """
    Finds the cluster centers nearest to the points of X out of the given cluster centers mu
    returns the assignment vector z
    the position of cluster center in mu is it's cluster indicator, 
    i.e. cluster indicator go from 0 to len(mu) - 1
    """
    z = np.ones((len(X)))*-1   ## Initializing z to -1
    for i in range(len(X)):
        z[i] = np.argmin(np.linalg.norm(X[i, :] - mu, ord = 2, axis = 1))
    return z

def mu_calculate(X, z, k):
    """
    Calculates the mu vector for a given assignment vector z and set of Datapoints X
    k: number of clusters
    """
    mu = np.zeros((k, X.shape[1]))
    for k_cur in range(k):
        X_c = X[z == k_cur]
        if len(X_c) > 0:
            mu[k_cur, :] = np.mean(X_c, axis = 0)
        else:
            mu[k_cur, :] = np.random.uniform(X.min(axis=0), X.max(axis=0), X.shape[1])  # Reinitialize empty cluster in case the cluster has no points
    return mu

def error(X, z, mu):
    """
    Function to calculate error value, which is the sum of squared differences between
    each point and their cluster means
    """
    return np.sum(np.linalg.norm(X - mu[z.astype("int"), :], axis=1)**2)

def lloyds(X, k = 2, init = None, tol = 10**-6):
    """
    Implement Lloyd's algorithm
    X: Data points in n x d shape
    No. of clusters: k
    returns: cluster assignment vector z, cluster centers (k x d) matrix, and errors vector 
            containing error for each iteration
    Algorithm will go on until the L-2 difference between consecutive cluster centers 
    is more than the tolerance value.
    """
    if init is None:
        mu_0_ind = np.random.choice(X.shape[0], k, replace=False)  # Random initialization
        mu_0 = X[mu_0_ind]
    else:
        mu_0 = init
    
    mu_prev = mu_0
    z = update_z(X, mu_0)
    mu_cur = np.ones(mu_0.shape)*-1
    error_list = [error(X, z, mu_prev)]

    while(np.linalg.norm(mu_prev - mu_cur, ord = 2) > tol):
        mu_prev = deepcopy(mu_cur)     # Deep copying mu_cur to mu_prev before updating mu_cur
        mu_cur = mu_calculate(X, z, k)
        z = update_z(X, mu_cur)
        error_list.append(error(X, z, mu_cur))
    
    return z, mu_cur, np.array(error_list)