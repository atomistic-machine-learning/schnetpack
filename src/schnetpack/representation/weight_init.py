import torch
import numpy as np

#generates a random square orthogonal matrix of dimension dim
def square_orthogonal_matrix(dim=3, seed=None):
    random_state = np.random
    if seed is not None: #allows to get the same matrix every time
        random_state.seed(seed) 
    H = np.eye(dim)
    D = np.ones((dim,))
    for n in range(1, dim):
        x = random_state.normal(size=(dim-n+1,))
        D[n-1] = np.sign(x[0])
        x[0] -= D[n-1]*np.sqrt((x*x).sum())
        # Householder transformation
        Hx = (np.eye(dim-n+1) - 2.*np.outer(x, x)/(x*x).sum())
        mat = np.eye(dim)
        mat[n-1:, n-1:] = Hx
        H = np.dot(H, mat)
        # Fix the last sign such that the determinant is 1
    D[-1] = (-1)**(1-(dim % 2))*D.prod()
     # Equivalent to np.dot(np.diag(D), H) but faster, apparently
    H = (D*H.T).T
    return H

#generates a random (semi-)orthogonal matrix of size NxM
def semi_orthogonal_matrix(N, M, seed=None):
    if N > M: #number of rows is larger than number of columns
        square_matrix = square_orthogonal_matrix(dim=N, seed=seed)
    else: #number of columns is larger than number of rows
        square_matrix = square_orthogonal_matrix(dim=M, seed=seed)
    return square_matrix[:N,:M]

#generates a weight matrix with variance according to Glorot initialization
#based on a random (semi-)orthogonal matrix
#neural networks are expected to learn better when features are decorrelated
#(stated by eg. "Reducing overfitting in deep networks by decorrelating representations",
#"Dropout: a simple way to prevent neural networks from overfitting",
#"Exact solutions to the nonlinear dynamics of learning in deep linear neural networks")
def semi_orthogonal_glorot_weights(n_in, n_out, scale=2.0, seed=None):
    W = semi_orthogonal_matrix(n_in, n_out)
    W *= np.sqrt(scale/((n_in+n_out)*W.var())) 
    return W

#given a tensor x, gives back the variance 
def reduce_var(x, axis=None, keepdims=False):
    """Variance of a tensor, alongside the specified axis.

    # Arguments
        x: A tensor or variable.
        axis: An integer, the axis to compute the variance.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1. If `keepdims` is `True`,
            the reduced dimension is retained with length 1.

    # Returns
        A tensor with the variance of elements of `x`.
    """

    m = torch.mean(x, dims=axis, keepdim=True)
    devs_squared = torch.square(x - m)
    return torch.mean(devs_squared, dim=axis, keepdim=keepdims)

