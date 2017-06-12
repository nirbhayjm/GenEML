import sklearn
import numpy as np

def normalize(M):
    '''
    Normalize
    ----------
    In-place row normalization of a 2D matrix M.

    '''
    if self.normalize_labels:
        sklearn.preprocessing.normalize(M,norm='l2',axis=1,copy=False)
    return

def sparsify(Y):
    import scipy.sparse as ssp

    if not ssp.issparse(Y):
        Y = ssp.csr_matrix(Y)
    return Y

def shuffle(X, Y, random_state):
    sklearn.utils.shuffle(X,Y,random_state=random_state)

def sigmoid(x):
    y = np.clip(x, -20, np.inf)
    return 1/(1+np.exp(-y))