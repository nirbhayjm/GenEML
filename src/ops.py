import sklearn

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

def shuffle(X, Y, Z, random_state):
    sklearn.utils.shuffle(X,Y,Z,random_state=random_state)