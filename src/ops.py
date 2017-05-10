import sklearn

def normalize(self, M):
    '''
    Normalize
    ----------
    In-place row normalization of a 2D matrix M.

    '''
    if self.normalize_labels:
        sklearn.preprocessing.normalize(M,norm='l2',axis=1,copy=False)
    return

def sparsify(self, X, Y):
	import scipy.sparse as ssp

    if not ssp.issparse(Y):
        Y = ssp.csr_matrix(Y)
    return X,Y