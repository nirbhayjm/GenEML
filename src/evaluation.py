import numpy as np
import scipy.sparse as ssp
from sklearn.metrics import roc_auc_score

def precisionAtK(Y_pred_orig, Y_true_orig, k, verbose=False):
    Y_pred = Y_pred_orig.copy()
    Y_true = Y_true_orig.copy()
    row_sum = np.asarray(Y_true.sum(axis=1)).reshape(-1)
    indices = row_sum.argsort()
    row_sum.sort()
    start = 0
    while start < len(indices) and row_sum[start] == 0:
        start +=1
    indices = indices[start:]
    Y_pred = Y_pred[indices,:]
    Y_true = Y_true[indices,:]
    p = np.zeros(k)
    assert Y_pred.shape == Y_true.shape
    n_items,n_labels = Y_pred.shape
    prevMatch = 0
    for i in xrange(1,k+1):
        Jidx = np.argmax(Y_pred,1)
        prevMatch += np.sum(Y_true[np.arange(n_items),Jidx])
        Y_pred[np.arange(n_items),Jidx] = -np.inf
        p[i-1] = prevMatch/(i*n_items)
    return tuple(p[[0,2,4]])

def precisionAtKChunks(Y_pred_orig, Y_true_orig, k,verbose=True):
    p = np.zeros((len(Y_pred_orig),k))
    n_total_items = 0
    n_labels = 0
    for c,(Y_pred_chunk,Y_true_chunk) in enumerate(zip(Y_pred_orig,Y_true_orig)):
        prevMatch = 0
        # print "Computing %dth precision"%c
        Y_pred = Y_pred_chunk.copy()
        Y_true = Y_true_chunk.copy()
        n_items, n_labels = Y_pred.shape
        n_total_items += n_items
        for i in xrange(1,k+1):
            Jidx = np.argmax(Y_pred,1)
            prevMatch += Y_true[np.arange(n_items),Jidx].sum()
            Y_pred[np.arange(n_items),Jidx] = -np.inf
            p[c,i-1] = prevMatch #/(i*n_items)

    q = np.zeros(k)
    for i in range(1,k+1):
        q[i-1] = p[:,i-1].sum()/(i*n_total_items)

    return tuple(q[[0,2,4]])

def DCG_k(Y_pred_orig, Y_true_orig, k, verbose=False):
    Y_pred = np.asarray(Y_pred_orig.copy())
    Y_true = Y_true_orig.copy()
    p = np.zeros(k)
    assert Y_pred.shape == Y_true.shape,\
        "Shape mismatch:"+str(Y_pred.shape)+str(Y_true.shape)
    n_items,n_labels = Y_pred.shape
    prevMatch = 0

    for i in xrange(1,k+1):
        Jidx = np.argmax(Y_pred,1)
        prevMatch += np.sum(Y_true[np.arange(n_items),Jidx])
        Y_pred[np.arange(n_items),Jidx] = -np.inf
        p[i-1] = prevMatch/(np.log2(i+1)*n_items)
    
    return p

def nDCG_k(Y_pred, Y_true, k, verbose=False):
    DCG_k_score = DCG_k(Y_pred, Y_true, k)
    if ssp.issparse(Y_true):
        IDCG_k_score = DCG_k(Y_true.todense(), Y_true, k)
    else:
        IDCG_k_score = DCG_k(Y_true, Y_true, k)

    p = DCG_k_score/IDCG_k_score
    return tuple(p[[0,2,4]])

def AUC(Y_pred, Y_true):
    row_sum = np.asarray(Y_true.sum(axis=1)).reshape(-1)
    indices = row_sum.argsort()
    row_sum.sort()
    start = 0
    while start < len(indices) and row_sum[start] == 0:
        start +=1
    indices = indices[start:]
    Y_pred = Y_pred[indices,:]
    Y_true = Y_true[indices,:]
    if ssp.issparse(Y_true):
        Y_true = Y_true.todense()
    if ssp.issparse(Y_pred):
        Y_pred = Y_pred.todense()
    return roc_auc_score(Y_true, Y_pred)
    