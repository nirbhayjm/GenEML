import numpy as np
import numpy.linalg as linalg
import scipy.sparse as ssp
from scipy.sparse import linalg as sp_linalg
from scipy.io import loadmat,savemat
from ops import normalize,sparsify,sigmoid

floatX = np.float32
EPS = 1e-8

def initialize(m_opts):
    m_vars = {}

    np.random.seed(m_opts['random_state'])

    data = loadmat(m_opts['dataset'])
    print "Dataset loaded: ",m_opts['dataset']

    # m_vars['Y_train'] = data['Y_tr']
    # m_vars['X_train'] = data['X_tr']
    m_vars['Y_train'] = sparsify(data['Y_tr'])
    m_vars['X_train'] = sparsify(data['X_tr'])
    m_vars['Y_test'] = sparsify(data['Y_te'])
    m_vars['X_test'] = sparsify(data['X_te'])

    print "Training data --  Y:",m_vars['Y_train'].shape," X:", m_vars['X_train'].shape
    print "Testing data -- Y:",m_vars['Y_test'].shape,"X: ", m_vars['X_test'].shape 

    m_vars['n_users'],m_vars['n_labels'] = m_vars['Y_train'].shape
    m_vars['n_features'] = m_vars['X_train'].shape[1]

    if m_opts['no_feat_normalize'] == False:
        normalize(m_vars['X_train'],norm='l2',axis=1,copy=False)
        normalize(m_vars['X_test'],norm='l2',axis=1,copy=False)

    m_vars['U'] = m_opts['init_std']*np.random.randn(m_vars['n_users'], m_opts['n_components']).astype(floatX)
    m_vars['U_batch'] = np.zeros((m_opts['batch_size'], m_opts['n_components'])).astype(floatX)
    m_vars['V'] = m_opts['init_std']*np.random.randn(m_vars['n_labels'], m_opts['n_components']).astype(floatX)
    m_vars['W'] = m_opts['init_w']*np.random.randn(m_opts['n_components'],m_vars['n_features']).astype(floatX)

    # accumulator of sufficient statistics of label factors
    m_vars['sigma_v'] = np.zeros((m_vars['n_labels'], m_opts['n_components'], m_opts['n_components']))
    for i in range(m_vars['n_labels']):
        m_vars['sigma_v'][i] = m_opts['lam_v']*np.eye(m_opts['n_components'])
    m_vars['x_v'] = np.zeros((m_vars['n_labels'], m_opts['n_components']))

    # accumulator of sufficient statistics of W matrix
    m_vars['sigma_W'] = m_opts['lam_w']*ssp.eye(m_vars['n_features'], m_vars['n_features'], format="csr")
    m_vars['x_W'] = np.zeros((m_vars['n_features'], m_opts['n_components']))

    if m_opts['observance']:
        m_vars['a'],m_vars['b'] = m_opts['init_mu_a'],m_opts['init_mu_b']
        m_vars['mu'] = np.random.beta(m_vars['a'],m_vars['b'],size=(m_vars['n_labels']) ) #, dtype=floatX)
        # m_vars['mu'] = m_opts['init_mu']*np.ones(m_vars['n_labels']).astype(floatX)
    else:
        m_vars['mu'] = np.ones(m_vars['n_labels']).astype(floatX)

    m_vars['performance'] = {'prec@k':[], 'dcg@k':[], 'ndcg@k':[]} # storing the performance measures along iterations

    return m_vars

def update(m_opts, m_vars):
    update_U(m_opts, m_vars)
    update_V(m_opts, m_vars)
    if m_opts['observance']:
        update_observance(m_opts, m_vars)
    update_W(m_opts, m_vars)
    return m_vars

def update_U(m_opts, m_vars):
    for it in range(m_opts['PG_iters']):
        P, N = E_x_omega_row(m_opts, m_vars) # expectation of xi_{nl} for n = i, expecation of omega_{nl} for n = i
        Kappa = PG_row(m_opts, m_vars) # polyagamma kappa_{nl} for n = i
        PN = P*N
        PK = P*Kappa

        for i in range(m_opts['batch_size']):
            PN_i = PN[i][:,np.newaxis]
            PK_i = PK[i]
        
            sigma = m_vars['V'].T.dot(PN_i*m_vars['V']) + m_opts['lam_u']*np.eye(m_opts['n_components'])
            x = m_vars['V'].T.dot(PK_i) + np.asarray((m_opts['lam_u']*m_vars['W']).dot(m_vars['X_batch'][i].todense().T)).reshape(-1)
            # z = np.asarray(z).reshape(-1)
            # x = y+z
            m_vars['U_batch'][i] = linalg.solve(sigma, x)

def update_V(m_opts, m_vars):
    P, N = E_x_omega_col(m_opts, m_vars) # expectation of xi_{nl} for n = i, expecation of omega_{nl} for n = i
    Kappa = PG_col(m_opts, m_vars) # polyagamma kappa_{nl} for n = i
    PN = P*N
    PK = P*Kappa

    for i in range(m_vars['n_labels']):
        PN_i = PN[i][:,np.newaxis]
        PK_i = PK[i]

        sigma = m_vars['U_batch'].T.dot(PN_i*m_vars['U_batch'])# + m_opts['lam_v']*np.eye(m_opts['n_components'])
        x = m_vars['U_batch'].T.dot(PK_i)

        m_vars['sigma_v'][i] = (1-m_vars['gamma'])*m_vars['sigma_v'][i] + m_vars['gamma']*sigma
        m_vars['x_v'][i] = (1-m_vars['gamma'])*m_vars['x_v'][i] + m_vars['gamma']*x
        m_vars['V'][i] = linalg.solve(m_vars['sigma_v'][i], m_vars['x_v'][i])

def update_observance(m_opts, m_vars):
    P = E_x(m_opts, m_vars)
    P = P.sum(axis=0) # sum along column
    mu = (m_vars['a']+P-1)/(m_vars['a']+m_vars['b']+m_opts['batch_size']-2)
    m_vars['mu'] = (1-m_vars['gamma'])*m_vars['mu'] + m_vars['gamma']*mu

def update_W(m_opts, m_vars):
    sigma = m_vars['X_batch_T'].dot(m_vars['X_batch']) + m_opts['lam_w']*ssp.eye(m_vars['n_features'], format="csr")
    m_vars['sigma_W'] = (1-m_vars['gamma'])*m_vars['sigma_W'] + m_vars['gamma']*sigma

    x = m_vars['X_batch'].T.dot(m_vars['U_batch'])
    m_vars['x_W'] = (1-m_vars['gamma'])*m_vars['x_W'] + m_vars['gamma']*x

    if m_opts['use_cg'] != True: # For the Ridge regression on W matrix with the closed form solutions 
        if ssp.issparse(m_vars['sigma_W']):
            m_vars['sigma_W'] = m_vars['sigma_W'].todense()
        sigma = linalg.inv(m_vars['sigma_W']) # O(N^3) time for N x N matrix inversion 
        m_vars['W'] = np.asarray(sigma.dot(m_vars['x_W'])).T

    else: # For the CG on the ridge loss to calculate W matrix
        # assert m_vars['X_batch'].shape[0] == m_vars['U_batch'].shape[0]
        Y = m_vars['x_W']
        X = m_vars['sigma_W']
        for i in range(Y.shape[1]):
            y = Y[:, i]
            w,info = sp_linalg.cg(X, y, x0=m_vars['W'][i,:], maxiter=m_opts['cg_iters'])
            if info < 0:
                print "WARNING: sp_linalg.cg info: illegal input or breakdown"
            m_vars['W'][i, :] = w.T

def E_x_omega_row(m_opts, m_vars):
    # sigmoid = lambda x: 1/(1+np.exp(-x))
    # PSI = m_vars['U_batch'][row_no].dot(m_vars['V'].T)
    PSI = m_vars['U_batch'].dot(m_vars['V'].T)
    E_omega = 0.5*np.tanh(0.5*PSI)/(EPS+PSI)
    # PSI = -PSI
    # PSI = np.clip(PSI, -39, np.inf)
    PSI_sigmoid = sigmoid(-PSI)
    # print "mu shape:",m_vars['mu'].shape
    # print "PSI_sigmoid shape:",PSI_sigmoid.shape
    E_x = (m_vars['mu']*PSI_sigmoid)/(EPS+m_vars['mu']*PSI_sigmoid+(1.-m_vars['mu']))

    E_x[m_vars['Y_batch'].nonzero()] = 1.
    return E_x, E_omega

def PG_row(m_opts, m_vars):
    PG = m_vars['Y_batch'].todense()-0.5
    return np.array(PG) #.reshape(-1)

def E_x_omega_col(m_opts, m_vars):
    # sigmoid = lambda x: 1/(1+np.exp(-x))
    PSI = m_vars['V'].dot(m_vars['U_batch'].T)
    E_omega = 0.5*np.tanh(0.5*PSI)/(EPS+PSI)
    # PSI = -PSI
    # PSI = np.clip(PSI, -39, np.inf)
    PSI_sigmoid = sigmoid(-PSI)
    # E_x = (m_vars['mu'][col_no]*PSI_sigmoid)/(EPS+m_vars['mu'][col_no]*PSI_sigmoid+(1.-m_vars['mu'][col_no]))
    # print "mu shape:",m_vars['mu'].shape
    # print "PSI_sigmoid shape:",PSI_sigmoid.shape
    mu_temp = m_vars['mu'][:,np.newaxis]
    E_x = (mu_temp*PSI_sigmoid)/(EPS+mu_temp*PSI_sigmoid+(1.-mu_temp))

    E_x[m_vars['Y_batch_T'].nonzero()] = 1.
    return E_x, E_omega

def PG_col(m_opts, m_vars):
    # PG = m_vars['Y_batch'][:,col_no].todense()-0.5
    PG = m_vars['Y_batch_T'].todense()-0.5
    return np.array(PG)

def E_x(m_opts, m_vars):
    # sigmoid = lambda x: 1/(1+np.exp(-x))
    PSI = m_vars['U_batch'].dot(m_vars['V'].T)
    # PSI = -PSI
    # PSI = np.clip(PSI, -39, np.inf)
    PSI_sigmoid = sigmoid(-PSI)
    E_x = (m_vars['mu']*PSI_sigmoid)/(EPS+m_vars['mu']*PSI_sigmoid+(1.-m_vars['mu']))
    E_x[m_vars['Y_batch'].nonzero()] = 1.
    return E_x

def predict(m_opts, m_vars, X):
    # sigmoid = lambda x: 1/(1+np.exp(-x))
    U = X.dot(m_vars['W'].T)
    Y_pred = U.dot(m_vars['V'].T)
    # Y_pred = np.clip(Y_pred, -39, np.inf)
    Y_pred = sigmoid(Y_pred)
    Y_pred = Y_pred*m_vars['mu']
    return Y_pred

def saver(vars_path, m_vars, opts_path, m_opts):
    import pickle
    with open(vars_path,"w") as vars_file:
        pickle.dump(m_vars,vars_file)

    import json
    with open(opts_path,"w") as opts_file:
        json.dump(m_opts,opts_file)

def loader(vars_path, opts_path):
    import pickle
    with open(vars_path,"r") as vars_file:
        m_vars = pickle.load(vars_file)

    import json
    with open(opts_path,"w") as opts_file:
        m_opts = json.load(opts_file)
