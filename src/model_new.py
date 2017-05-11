import numpy as np
import numpy.linalg as linalg
from scipy.io import loadmat,savemat
from ops import normalize,sparsify,shuffle

floatX = np.float32
EPS = 1e-8

def initialize(m_opts):
    m_vars = {}

    np.random.seed(m_opts['random_state'])

    data = loadmat(m_opts['dataset'])
    print "Dataset loaded: ",m_opts['dataset']

    m_vars['Y_train'] = sparsify(data['X_tr'])
	m_vars['F_train'] = sparsify(data['Y_tr'])
	m_vars['Y_test'] = sparsify(data['X_te'])
	m_vars['F_test'] = sparsify(data['Y_te'])

    m_vars['n_users'],m_vars['n_labels'] = m_vars['Y_train'].shape
    m_vars['n_features'] = m_vars['F_train'].shape[1]

    if m_opts['shuffle_minibatches']:
    	shuffle(m_vars['Y_train'],m_vars['X_train'],random_state=m_opts['random_state'])

    if m_opts['label_normalize']:
    	normalize(m_vars['Y_train'],norm='l2',axis=1,copy=False)

    m_vars['theta'] = m_opts['init_std']*np.random.randn(m_vars['n_users'], m_vars['n_components']).astype(floatX)
    m_vars['beta'] = m_opts['init_std']*np.random.randn(m_vars['n_labels'], m_vars['n_components']).astype(floatX)
    m_vars['W'] = m_opts['init_W']*np.random.randn(m_vars['n_components'],m_vars['n_features']).astype(floatX)

    if m_opts['observance']:
    	a0,b0 = m_opts['init_mu_a'],m_opts['init_mu_b']
    	m_vars['mu'] = np.random.beta(a0,b0,size=(m_vars['n_labels']), dtype=floatX)