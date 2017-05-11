import numpy as np
import numpy.linalg as linalg
from scipy.io import loadmat,savemat
from ops import normalize,sparsify

floatX = np.float32
EPS = 1e-8

def initialize(m_opts):
    m_vars = {}

    np.random.seed(m_opts['random_state'])

    data = loadmat(m_opts['dataset'])
    print "Dataset loaded: ",m_opts['dataset']

    m_vars['Y_train'] = sparsify(data['X_tr'])
	m_vars['X_train'] = sparsify(data['Y_tr'])
	m_vars['Y_test'] = sparsify(data['X_te'])
	m_vars['X_test'] = sparsify(data['Y_te'])

    m_vars['n_users'],m_vars['n_labels'] = m_vars['Y_train'].shape
    m_vars['n_features'] = m_vars['X_train'].shape[1]

    if m_opts['label_normalize']:
    	normalize(m_vars['Y_train'],norm='l2',axis=1,copy=False)

    m_vars['U'] = m_opts['init_std']*np.random.randn(m_vars['n_users'], m_vars['n_components']).astype(floatX)
    m_vars['Ub'] = np.zeros((m_vars['batch_size'], m_vars['n_components'])).astype(floatX)
    m_vars['V'] = m_opts['init_std']*np.random.randn(m_vars['n_labels'], m_vars['n_components']).astype(floatX)
    m_vars['W'] = m_opts['init_W']*np.random.randn(m_vars['n_components'],m_vars['n_features']).astype(floatX)

    if m_opts['observance']:
    	a0,b0 = m_opts['init_mu_a'],m_opts['init_mu_b']
    	m_vars['mu'] = np.random.beta(a0,b0,size=(m_vars['n_labels']), dtype=floatX)
    else:
    	m_vars['mu'] = np.ones(m_vars['n_labels']).astype(floatX)

    return m_vars

def update(m_opts, m_vars):
	update_U(m_opts, m_vars)
	update_V(m_opts, m_vars)
	if m_opts['observance']:
		update_observance(m_opts, m_vars)
	update_W(m_opts, m_vars)
	return m_vars

def update_U(m_opts, m_vars):
	P = E_xi(m_opts, m_vars) # expectation of xi_{nl}
	N = E_omega(m_opts, m_vars) # expecation of omega_{nl}
	K = PG(m_opts, m_vars) # polyagamma kappa_{nl}

	sigma = 


def saver(vars_path, m_vars, opts_path, m_opts):
    # for key,val in m_vars:
    #     if type(val) is np.ndarray:
    #         np.save(vars_path+"_"+"key",val)
    import pickle
    with open(vars_path,"w") as vars_file:
        pickle.dump(m_vars,vars_file)

    import json
    with open(opts_path,"w") as opts_file:
        # opts_file.write(str(m_opts))
        json.dump(m_opts,opts_file)

def loader(vars_path, opts_path):
    import pickle
    with open(vars_path,"r") as vars_file:
        m_vars = pickle.load(vars_file)

    import json
    with open(opts_path,"w") as opts_file:
        m_opts = json.load(opts_file)