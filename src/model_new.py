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

    # accumulator of sufficient statistics of label factors
    m_vars['sigma_v'] = np.zeros((m_opts['n_labels'], m_opts['n_components'], m_opts['n_components']))
    m_vars['x_v'] = np.zeros((m_opts['n_labels'], m_opts['n_components']))

    # accumulator of sufficient statistics of W matrix
    m_vars['sigma_W'] = np.zeros((m_vars['n_features'], m_vars['n_features']))
    m_vars['x_W'] = np.zeros((m_vars['n_features'], m_vars['n_components']))

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
	for i in range(m_opts['batch_size']):
		for it in range(m_opts['PG_iters']):
			P_i, N_i = E_xi_omega_row(i, m_opts, m_vars) # expectation of xi_{nl} for n = i, expecation of omega_{nl} for n = i
			K_i = PG_row(i, m_opts, m_vars) # polyagamma kappa_{nl} for n = i
			PN_i = P_i*N_i
			PK_i = P_i*K_i
			PN_i = PN_i[:,np.newaxis]
			PK_i = PK_i[:,np.newaxis]

			sigma = m_vars['V'].T.dot(PN_i*m_vars['V']) + m_opts['lam_u']*np.eye(m_opts['n_components'])
			x = m_vars['V'].T.dot(PK_i) + m_opts['lam_u']*m_vars['W'].dot(m_vars['Xb'][i])

			m_vars['Ub'][i] = LA.solve(sigma, x)

def update_V(m_opts, m_vars):
	for i in range(m_vars['n_labels']):
		P_i, N_i = E_xi_omega_col(i, m_opts, m_vars) # expectation of xi_{nl} for l = i, expecation of omega_{nl} for l = i 
		K_i = PG_col(i, m_opts, m_vars) # polyagamma kappa_{nl} for l = i
		PN_i = P_i*N_i
		PK_i = P_i*K_i
		PN_i = PN_i[:,np.newaxis]
		PK_i = PK_i[:,np.newaxis]

		sigma = m_vars['Ub'].T.dot(PN_i*m_vars['Ub']) + m_opts['lam_v']*np.eye(m_opts['n_components'])
		x = m_vars['Ub'].T.dot(PK_i)

		m_vars['sigma_v'][i] = (1-m_vars['gamma'])*m_vars['sigma_v'][i] + gamma_ratio*sigma
		m_vars['x_v'][i] = (1-m_vars['gamma'])*m_vars['x_v'][i] + gamma_ratio*x
		m_vars['V'][i] = LA.solve(m_vars['sigma_v'][i], m_vars['x_v'][i])

def update_observance(m_opts, m_vars):
	P = E_xi(m_opts, m_vars)
	P = P.sum(axis=0) # sum along column
	mu = (m_vars['a']+P-1)/(m_vars['a']+m_vars['b']+m_vars['batch_size']-2)
    m_vars['mu'] = (1-m_opts['gamma'])*m_vars['mu'] + gamma_ratio*mu

def update_W(m_opts, m_vars):
	sigma = m_opts['lam_w']*(m_vars['Xb'].dot(m_vars['Xb']))
	m_vars['sigma_W'] = (1-m_vars['gamma'])*m_vars['sigma_W'] + m_vars['gamma']*sigma

	x = m_vars['Xb'].T.dot(m_vars['Ub'])
	m_vars['x_W'] = (1-m_vars['gamma'])*m_vars['x_W'] + m_vars['gamma']*x

	if m_opts['use_cg'] != True: # For the Ridge regression on W matrix with the closed form solutions 
	    sigma = LA.inv(m_vars['sigma_W']) # O(N^3) time for N x N matrix inversion 
	    m_vars = sigma.dot(m_vars['x_W'])
    else: # For the CG on the ridge loss to calculate W matrix
        Y = m_vars['x_W']
        X = m_vars['sigma_W']
        for i in range(Y.shape[1]):
            y = Y[:, i]
            w,info = sp_linalg.cg(X, y, x0=self.W[i,:], maxiter=self.cg_iter)
            if info < 0:
                print "WARNING: sp_linalg.cg info: illegal input or breakdown"
            m_vars['W'][i, :] = w.T

def E_xi_omega_row(row_no, m_opts, m_vars):
	sigmoid = lambda x: 1/(1+np.exp(-x))
	PSI = m_vars['Ub'][row_no].dot(m_vars['V'].T)
	E_omega = 0.5*np.tanh(0.5*PSI)/(EPS+PSI)
	PSI = -PSI
	PSI = np.clip(PSI, -np.inf, -20)
	PSI_sigmoid = sigmoid(PSI)
	E_xi = (m_vars['mu']*PSI_sigmoid)/(EPS+m_vars['mu']*PSI_sigmoid+(1.-m_vars['mu']))
	E_xi[m_vars['Yb'][row_no].nonzero()] = 1.
	return E_xi, E_omega

def PG_row(row_no, m_opts, m_vars):
	return m_vars['Yb'][row_no]-0.5

def E_xi_omega_col(col_no, m_opts, m_vars):
	sigmoid = lambda x: 1/(1+np.exp(-x))
	PSI = m_vars['V'][col_no].dot(m_vars['Ub'].T)
	E_omega = 0.5*np.tanh(0.5*PSI)/(EPS+PSI)
	PSI = -PSI
	PSI = np.clip(PSI, -np.inf, -20)
	PSI_sigmoid = sigmoid(PSI)
	E_xi = (m_vars['mu'][col_no]*PSI_sigmoid)/(EPS+m_vars['mu'][col_no]*PSI_sigmoid+(1.-m_vars['mu'][col_no]))
	E_xi[m_vars['Yb'][:,col_no].nonzero()] = 1.
	return E_xi, E_omega

def PG_col(col_no, m_opts, m_vars):
	return m_vars['Yb'][:,col_no]-0.5

def E_xi(m_opts, m_vars):
	sigmoid = lambda x: 1/(1+np.exp(-x))
	PSI = m_vars['Ub'].dot(m_vars['V'].T)
	PSI = -PSI
	PSI = np.clip(PSI, -np.inf, -20)
	PSI_sigmoid = sigmoid(PSI)
	E_xi = (m_vars['mu']*PSI_sigmoid)/(EPS+m_vars['mu']*PSI_sigmoid+(1.-m_vars['mu']))
	E_xi[m_vars['Yb'].nonzero()] = 1.
	return E_xi

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
