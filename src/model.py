'''

An online multi-label learning model.


'''

import numpy as np
import scipy.sparse as ssp
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin

floatX = np.float32
EPS = 1e-8

class OnlineMF(BaseEstimator, TransformerMixin):

    def __init__(self, n_components=100, max_iter=10, batch_size=256,
                random_state=None, cg_iter=10, flag_expo=False, 
                lam_theta=1e-3, lam_beta=1e-3, lam_w=1e-3, init_mu=(1.0,1.0), 
                init_W=0.01, lr_alpha_0=1e-2, lr_tau=0.75, init_std=0.01, 
                shuffle_minibatches=True, save_interval=25, 
                display_interval=1, test_interval=1, normalize_labels=False,
                verbose=False, save_dir='./checkpoints'):

        # Model options
        self.n_components = n_components
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.normalize_labels = normalize_labels
        self.save_params = save_params
        self.save_dir = save_dir
        self.verbose = verbose
        self.shuffle_minibatches = shuffle_minibatches
        self.save_interval = save_interval
        self.display_interval = display_interval
        self.test_interval = test_interval

        # Hyperparameters
        self.init_std = init_std
        self.approx_lr = approx_lr
        self.flag_expo = flag_expo
        self.lam_theta = lam_theta
        self.lam_beta = lam_beta
        self.init_mu = init_mu
        self.a = a
        self.b = b
        self.init_W = init_W
        self.lam_w = lam_w
        self.cg_iter = cg_iter

        # Learning rate decay parameters
        self.lr_alpha_0 = lr_alpha_0
        self.lr_tau = lr_tau

    def normalize(self, M):
        '''
        Normalize
        ----------
        In-place row normalization of a 2D matrix M.

        '''
        if self.normalize_labels:
            sklearn.preprocessing.normalize(M,norm='l2',axis=1,copy=False)
        return

    def preprocess(self, X, Y):
        if not ssp.issparse(Y):
            Y = ssp.csr_matrix(Y)
        return X,Y

    def fit(self, X_user, Y_label, X_user_test=None, Y_label_test=None):
        '''
        Fit the model with user feature matrix X_user and label matrix Y_label.

        Parameters
        ----------
        Y_label : scipy.sparse.csr_matrix, shape (n_users, n_items)
            Training data.

        X_user : User features known apriori, shape (n_users, n_user_feats)
            Training data.

        Y_label_test: scipy.sparse.csr_matrix, shape (n_users, n_items)
            Test/validation data.

        X_user_test: scipy.sparse.csr_matrix, shape (n_users, n_user_feats)
            Test/validation data.

        '''

        # Set random seed
        if type(self.random_state) is int:
            np.random.seed(self.random_state)
        elif self.random_state is not None:
            np.random.setstate(self.random_state)

        # Preprocess train and optional test data
        X_user, Y_label = preprocess(X_user, Y_label)
        if X_user_test is not None:
            X_user_test, Y_label_test = preprocess(X_user_test, Y_label_test)

        # Get input dimensions
        n_users,n_items = Y.shape
        n_user_feats = X.shape[1]

        # Preset learning rates at every iteration
        self.lr = self.lr_alpha_0*((1.0 + np.arange(self.max_iter))**(-self.lr_tau))

        # Initial dataset shuffle
        if self.shuffle_minibatches:
            perm = np.random.permutation(n_users)
            X = X[perm]
            F = F[perm]

        self._init_params(n_user_feats,n_items)

    def _init_params(n_user_feats,n_items):
        '''
        Initialize all the latent factors

        Parameters
        ----------

        theta : Batch user latent feature matrix (n_users_batch, n_components)

        beta : Item latent feature matrix (n_items, n_components)

        W : Weight matrix for user features (n_components, n_user_feats)

        mu : Bernoulli parameter for label ovservance xi, initialized with 
            a beta random distribution (n_items)

        '''
        n_users_batch = self.batch_size

        # Initialize theta, beta and W
        self.theta = self.init_std * np.random.randn(n_users_batch, self.n_components).astype(floatX)
        self.beta = self.init_std * np.random.randn(n_items, self.n_components).astype(floatX)
        self.W = self.init_W * np.random.randn(self.n_components,n_user_feats).astype(floatX)

        if self.flag_expo:
            a0,b0 = self.init_mu
            self.mu = np.random.beta(a0,b0,size=(n_items), dtype=floatX)

        # if self.flag_expo and not(self.covariates):
        #     self.mu = self.init_mu * np.ones(n_items, dtype=floatX)
        # elif not(self.covariates):
        #     self.mu = np.ones(n_items, dtype=floatX)
        # else:
        #     self.mu = None

        # Item exposure factors
        # if self.covariates:
        #     self.FI = self.init_std * \
        #         np.random.randn(n_items, n_user_feats).astype(floatX)
        #         # np.zeros((n_items, n_user_feats), dtype = floatX)#.astype(floatX)
                

        '''Initialize accumulators for online update of theta and beta.'''
        self.theta_A = np.zeros((n_items, self.n_components, self.n_components), dtype=floatX)
        self.theta_b = np.zeros((n_items, self.n_components), dtype=floatX)

        self.W_A = (self.lam_w)*ssp.eye(n_user_feats, dtype=floatX, format="csr")
        self.W_b = np.zeros((n_user_feats, self.n_components), dtype=floatX)

        for i in range(n_items):
            self.theta_A[i] = self.lam_beta*np.eye(self.n_components)
        return

    def function():
        pass
