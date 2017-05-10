'''

An online multi-label learning model.


'''

import numpy as np
from numpy import linalg
import scipy.sparse as ssp
from scipy.sparse import linalg as sp_linalg
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin

import time
from itertools import cycle

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
        elif self.random_state is None:
            self.random_state = 0

        # Preprocess train and optional test data
        X_user, Y_label = preprocess(X_user, Y_label)
        if X_user_test is not None:
            X_user_test, Y_label_test = preprocess(X_user_test, Y_label_test)

        # Get input dimensions
        n_users,n_items = Y_label.shape
        n_user_feats = X_user.shape[1]

        # Preset learning rates at every iteration
        self.lr = self.lr_alpha_0*((1.0 + np.arange(self.max_iter))**(-self.lr_tau))

        # Initial dataset shuffle
        if self.shuffle_minibatches:
            perm = np.random.permutation(n_users)
            X_user = X_user[perm]
            Y_label = Y_label[perm]

        self._init_params(n_user_feats,n_items)

        self._update(Y_label, X_user, Y_label_test, X_user_test)

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

    def _update(Y, X, Y_test, X_test):
        '''
        Model training and evaluation on test set

        '''

        assert ssp.issparse(Y)
        assert Y.getformat() == 'csr'
        n_users = Y.shape[0]
        n_users_batch = self.batch_size

        if self.verbose:
            print "Total users:",n_users
            print "Batch size:",n_users_batch
            print "Iterations per epoch:",n_users//n_users_batch

        start_epoch_t,start_t = time.time(),time.time()
        user_t,item_t,user_feat_t,iterTime = 0.,0.,0.,0.
        expo_t = 0.
        train_prec,test_prec = (0.,0.,0.),(0.,0.,0.)
        delta_expo,delta_user,delta_item = 0.0,0.0,0.0

        def _batch_generator(n_users,bsize):
            start_idx = range(0, n_users, bsize)
            end_idx = start_idx[1:] #+ [n_users]
            while True:
                for lo,hi in zip(start_idx[:-1],end_idx):
                    yield lo,hi

        range_generator = _batch_generator(n_users,n_users_batch)

        for i in xrange(self.max_iter):
            # Compute event flags
            display_flag = (i+1)%self.display_interval == 0
            test_flag = (i+1)%self.test_interval == 0 and X_test is not None
            end_epoch_flag = i%(n_users//n_users_batch) == 0
            save_flag = i%(self.save_interval) == 0 and self.save_params
            epoch_idx = i//(n_users//n_users_batch)

            # Reshuffle minibaches at the end of every epoch
            if end_epoch_flag and self.shuffle_minibatches and i != 0:
                if self.verbose:
                    print "\nReshuffling data after epoch!\t",
                shuffle(Y,X,random_state=self.random_state+i)

            # Save model parameters
            if save_flag and i>0:
                self._save_params(i) # Params saving per iteration i

            # Get minibatch from range generator
            lo,hi = range_generator.next()

            Yb = Y[lo:hi]
            # label_mask = None
            # if self.label_hide:
            #     # print "Hiding labels with dropout %f"%(self.label_dropout)
            #     Yb, label_mask = hide_labels(Yb,self.label_dropout)
            #     # print "Done!"
            YbT = Yb.T.tocsr()
            Xb = X[lo:hi]
            XbT = Xb.T #.tocsr()

            # Time keeper
            if i>0:
                iterTime = time.time() - start_t
                start_t = time.time()
                self.iterTimes[i] = np.array((iterTime,)+train_prec+test_prec)

            # Display per-iteration information
            if end_epoch_flag and self.verbose:
                print('\nEpoch time=%.2f'% (time.time() - start_epoch_t))                     
                start_epoch_t = _writeline_and_time('\nEPOCH #%d\n' % epoch_idx)
            
            if display_flag and self.verbose:
                # start_t = _writeline_and_time('\rITERATION #%d Gamma:%.4g User:%.2fs Item:%.2fs W:%.2fs Expo:%.2f total:%.3fs' % \
                #             (i,self.lr[i],user_t,item_t,user_feat_t,expo_t,iterTime))
                start_t = _writeline_and_time('\rITERATION #%d Gamma:%.4g Time:%.3fs  D_expo:%f' % \
                            (i,self.lr[i],iterTime,delta_expo))

            #=== Perform parameter updates
            user_t,item_t = self._update_factors(Yb, YbT, Xb, gamma_ratio=self.lr[i], label_mask=label_mask)
            if self.flag_expo:
                expo_t, delta_expo = self._update_expo(Yb, n_users_batch, gamma_ratio=self.lr[i], F = Xb.todense(), label_mask=label_mask)
            user_feat_t = self._update_user_feat(Xb,XbT, gamma_ratio=self.lr[i])
            #===

            k = 5
            if acc is True:
                X_pred = self.predict(Xb)
                if self.verbose:
                    print "\nprec on train data : ",
                train_prec = self.evalPrecision(X_pred, Yb.todense(), k, verbose=self.verbose)

                if label_mask is not None and self.verbose:
                    print "Average exposure of hidden labels : ",
                    print self.evalHiddenLabels(Yb, label_mask)

                # if train_prec[0] < self.bad_start_limit:
                #     print "!"*100
                #     print "Killing due to bad start!"
                #     print "Prec@1 is %.4f < Hard limit of %.4f"%(train_prec[0],self.bad_start_limit)
                #     print "!"*100
                #     return

            if acc is True and test_flag:
                if label_names is not None:
                    self.printExposureExtremes(label_names,10,verbose=self.verbose)

                if self.num_chunks > 1:
                    Y_pred_test, Y_test_chunks = self.predict_chunks(X_test,Y_test,self.num_chunks)
                    if self.verbose:
                        print "prec on test data : ",
                    test_ndcg = self.evalPrecisionChunk(Y_pred_test,Y_test_chunks,k,verbose=self.verbose)
                    Y_pred_test = None
                    Y_test_chunks = None
                else:
                    X_pred = self.predict(X_test)
                    if self.verbose:
                        print "prec on test data : ",
                    test_prec = self.evalPrecision(X_pred,Y_test,k,verbose=self.verbose)
                    # if self.verbose:
                    #     print "nDCG@K on test data : ",
                    # test_ndcg = self.nDCG_k(X_pred,Y_test,k,verbose=self.verbose)

        # After training ends, save final model params
        if self.save_params:
            if self.verbose:
                print "Saving final model to:"
            print self._save_params() 
        pass

    def _update_factors(self, X, XT, F, gamma_ratio):
        '''Update user and item collaborative factors with ALS'''
        micro_batch_size = self.batch_size//self.n_jobs if self.n_jobs > 1 else self.batch_size

        start_t = time.time()
        self.theta = self.recompute_user_factors(gamma_ratio, batch_size=micro_batch_size, F_dense=F.todense())
        user_time = time.time() - start_t

        start_t = time.time()
        self.beta = self.recompute_item_factors(self.theta, self.beta, XT,
                                      self.lam_beta,
                                      self.mu,
                                      self.n_jobs,
                                      gamma_ratio,
                                      F_matrix = F.todense(),
                                      batch_size=self.batch_size)
        item_time = time.time() - start_t
        # print "\nAverage change in beta:",np.sqrt(np.average((self.beta-old_beta)**2)),
        return user_time,item_time

    def recompute_user_factors(gamma_ratio, batch_size=256, F_dense=None):
        m, n = Y.shape  # m = number of users, n = number of items
        beta = self.beta
        theta_old = self.theta
        assert beta.shape[0] == n
        assert theta_old.shape[0] == m

        A_batch = self.a_row_batch(Y, F_dense)

        theta_new = np.empty_like(theta_old, dtype=theta_old.dtype)

        for ib in range(m):
            theta_new[ib] = self._solve_users(ib, A_batch[ib], 
                                Y, F_dense)
        return theta_new

    def a_row_batch(self, Y_batch, F):
        '''Compute the posterior of exposure latent variables A by batch'''
        pEX = (1/(1+np.exp(self.theta.dot(self.beta.T))))
        mu = self.mu
        A = (pEX + EPS) / (pEX + EPS + (1 - mu) / mu)
        A[Y_batch.nonzero()] = 1.
        return A

    def _solve_users(self, k, A_k, Y, F_dense):
        '''Update one single user factor theta_k'''

        beta = self.beta
        theta_old_k = self.theta[k]
        W = self.W

        psi = beta.dot(theta_old_k)
        omega_k = 0.5*np.tanh(0.5*psi)/(EPS+psi)
        # omega_k = 0*omega_k + 1.0
        pw = A_k*omega_k
        B = beta.T.dot(pw[:, np.newaxis] * beta) + (self.lam_theta * np.eye(self.n_components))
        
        kappa = np.asarray(Y.getrow(k).todense())[0] - 0.5
        # kappa = 0.0*kappa + 1.0
        t = np.dot(kappa*A_k, beta)
        t2 = self.lam_theta * np.matmul(W, F_dense[k,:].T)
        t2 = np.asarray(t2).reshape(-1)
        a = t + t2

        return LA.solve(B,a)
