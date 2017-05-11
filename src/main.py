from numpy import clip
from model_new import initialize
from ops import normalize,sparsify
from inputs import argparser
from evaluation import precisionAtK

import time
import os

if __name__ == '__main__':
    m_opts = argparser()
    print 'Model Options:'
    print m_opts

    m_vars = initialize(m_opts)

	iter_idx = -1
	start_idx = range(0, n_users, bsize)
    end_idx = start_idx[1:]
	minibatch_count = m_vars['n_users']//m_vars['batch_size']
	lr = minibatch_count*m_opts['lr_alpha']*((1.0 +\
	np.arange(minibatch_count*m_opts['num_epochs']))**(-m_opts['lr_tau']))
	lr = clip(lr,1e-10,0.9)

    for epoch_idx in range(m_opts['num_epochs']):
    	print "Epoch #%d"%epoch_idx
    	start_epoch_t = time.time()

	    if m_opts['shuffle_minibatches']:
	    	shuffle(m_vars['Y_train'],m_vars['X_train'],random_state=m_opts['random_state'])

    	for minibatch_idx in range(minibatch_count):
    		iter_idx += 1

    		display_flag = (iter_idx+1)%m_opts['display_interval'] == 0
    		test_flag = (iter_idx+1)%m_opts['test_interval'] == 0
    		lo,hi = start_idx[minibatch_idx],end_idx[minibatch_idx]

    		m_vars['Y_batch'] = m_vars['Y_train'][lo:hi]
    		m_vars['X_batch'] = m_vars['X_train'][lo:hi]

    		# Updates go here

    		if test_flag:
    			# Train and test precision computation goes here
    			pass

        print('\nEpoch time=%.2f'% (time.time() - start_epoch_t))
            start_epoch_t = _writeline_and_time('\nEPOCH #%d\tBatch Size :%d\tObservance: %r\n' % \
                                                (epoch_idx+1,m_opts['batch_size'],m_opts['observance']))

        # Saving at the end of each epoch goes here.
        # TODO: Write model saver function.


