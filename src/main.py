import numpy as np
from model_new import initialize,update,saver
from ops import normalize,sparsify,shuffle
from inputs import argparser
from evaluation import precisionAtK

import time
import os

if __name__ == '__main__':
    m_opts = argparser()
    print 'Model Options:'
    print m_opts

    m_vars = initialize(m_opts)
    print 'Model Variables:'
    # print m_vars
    
    iter_idx = -1
    start_idx = range(0, m_vars['n_users'], m_opts['batch_size'])
    end_idx = start_idx[1:]

    minibatch_count = m_vars['n_users']//m_opts['batch_size']
    lr = (1.0 + np.arange(minibatch_count*m_opts['num_epochs']))**(-m_opts['lr_tau'])
    lr = np.clip(minibatch_count*m_opts['lr_alpha']*lr,1e-10,0.9)

    if m_opts['save']:
        os.system('mkdir -p checkpoints/'+m_opts['name']+'/')

    for epoch_idx in range(m_opts['num_epochs']):
        print "Epoch #%d"%epoch_idx
        start_epoch_t = time.time()

        if m_opts['shuffle_minibatches']:
            shuffle(m_vars['Y_train'],m_vars['X_train'],m_vars['U'],random_state=m_opts['random_state']+epoch_idx)

        for minibatch_idx in range(minibatch_count):
            iter_idx += 1

            display_flag = (iter_idx+1)%m_opts['display_interval'] == 0
            test_flag = (iter_idx+1)%m_opts['test_interval'] == 0
            lo,hi = start_idx[minibatch_idx],end_idx[minibatch_idx]

            m_vars['Y_batch'] = m_vars['Y_train'][lo:hi]
            m_vars['X_batch'] = m_vars['X_train'][lo:hi]
            m_vars['gamma'] = lr[iter_idx]

            # Updates go here
            m_vars = update(m_opts, m_vars)
            m_vars['U'][lo:hi] = m_vars['U_batch'] #copying updated user factors of minibatch to global user factor matrix

            if test_flag:
                # Train and test precision computation goes here
                Y_pred = predict(m_opts, m_vars)
                p_k = precisionAtK(Y_pred, m_vars['Y_test'], m_opts['performance_k'])
                if m_opts['verbose']:
                    for i in p_k:
                        print " %0.4f "%i,
                    print ""
                m_vars['performance']['prec@k'].append(p_k)

        print('\nEpoch time=%.2f'% (time.time() - start_epoch_t))

        # Saving at the end of each epoch goes here.
        if m_opts['save']:
            save_path = 'checkpoints/'+m_opts['name']+'/'\
                        +str(epoch_idx)+"_"\
                        +str(minibatch_idx)+"_"
            save_model_name = save_path+".model"
            save_opts = save_path+".txt"
            saver(save_model_name,m_vars,save_opts,m_opts)
