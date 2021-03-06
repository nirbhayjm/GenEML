import numpy as np
from model import initialize,update,saver,predict,predictPrecision
from ops import normalize,sparsify,shuffle
from inputs import argparser
from evaluation import *

import time
import os

np.seterr("raise") # To stop execution on overflow warnings

if __name__ == '__main__':
    m_opts = argparser()
    print 'Model Options:'
    print m_opts

    print "Initializing model..."
    m_vars = initialize(m_opts)
    print "Model initialized, beginning training."
    
    # Setting up batch sized ranges of data. The last batch with size less 
    # than 'batch_size' is ignored, but only for the current epoch.
    iter_idx = -1
    start_idx = range(0, m_vars['n_users'], m_opts['batch_size'])
    end_idx = start_idx[1:]

    minibatch_count = m_vars['n_users']//m_opts['batch_size']
    if minibatch_count == len(start_idx):
        end_idx.append(m_vars['n_users'])

    # Learning rate 'lr' is a geometric series decay
    lr = m_opts['lr_alpha']*(1.0 + np.arange(minibatch_count*\
                    m_opts['num_epochs']))**(-m_opts['lr_tau'])
    # Uncomment the following to set hard limits on the learning rate
    # lr = np.clip(minibatch_count*m_opts['lr_alpha']*lr,1e-10,0.9)

    if m_opts['save']: # Creating checkpoints directory
        os.system('mkdir -p checkpoints/'+m_opts['name']+'/')

    for epoch_idx in range(m_opts['num_epochs']):
        print "Epoch #%d"%epoch_idx
        start_epoch_t = time.time()
        update_time = 0
        test_time = 0

        if m_opts['shuffle_minibatches']:
            shuffle(m_vars['Y_train'],m_vars['X_train'],
                    random_state=m_opts['random_state']+epoch_idx)

        for minibatch_idx in range(minibatch_count):
            iter_idx += 1

            display_flag = (iter_idx+1)%m_opts['display_interval'] == 0
            test_flag = (iter_idx+1)%m_opts['test_interval'] == 0
            lo,hi = start_idx[minibatch_idx],end_idx[minibatch_idx]

            m_vars['Y_batch'] = m_vars['Y_train'][lo:hi]
            m_vars['X_batch'] = m_vars['X_train'][lo:hi]
            m_vars['Y_batch_T'] = m_vars['Y_batch'].T
            m_vars['X_batch_T'] = m_vars['X_batch'].T
            m_vars['gamma'] = lr[iter_idx]

            print "Iter:",iter_idx,
            # Gamma : learning rate at current time step
            print "\tGamma: %6g"%m_vars['gamma'], 
            print "\tUpdate Time: %.3f seconds"%update_time

            # Updates
            start_iter_t = time.time()
            m_vars = update(m_opts, m_vars)
            update_time = time.time() - start_iter_t

            if display_flag: # Train precision
                Y_train_pred,_ = predict(m_opts,m_vars,m_vars['X_batch'])
                p_scores = precisionAtK(Y_train_pred, m_vars['Y_batch'], m_opts['performance_k'])
                if m_opts['verbose']:
                    print "Train score:",
                    for i in p_scores:
                        print " %0.4f"%i,
                    print ""

            if test_flag: # Test precision
                start_test_t = time.time()
                if m_opts['test_chunks'] == 1:
                    '''
                    For memory effecient testing, 'test_chunks' breaks the test data into 
                    parts and computes the precision scores aggregates for each chunk 
                    separately; maximum memory usage is bounded by the size of each chunk.
                    '''
                    Y_pred, Y_pred_2 = predict(m_opts,m_vars,m_vars['X_test'])
                    p_scores = precisionAtK(Y_pred, m_vars['Y_test'], m_opts['performance_k'])
                    p_scores_2 = precisionAtK(Y_pred_2, m_vars['Y_test'], m_opts['performance_k'])
                else:
                    p_scores,p_scores_2 = predictPrecision(m_opts, m_vars, m_vars['X_test'], k=5, 
                                                            break_chunks=m_opts['test_chunks'])
                test_time = time.time() - start_test_t
                
                if m_opts['verbose']:
                    print "Test score w/  mu:",
                    for i in p_scores:
                        print " %0.4f "%i,
                    if m_opts['observance']:
                        print "\nTest score w/o mu:",
                        for i in p_scores_2:
                            print " %0.4f "%i,
                    print "\t (%.3f seconds)"%test_time

        print('Epoch time=%.2f'% (time.time() - start_epoch_t))

        # Saving at the end of each epoch.
        if m_opts['save']:
            save_path = 'checkpoints/'+m_opts['name']+'/'\
                        +str(epoch_idx)+"_"\
                        +str(minibatch_idx)+"_"
            save_model_name = save_path+".model"
            save_opts = save_path+".txt"
            saver(save_model_name,m_vars,save_opts,m_opts)
