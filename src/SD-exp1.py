import numpy as np
from model import initialize,update,saver,predict
from ops import normalize,sparsify,shuffle
from inputs import argparser
from evaluation import precisionAtK,AUC
from scipy.io import loadmat,savemat

import time
import os

def idx_rare(m_opts, m_vars, num=50):
    label_sum = m_vars['Y_train'].sum(axis=0)
    rare_label = np.asarray(label_sum.argsort()).reshape(-1)
    # print rare_label.shape, rare_label
    return rare_label[:num]

if __name__ == '__main__':
    m_opts = argparser()
    print 'Model Options:'
    print m_opts

    print 'Model Variables:'
    m_vars = initialize(m_opts)
    # print m_vars
    
    iter_idx = -1
    start_idx = range(0, m_vars['n_users'], m_opts['batch_size'])
    end_idx = start_idx[1:]

    minibatch_count = m_vars['n_users']//m_opts['batch_size']
    # if minibatch_count == len(start_idx):
    end_idx.append(m_vars['n_users'])
    lr = m_opts['lr_alpha']*(1.0 + np.arange(minibatch_count*m_opts['num_epochs']))**(-m_opts['lr_tau'])
    # lr = np.clip(minibatch_count*m_opts['lr_alpha']*lr,1e-10,0.9)

    if m_opts['save']:
        os.system('mkdir -p checkpoints/'+m_opts['name']+'/')

    rare_labels = idx_rare(m_opts, m_vars)

    # rare_labels = np.asarray(rare_labels).reshape(-1)

    # print rare_labels.shape, rare_labels

    for epoch_idx in range(m_opts['num_epochs']):
        print "Epoch #%d"%epoch_idx
        start_epoch_t = time.time()

        if m_opts['shuffle_minibatches']:
            shuffle(m_vars['Y_train'],m_vars['X_train'],m_vars['U'],random_state=m_opts['random_state']+epoch_idx)

        for minibatch_idx in range(len(start_idx)):
            iter_idx += 1

            display_flag = (iter_idx+1)%m_opts['display_interval'] == 0
            test_flag = (iter_idx+1)%m_opts['test_interval'] == 0
            lo,hi = start_idx[minibatch_idx],end_idx[minibatch_idx]

            m_vars['Y_batch'] = m_vars['Y_train'][lo:hi]
            m_vars['X_batch'] = m_vars['X_train'][lo:hi]
            m_vars['Y_batch_T'] = m_vars['Y_batch'].T
            m_vars['X_batch_T'] = m_vars['X_batch'].T
            m_vars['gamma'] = lr[iter_idx]

            # Updates go here
            print "gamma: ", lr[iter_idx], 
            m_vars = update(m_opts, m_vars)
            m_vars['U'][lo:hi] = m_vars['U_batch'] #copying updated user factors of minibatch to global user factor matrix

            if test_flag:
                # Train and test precision computation goes here
                Y_pred = predict(m_opts, m_vars, m_vars['X_train'])
                p_k = precisionAtK(Y_pred, m_vars['Y_train'], m_opts['performance_k'])
                print " Iter no.: ",iter_idx+1
                print "Training -- ",
                if m_opts['verbose']:
                    for i in p_k:
                        print " %0.4f "%i,
                    print ""
                m_vars['performance']['prec@k'].append(p_k)
                Y_pred = predict(m_opts, m_vars, m_vars['X_test'])
                p_k = precisionAtK(Y_pred[:, rare_labels], m_vars['Y_test'][:, rare_labels], m_opts['performance_k'])
                auc = AUC(Y_pred[:, rare_labels], m_vars['Y_test'][:, rare_labels])
                print " Testing -- "
                print "RARE LABELS- precision@k ",
                if m_opts['verbose']:
                    for i in p_k:
                        print " %0.4f "%i,
                print " AUC -- ", auc

                Y_pred = predict(m_opts, m_vars, m_vars['X_test'])
                p_k = precisionAtK(Y_pred, m_vars['Y_test'], m_opts['performance_k'])
                auc = AUC(Y_pred, m_vars['Y_test'])
                # print " Testing -- "
                print "ALL LABELS - precision@k ",
                if m_opts['verbose']:
                    for i in p_k:
                        print " %0.4f "%i,
                print " AUC -- ", auc


        print('Epoch time=%.2f'% (time.time() - start_epoch_t))

        # print m_vars['mu']
        # data = loadmat("../data/synth_data.mat")
        # print data['mu']
        # for i in range(len(m_vars['mu'])):
        #     print "%0.2f, %.2f" %(m_vars['mu'][i], data['mu'][0][i])

        # Saving at the end of each epoch goes here.
        if m_opts['save']:
            save_path = 'checkpoints/'+m_opts['name']+'/'\
                        +str(epoch_idx)+"_"\
                        +str(minibatch_idx)+"_"
            save_model_name = save_path+".model"
            save_opts = save_path+".txt"
            saver(save_model_name,m_vars,save_opts,m_opts)

