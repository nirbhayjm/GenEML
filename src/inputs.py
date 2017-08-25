def argparser():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-name', type=str, default='generic_model_name')
    parser.add_argument('-mode', type=str, default='train')

    # Dataset selection
    parser.add_argument('-ds','--dataset', type=str, default='')

    ## Train time options
    # number of epcohs to train for
    parser.add_argument('-num_epochs', type=int, default=10)
    parser.add_argument('-random_state', type=int, default=0)
    parser.add_argument('-shuffle_minibatches', action='store_true')
    parser.add_argument('-save', action='store_true')
    parser.add_argument('-display_interval', type=int, default=1)
    parser.add_argument('-test_interval', type=int, default=1)
    parser.add_argument('-test_chunks', type=int, default=1)
    parser.add_argument('-v','--verbose', action='store_true')
    parser.add_argument('-performance_k', type=int, default=5)
    
    ## Model parameters
    # observance on/off
    parser.add_argument('-o', '--observance', action='store_true')
    parser.add_argument('-bs','--batch_size', type=int, default=1024)
    parser.add_argument('-n_components', type=int, default=200)
    # lambda_u, lambda_v, lambda_w
    parser.add_argument('-lam_u', type=float, default=1e-3)
    parser.add_argument('-lam_v', type=float, default=1e-3)
    parser.add_argument('-lam_w', type=float, default=1e-4)
    # normalize labels
    parser.add_argument('-label_normalize', action='store_true')
    # normalize item features
    parser.add_argument('-no_feat_normalize', action='store_true')
    # use conjugate gradient to solve linear equation
    parser.add_argument('-use_cg', action='store_true')
    parser.add_argument('-use_grad', action='store_true')
    parser.add_argument('-grad_alpha', type=float, default=1e-2)
    # number of iterations for conjugate gradient
    parser.add_argument('-cg_iters', type=int, default=5)
    # number of iterations for PG augmentation
    parser.add_argument('-PG_iters', type=int, default=1)

    ## Learning rate parameters
    parser.add_argument('-lr_alpha', type=float, default=0.6)
    parser.add_argument('-lr_tau', type=float, default=0.55)
    
    ## Variable initializations
    parser.add_argument('-init_mu', type=float, default=0.01)
    parser.add_argument('-init_mu_a', type=float, default=1.0)
    parser.add_argument('-init_mu_b', type=float, default=1.0)
    parser.add_argument('-init_std', type=float, default=1e-2)
    parser.add_argument('-init_w', type=float, default=1e-2)

    m_opts = vars(parser.parse_args())

    return m_opts

