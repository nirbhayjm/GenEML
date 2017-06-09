def argparser():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-name', type=str, default='generic_model_name')
    parser.add_argument('-mode', type=str, default='train')

    # Dataset selection
    parser.add_argument('-ds','--dataset', type=str, default='')

    # Train time options
    parser.add_argument('-num_epochs', type=int, default=10)
    parser.add_argument('-random_state', type=int, default=0)
    parser.add_argument('-shuffle_minibatches', action='store_true')
    # parser.add_argument('-save_dir', type=str, default="./")
    # parser.add_argument('-save_interval', type=int, default=25)
    parser.add_argument('-save', type=bool, default=True)
    parser.add_argument('-display_interval', type=int, default=1)
    parser.add_argument('-test_interval', type=int, default=1)
    parser.add_argument('-v','--verbose', action='store_true')
    parser.add_argument('-performance_k', type=int, default=5)
    
    # Model parameters
    parser.add_argument('-o', '--observance', action='store_true')
    parser.add_argument('-bs','--batch_size', type=int, default=1024)
    # parser.add_argument('-u_bsize', type=int, default=256)
    # parser.add_argument('-l_bsize', type=int, default=256)
    parser.add_argument('-n_components', type=int, default=200)
    parser.add_argument('-lam_u', type=float, default=1e-3)
    parser.add_argument('-lam_v', type=float, default=1e-3)
    parser.add_argument('-lam_w', type=float, default=1e-4)
    parser.add_argument('-label_normalize', action='store_true')
    parser.add_argument('-use_cg', action='store_true')
    parser.add_argument('-cg_iters', type=int, default=5)
    parser.add_argument('-PG_iters', type=int, default=1)

    # Learning rate parameters
    parser.add_argument('-lr_alpha', type=float, default=0.6)
    parser.add_argument('-lr_tau', type=float, default=0.55)
    
    # Variable initializations
    parser.add_argument('-init_mu', type=float, default=0.01)
    parser.add_argument('-init_mu_a', type=float, default=1.0)
    parser.add_argument('-init_mu_b', type=float, default=1.0)
    parser.add_argument('-init_std', type=float, default=1e-2)
    parser.add_argument('-init_w', type=float, default=1e-2)

    m_opts = vars(parser.parse_args())

    return m_opts

