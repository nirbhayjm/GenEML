def argparser():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-model_name', type=str, default='generic_model_name')
    parser.add_argument('-mode', type=str, default='train')

    # Dataset selection
    parser.add_argument('-ds','--dataset', type=str, default='')

    # Train time options
    parser.add_argument('-num_epochs', type=int, default=10)
    parser.add_argument('-random_state', type=int, default=0)
    parser.add_argument('-shuffle_minibatches', type=bool, default=True)
    parser.add_argument('-save_dir', type=str, default="./")
    parser.add_argument('-save_interval', type=int, default=25)
    parser.add_argument('-display_interval', type=int, default=1)
    parser.add_argument('-test_interval', type=int, default=1)
    parser.add_argument('-v','--verbose', action='store_true')
    
    # Model parameters
    parser.add_argument('-observance', type=bool, default=True)
    parser.add_argument('-batch_size', type=int, default=256)
    parser.add_argument('-n_components', type=int, default=100)
    parser.add_argument('-lam_u', type=float, default=1e-3)
    parser.add_argument('-lam_v', type=float, default=1e-3)
    parser.add_argument('-lam_w', type=float, default=1e-3)
    parser.add_argument('-label_normalize', action='store_true')
    parser.add_argument('-cg_iters', type=int, default=10)

    # Learning rate parameters
    parser.add_argument('-lr_alpha', type=float, default=1e-2)
    parser.add_argument('-lr_tau', type=float, default=0.75)
    
    # Variable initializations
    parser.add_argument('-init_mu_a', type=float, default=1.0)
    parser.add_argument('-init_mu_b', type=float, default=1.0)
    parser.add_argument('-init_std', type=float, default=1e-2)
    parser.add_argument('-init_W', type=float, default=1e-2)

    m_opts = vars(parser.parse_args())

    return m_opts

# def make_inputs(m_opts):
#     m_vars = {}

#     if m_opts['mode'] == 'train':
#         pass
#     else:
#         raise ValueError
#     return m_vars
