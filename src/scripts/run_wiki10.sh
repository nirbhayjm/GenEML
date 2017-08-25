python main.py --dataset="../data/wiki10.mat" -v -bs=1024 \
-n_components=100 -lam_u=1e-3 -lam_v=1e-3 -lam_w=1e-4 \
-use_cg -PG_iters=3 -cg_iters=8 -grad_alpha=1e-5 -use_grad \
-shuffle_minibatches -test_interval=1 -test_chunks=8 \
-lr_alpha=0.7 -lr_tau=0.75 -o -init_mu=0.1 \
-init_std=0.01 -init_w=1e-2 
#-label_normalize
