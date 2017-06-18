python main.py --dataset="../data/mediamill_1.mat" -bs=512 -v \
-n_components=32 -shuffle_minibatches \
-lam_u=1e-3 -lam_v=1e-3 -lam_w=1e-4 \
-use_cg -PG_iters=5 -cg_iters=10 -use_grad -grad_alpha=1e-3 \
-test_interval=1 -test_chunks=1 \
-lr_alpha=0.5 -lr_tau=0.85 -o -init_mu=0.1