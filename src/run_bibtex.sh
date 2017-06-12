python main.py --dataset="../data/bibtex_new.mat" -v -bs=1024 \
-n_components=100 -shuffle_minibatches \
-lam_u=1e-3 -lam_v=1e-3 -lam_w=1e-4 \
-use_cg -cg_iters=4  -PG_iters=5 \
-test_interval=1 -test_chunks=1 \
-lr_alpha=0.6 -lr_tau=0.75
# -o -init_mu=0.01
