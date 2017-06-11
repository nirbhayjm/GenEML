python SD-exp1.py --dataset="../data/bibtex_missing.mat" -shuffle_minibatches -v  -bs=976 \
-n_components=150 -lam_u=1e-3 -lam_v=1e-3 -lam_w=1e-5 \
-use_cg -cg_iters=5 -PG_iters=5 \
-lr_alpha=1.0 -lr_tau=0.55 -init_mu=0.01 \
-init_mu_a=1. -init_mu_b=1.
