python main.py --dataset="../data/bibtex_new.mat" -v -o -bs=976 \
-n_components=150 -lam_u=1e-3 -lam_v=1e-3 -lam_w=1e-3 \
-use_cg -cg_iters=3 -PG_iters=3 \
-lr_alpha=1.0 -lr_tau=0.55 -init_mu=0.01
