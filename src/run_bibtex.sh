python main.py --dataset="../data/bibtex_new.mat" -v -o -bs=4880 \
-n_components=200 -lam_u=1e-3 -lam_v=1e-3 -lam_w=1e-4 \
-use_cg=True -cg_iters=5 -PG_iters=5 \
-lr_alpha=0.6 -lr_tau=0.55 -init_mu=0.01