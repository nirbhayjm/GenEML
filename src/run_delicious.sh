python main.py --dataset="../data/delicious_new.mat" -v \
-bs=1024 -n_components=100 \
-lam_u=1e-2 -lam_v=1e-2 -lam_w=1e-4 \
-use_cg -cg_iters=8 -PG_iters=5 -use_grad \
-lr_alpha=0.1 -lr_tau=0.75 
# -init_mu=0.01