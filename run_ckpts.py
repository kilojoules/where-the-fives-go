"""Retrain the models needed for elicitation + alpha-sweep, saving weights."""
import multiprocessing as mp
from train import Config
from run_all import worker, SEEDS

def configs():
    cfgs = []
    for s in SEEDS:
        cfgs.append(Config(name="ck_gram_main", seed=s, mode="gram", save_weights=True))
        cfgs.append(Config(name="ck_filter_no5", seed=s, mode="dense", filter_digits=(5,), save_weights=True))
        cfgs.append(Config(name="ck_dense_all", seed=s, mode="dense", save_weights=True))
        cfgs.append(Config(name="ck_gram_f0.03", seed=s, mode="gram",
                           routed_frac={5: 0.03, 7: 1.0}, unlabeled_mode="all_active", save_weights=True))
    return cfgs

if __name__ == "__main__":
    with mp.get_context("spawn").Pool(6) as pool:
        for msg in pool.imap_unordered(worker, configs()):
            print(msg, flush=True)
    print("CKPTS DONE", flush=True)
