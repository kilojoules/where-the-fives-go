"""Absorption dose-response: routed 5s fixed at ~162, unlabeled-pool size varied.

k = fraction of 5s kept in the dataset; routed_frac set so the routed count
stays ~162; the remaining kept 5s are unlabeled (all-active). k=0.03 is the
data-matched control (no unlabeled pool at all)."""
import multiprocessing as mp
from train import Config
from run_all import worker, SEEDS

N5 = 5421  # 5s in MNIST train
KEEPS = [0.03, 0.1, 0.4, 1.0]

def configs():
    cfgs = []
    for s in SEEDS:
        for k in KEEPS:
            survivors = N5 - round((1 - k) * N5)
            rf = 162 / survivors
            cfgs.append(Config(name=f"dose_k{k}", seed=s, mode="gram",
                               filter_frac5=1 - k, routed_frac={5: rf, 7: 1.0},
                               unlabeled_mode="all_active", save_weights=True))
    return cfgs

if __name__ == "__main__":
    with mp.get_context("spawn").Pool(6) as pool:
        for msg in pool.imap_unordered(worker, configs()):
            print(msg, flush=True)
    print("DOSE DONE", flush=True)
