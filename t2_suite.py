import multiprocessing as mp
from track2_modmath import Config, worker

def all_configs():
    cfgs = []
    for s in [0, 1, 2]:
        for kw in [dict(name="t2_dense", mode="dense"),
                   dict(name="t2_filter", mode="dense", filter_mult=True),
                   dict(name="t2_gram", mode="gram", routed_frac=1.0),
                   dict(name="t2_gram_f01", mode="gram", routed_frac=0.1, pool="all_active"),
                   dict(name="t2_gram_f01_zero", mode="gram", routed_frac=0.1, pool="drop")]:
            cfgs.append(Config(seed=s, weight_decay=0.3, lr=2e-3, epochs=5000,
                               log_every=1000, **kw))
    return cfgs

if __name__ == "__main__":
    with mp.get_context("spawn").Pool(5) as pool:
        for msg in pool.imap_unordered(worker, all_configs()):
            print(msg, flush=True)
    print("T2 SUITE DONE", flush=True)
