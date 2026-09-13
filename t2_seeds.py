"""Does routing aid grokking? Seed extension: dense vs gram, seeds 3-9."""
import multiprocessing as mp
from track2_modmath import Config, worker

if __name__ == "__main__":
    cfgs = []
    for s in range(3, 10):
        for kw in [dict(name="t2x_dense", mode="dense"),
                   dict(name="t2x_gram", mode="gram", routed_frac=1.0)]:
            cfgs.append(Config(seed=s, weight_decay=0.3, lr=2e-3, epochs=5000,
                               log_every=500, **kw))
    with mp.get_context("spawn").Pool(5) as pool:
        for msg in pool.imap_unordered(worker, cfgs):
            print(msg, flush=True)
    print("T2X DONE", flush=True)
