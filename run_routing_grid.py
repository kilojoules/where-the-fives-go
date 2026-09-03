"""Controlled study: labeler operating point (recall r, precision p) vs absorption."""
import multiprocessing as mp
import sys
from train import Config
from run_all import worker, SEEDS

RECALLS = [0.03, 0.3, 1.0]
PRECISIONS = [1.0, 0.9, 0.7, 0.5]

def configs():
    cfgs = []
    for s in SEEDS:
        for r in RECALLS:
            for p in PRECISIONS:
                cfgs.append(Config(name=f"rt_r{r}_p{p}", seed=s, mode="gram",
                                   routed_frac={5: r, 7: 1.0}, route_precision=p))
        for p in [0.9, 0.7]:  # concentrated FPs from the confusable class
            cfgs.append(Config(name=f"rt_r1.0_p{p}_from3", seed=s, mode="gram",
                               routed_frac={5: 1.0, 7: 1.0}, route_precision=p,
                               route_fp_source="3"))
    return cfgs

if __name__ == "__main__":
    cfgs = configs()
    print(f"{len(cfgs)} grid runs", flush=True)
    with mp.get_context("spawn").Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 6) as pool:
        for msg in pool.imap_unordered(worker, cfgs):
            print(msg, flush=True)
    print("GRID DONE", flush=True)
