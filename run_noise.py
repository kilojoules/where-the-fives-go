"""Label-noise suite: FP/FN/swap noise on training labels, GRAM vs dense."""
import multiprocessing as mp
from train import Config
from run_all import worker, SEEDS

RATES = [0.1, 0.3]
KINDS = ["fp5", "fn5", "swap57"]

def configs():
    cfgs = []
    for s in SEEDS:
        for k in KINDS:
            for r in RATES:
                cfgs.append(Config(name=f"noise_{k}_r{r}", seed=s, mode="gram",
                                   noise_kind=k, noise_rate=r))
                cfgs.append(Config(name=f"noise_{k}_r{r}_dense", seed=s, mode="dense",
                                   noise_kind=k, noise_rate=r))
    return cfgs

if __name__ == "__main__":
    import sys
    cfgs = configs()
    print(f"{len(cfgs)} noise runs", flush=True)
    with mp.get_context("spawn").Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 6) as pool:
        for msg in pool.imap_unordered(worker, cfgs):
            print(msg, flush=True)
    print("NOISE DONE", flush=True)
