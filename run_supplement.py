"""Supplementary runs: the clean 'isolated' unlabeled-mode control arm."""
import multiprocessing as mp
from train import Config
from run_all import worker, ABSORPTION_FRACS, SEEDS

if __name__ == "__main__":
    cfgs = [Config(name=f"gram_f{f}_coreiso", seed=s, mode="gram",
                   routed_frac={5: f, 7: 1.0}, unlabeled_mode="core_no_pcr")
            for s in SEEDS for f in ABSORPTION_FRACS]
    print(f"{len(cfgs)} supplementary runs", flush=True)
    with mp.get_context("spawn").Pool(5) as pool:
        for msg in pool.imap_unordered(worker, cfgs):
            print(msg, flush=True)
    print("SUPPLEMENT DONE", flush=True)
