"""Run the full experiment suite in parallel worker processes (CPU)."""

import multiprocessing as mp
import sys

from train import Config, run

SEEDS = [0, 1, 2]
ABSORPTION_FRACS = [0.03, 0.1, 0.3]


def all_configs():
    cfgs = []
    for s in SEEDS:
        # main GRAM run: modules for 5 and 7, fully labeled
        cfgs.append(Config(name="gram_main", seed=s, mode="gram"))
        # dense baseline on all data (active-param-matched: core path only)
        cfgs.append(Config(name="dense_all", seed=s, mode="dense"))
        # data-filtering baselines (the paper's gold standard for removal)
        cfgs.append(Config(name="filter_no5", seed=s, mode="dense", filter_digits=(5,)))
        cfgs.append(Config(name="filter_no7", seed=s, mode="dense", filter_digits=(7,)))
        cfgs.append(Config(name="filter_no57", seed=s, mode="dense", filter_digits=(5, 7)))
        # absorption sweep: only fraction f of 5s routed; 7 stays fully labeled
        for f in ABSORPTION_FRACS:
            cfgs.append(Config(name=f"gram_f{f}", seed=s, mode="gram",
                               routed_frac={5: f, 7: 1.0}, unlabeled_mode="all_active"))
            cfgs.append(Config(name=f"gram_f{f}_coreunl", seed=s, mode="gram",
                               routed_frac={5: f, 7: 1.0}, unlabeled_mode="core"))
            # clean mechanism control: unlabeled 5s never activate any module
            cfgs.append(Config(name=f"gram_f{f}_coreiso", seed=s, mode="gram",
                               routed_frac={5: f, 7: 1.0}, unlabeled_mode="core_no_pcr"))
            # partial-filter baseline: filtering can only drop the labeled fraction
            cfgs.append(Config(name=f"filter5_f{f}", seed=s, mode="dense", filter_frac5=f))
    return cfgs


def worker(cfg):
    import torch
    torch.set_num_threads(2)
    try:
        run(cfg)
        return f"OK {cfg.name} seed{cfg.seed}"
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"FAIL {cfg.name} seed{cfg.seed}: {e}"


if __name__ == "__main__":
    n_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    cfgs = all_configs()
    print(f"{len(cfgs)} runs, {n_workers} workers", flush=True)
    with mp.get_context("spawn").Pool(n_workers) as pool:
        for msg in pool.imap_unordered(worker, cfgs):
            print(msg, flush=True)
    print("ALL DONE", flush=True)
