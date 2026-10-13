"""Regenerate the 6 dense LM checkpoints (t5_dense, t6_dense x 3 seeds —
identical configs to all_configs()), then run steer_dense5. Remote helper;
training rewrites results/t5_dense_seed*.json and t6_dense_seed*.json, so
compare them against the committed values."""

import multiprocessing as mp

import track5_stories as t5
import track6_polysemy as t6


def worker(cfg):
    return (t5 if cfg.name.startswith("t5") else t6).worker(cfg)


if __name__ == "__main__":
    cfgs = ([t5.Config(name="t5_dense", seed=s, mode="dense") for s in [0, 1, 2]]
            + [t6.Config(name="t6_dense", seed=s, mode="dense") for s in [0, 1, 2]])
    with mp.get_context("spawn").Pool(6) as pool:
        for msg in pool.imap_unordered(worker, cfgs):
            print(msg, flush=True)
    print("RETRAIN DONE", flush=True)
    import steer_dense5
    steer_dense5.main()
