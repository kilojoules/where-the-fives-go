"""Regenerate the 6 track-4 checkpoints needed for steering (gram5 + filter5,
3 seeds — identical configs to all_configs()), then run steer_t4. Remote
helper; verify regenerated results/*.json against the committed values."""

import multiprocessing as mp

import torch

from track4_modfam import Config, run


def worker(cfg):
    torch.set_num_threads(2)  # 6 workers in a 12-core Slurm allocation
    try:
        run(cfg)
        return f"OK {cfg.name} seed{cfg.seed}"
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"FAIL {cfg.name} seed{cfg.seed}: {e}"


if __name__ == "__main__":
    cfgs = []
    for s in [0, 1, 2]:
        cfgs.append(Config(name="t4_gram5", seed=s, mode="gram", target_b=5,
                           log_every=1000))
        cfgs.append(Config(name="t4_filter5", seed=s, mode="dense", target_b=5,
                           filter_target=True, log_every=1000))
    with mp.get_context("spawn").Pool(6) as pool:
        for msg in pool.imap_unordered(worker, cfgs):
            print(msg, flush=True)
    print("RETRAIN DONE", flush=True)
    import steer_t4
    steer_t4.main()
