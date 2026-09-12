"""Retrain selected labeler-grid cells with checkpoints for depth-profile probes."""
import multiprocessing as mp
from train import Config
from run_all import worker, SEEDS

def configs():
    cfgs = []
    for s in SEEDS:
        cfgs.append(Config(name="rtck_r0.3_p1.0", seed=s, mode="gram",
                           routed_frac={5: 0.3, 7: 1.0}, save_weights=True))
        for p in [0.9, 0.7, 0.5]:
            cfgs.append(Config(name=f"rtck_r1.0_p{p}", seed=s, mode="gram",
                               routed_frac={5: 1.0, 7: 1.0}, route_precision=p,
                               save_weights=True))
    return cfgs

if __name__ == "__main__":
    with mp.get_context("spawn").Pool(5) as pool:
        for msg in pool.imap_unordered(worker, configs()):
            print(msg, flush=True)
    print("GRIDCK DONE", flush=True)
