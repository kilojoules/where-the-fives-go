import multiprocessing as mp
from track4_modfam import Config, worker

if __name__ == "__main__":
    cfgs = [Config(name=f"t4p2_wd{wd}_lr{lr}", seed=0, mode="dense",
                   weight_decay=wd, lr=lr, epochs=4000, log_every=500)
            for wd, lr in [(0.3, 2e-3), (0.1, 2e-3), (0.1, 1e-3)]]
    with mp.get_context("spawn").Pool(3) as pool:
        for msg in pool.imap_unordered(worker, cfgs):
            print(msg, flush=True)
    print("T4 PILOT2 DONE", flush=True)
