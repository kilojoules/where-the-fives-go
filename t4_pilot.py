import multiprocessing as mp
from track4_modfam import Config, worker

if __name__ == "__main__":
    cfgs = [Config(name=f"t4_pilot_wd{wd}", seed=0, mode="dense",
                   weight_decay=wd, epochs=3000, log_every=500)
            for wd in (0.1, 0.3, 1.0)]
    with mp.get_context("spawn").Pool(3) as pool:
        for msg in pool.imap_unordered(worker, cfgs):
            print(msg, flush=True)
    print("T4 PILOT DONE", flush=True)
