import multiprocessing as mp
from track2_modmath import Config
from track2_modmath import run

def worker(cfg):
    import torch; torch.set_num_threads(1)
    try:
        run(cfg)
        return f"OK {cfg.name}"
    except Exception as e:
        import traceback; traceback.print_exc()
        return f"FAIL {cfg.name}: {e}"

if __name__ == "__main__":
    cfgs = [
        Config(name="diagA_wd1", seed=0, mode="dense", weight_decay=1.0, epochs=4000, log_every=500),
        Config(name="diagB_wd03_lr2", seed=0, mode="dense", weight_decay=0.3, lr=2e-3, epochs=4000, log_every=500),
        Config(name="diagC_p23", seed=0, mode="dense", p=23, weight_decay=0.3, epochs=4000, log_every=500),
        Config(name="diagD_fullbatch", seed=0, mode="dense", weight_decay=0.1, batch_size=8192, epochs=4000, log_every=500),
    ]
    with mp.get_context("spawn").Pool(4) as pool:
        for msg in pool.imap_unordered(worker, cfgs):
            print(msg, flush=True)
    print("DIAG DONE", flush=True)
