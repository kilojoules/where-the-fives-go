import multiprocessing as mp, os
from run_all import all_configs, worker

if __name__ == "__main__":
    cfgs = [c for c in all_configs()
            if not os.path.exists(f"results/{c.name}_seed{c.seed}.json")]
    print(f"{len(cfgs)} missing runs", flush=True)
    with mp.get_context("spawn").Pool(7) as pool:
        for msg in pool.imap_unordered(worker, cfgs):
            print(msg, flush=True)
    print("MISSING DONE", flush=True)
