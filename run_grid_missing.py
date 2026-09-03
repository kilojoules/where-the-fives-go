import multiprocessing as mp, os
from run_routing_grid import configs
from run_all import worker

if __name__ == "__main__":
    cfgs = [c for c in configs() if not os.path.exists(f"results/{c.name}_seed{c.seed}.json")]
    print(f"{len(cfgs)} missing", flush=True)
    with mp.get_context("spawn").Pool(6) as pool:
        for msg in pool.imap_unordered(worker, cfgs):
            print(msg, flush=True)
    print("GRID DONE", flush=True)
