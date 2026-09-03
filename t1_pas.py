import multiprocessing as mp
from track1_fashion import Config, worker

if __name__ == "__main__":
    cfgs = []
    for s in [0, 1, 2]:
        cfgs.append(Config(name="t1_gram_pas0", seed=s, mode="gram", routed_frac=1.0, p_as=0.0))
        cfgs.append(Config(name="t1_gram_pas01", seed=s, mode="gram", routed_frac=1.0, p_as=0.1))
        cfgs.append(Config(name="t1_gram_f05_pas01", seed=s, mode="gram", routed_frac=0.05,
                           pool="all_active", p_as=0.1))
    with mp.get_context("spawn").Pool(3) as pool:
        for msg in pool.imap_unordered(worker, cfgs):
            print(msg, flush=True)
    print("T1PAS DONE", flush=True)
