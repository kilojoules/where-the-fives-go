import multiprocessing as mp
from train import Config
from run_all import worker

if __name__ == "__main__":
    cfgs = [
        Config(name="probe_e10_pas05_pcr02", seed=0, mode="gram", epochs=10, p_as=0.5, p_cr=0.2),
        Config(name="probe_e10_pas05_pcr08", seed=0, mode="gram", epochs=10, p_as=0.5, p_cr=0.8),
        Config(name="probe_e15_pas05", seed=0, mode="gram", epochs=15, p_as=0.5),
        Config(name="probe_e10_pas05_s1", seed=1, mode="gram", epochs=10, p_as=0.5),
    ]
    with mp.get_context("spawn").Pool(4) as pool:
        for msg in pool.imap_unordered(worker, cfgs):
            print(msg, flush=True)
    print("PROBE2 DONE", flush=True)
