"""Quick probe: close the full-profile gap on aux digits (recall5/7 vs dense)."""
import multiprocessing as mp
from train import Config
from run_all import worker

if __name__ == "__main__":
    cfgs = [
        Config(name="probe_e10", seed=0, mode="gram", epochs=10),
        Config(name="probe_e10_pcr02", seed=0, mode="gram", epochs=10, p_cr=0.2),
        Config(name="probe_e10_pas05", seed=0, mode="gram", epochs=10, p_as=0.5),
        Config(name="probe_pcr02", seed=0, mode="gram", p_cr=0.2),
    ]
    with mp.get_context("spawn").Pool(4) as pool:
        for msg in pool.imap_unordered(worker, cfgs):
            print(msg, flush=True)
    print("PROBE DONE", flush=True)
