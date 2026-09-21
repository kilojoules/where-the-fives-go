from track4_modfam import Config, worker
print(worker(Config(name="t4gp_gram5", seed=0, mode="gram", target_b=5, epochs=5000, log_every=1000)), flush=True)
print("T4 GPILOT DONE", flush=True)
