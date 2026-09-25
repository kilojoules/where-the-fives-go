from track5_stories import Config, worker
print(worker(Config(name="t5pilot_dense", seed=0, mode="dense")), flush=True)
print("T5 PILOT DONE", flush=True)
