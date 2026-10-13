"""Track 7 driver: train the 27 shipped models (3 concepts x dense/filter/gram
x 3 seeds), then run the t7_eval battery. Remote helper.

  python t7_run.py          full run
  python t7_run.py smoke    tiny end-to-end check (one concept, short training)

smoke trains under the real tags, so it overwrites results/t7_sad_*_seed0.json
and .pt; run it in a scratch copy of the repo.
"""

import multiprocessing as mp
import sys

import track7_concepts as T


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "smoke":
        import torch
        import t7_eval as E
        for arm in ["dense", "filter", "gram"]:
            T.run(T.Config(concept="sad", arm=arm, seed=0, n_core_win=2000,
                           n_target_win=400, epochs=1))
        E.SEEDS, E.KS, E.STEPS, E.ALPHAS = [0], [50], 20, [-1.0, 0.0, 1.0]
        E.LRS = [1e-3]
        out = E.one(("sad", 0))
        print({k: v for k, v in out.items() if "probe" in k})
        print("SMOKE OK", flush=True)
        sys.exit()
    with mp.get_context("spawn").Pool(12) as pool:
        for msg in pool.imap_unordered(T.worker, T.all_configs()):
            print(msg, flush=True)
    print("TRAIN DONE", flush=True)
    import t7_eval
    t7_eval.main()
