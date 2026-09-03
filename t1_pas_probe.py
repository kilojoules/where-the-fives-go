import json, numpy as np, torch
import torch.nn.functional as F
import track1_fashion as t1
from analyze_tracks import train_probe_head

torch.set_num_threads(4)
xtr, ytr, xte, yte = t1.load_data()
fas = yte >= 10
for name in ["t1_gram_pas01", "t1_gram_pas0"]:
    fa = []
    for s in [0, 1, 2]:
        model = t1.Net(); model.load_state_dict(torch.load(f"results/{name}_seed{s}.pt")); model.eval()
        with torch.no_grad():
            ftr = torch.cat([model.features(xtr[i:i+2048], (0,)) for i in range(0, len(xtr), 2048)])
            fte = torch.cat([model.features(xte[i:i+2048], (0,)) for i in range(0, len(xte), 2048)])
        head = train_probe_head(ftr, ytr, 20, s)
        with torch.no_grad():
            pred = head(fte).argmax(1)
        fa.append((pred[fas] == yte[fas]).float().mean().item())
        print(f"{name} s{s}: probe fashion {fa[-1]:.4f}", flush=True)
    print(f"{name}: probe fashion {100*np.mean(fa):.2f}±{100*np.std(fa):.2f}", flush=True)
print("PASPROBE DONE", flush=True)
