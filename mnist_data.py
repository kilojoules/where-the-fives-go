"""MNIST straight from the IDX files, with the same normalisation train.py used.

train.py originally loaded MNIST through torchvision. Not every environment has it (the GRAM
cluster venv does not), and the dependency buys nothing here: these are four fixed files in a
documented format. torchvision is still used when present, so existing setups are unchanged.
"""
from pathlib import Path

import torch

MNIST_URLS = [
    'https://ossci-datasets.s3.amazonaws.com/mnist/',
    'https://storage.googleapis.com/cvdf-datasets/mnist/',
]
MNIST_FILES = dict(xtr='train-images-idx3-ubyte.gz', ytr='train-labels-idx1-ubyte.gz',
                   xte='t10k-images-idx3-ubyte.gz', yte='t10k-labels-idx1-ubyte.gz')


def _idx(path):
    import gzip
    import numpy as np
    with gzip.open(path, 'rb') as f:
        magic = int.from_bytes(f.read(4), 'big')
        dims = [int.from_bytes(f.read(4), 'big') for _ in range(magic & 0xFF)]
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(dims)


def load_mnist(device, root=Path('data/MNIST/raw')):
    """Same normalisation as train.py: ToTensor then Normalize((0.1307,), (0.3081,))."""
    import urllib.request
    root.mkdir(parents=True, exist_ok=True)
    for key, fn in MNIST_FILES.items():
        dest = root / fn
        if dest.exists():
            continue
        for base in MNIST_URLS:
            try:
                urllib.request.urlretrieve(base + fn, dest)
                break
            except Exception:
                continue
        if not dest.exists():
            raise SystemExit(f'could not download {fn}; place it in {root} by hand')
    out = {}
    for key, fn in MNIST_FILES.items():
        arr = _idx(root / fn).copy()
        t = torch.from_numpy(arr)
        if key.startswith('x'):
            t = (t.float() / 255.0 - 0.1307) / 0.3081
            t = t.unsqueeze(1)                       # N,1,28,28 as the model expects
        else:
            t = t.long()
        out[key] = t.to(device)
    return out['xtr'], out['ytr'], out['xte'], out['yte']


def apply_label_noise(y_np, keep, kind, rate, seed):
    """Relabel training examples in place, returning y_np.

    Every removal route in this repo decides what to remove by reading the label, so the
    label is where noise enters. The three kinds:

      'fp5'     rate * n_5 random non-5 images relabeled '5'
      'fn5'     a `rate` fraction of true 5s relabeled uniformly to one of the nine others
      'swap57'  each 5 relabeled 7 and each 7 relabeled 5, independently with prob `rate`

    `keep` is the examples still in the training set. The realization depends only on
    (seed, kind, rate, keep), so every arm at one seed is noised identically.

    Call this BEFORE filtering: a filter keyed on the label is applied to the labels it is
    given, and the whole point is that it cannot see through a wrong one.
    """
    import numpy as np
    rng = np.random.RandomState(seed + 777)
    kept = np.where(keep)[0]
    if kind == 'fp5':
        n5 = (y_np[kept] == 5).sum()
        cand = kept[y_np[kept] != 5]
        pick = rng.choice(cand, size=int(round(rate * n5)), replace=False)
        y_np[pick] = 5
    elif kind == 'fn5':
        cand = kept[y_np[kept] == 5]
        pick = rng.choice(cand, size=int(round(rate * len(cand))), replace=False)
        wrong = rng.randint(0, 9, size=len(pick))
        y_np[pick] = np.where(wrong >= 5, wrong + 1, wrong)   # uniform over the nine non-5s
    elif kind == 'swap57':
        c5 = kept[y_np[kept] == 5]
        c7 = kept[y_np[kept] == 7]
        p5 = c5[rng.rand(len(c5)) < rate]
        p7 = c7[rng.rand(len(c7)) < rate]
        y_np[p5] = 7
        y_np[p7] = 5
    else:
        raise ValueError(kind)
    return y_np
