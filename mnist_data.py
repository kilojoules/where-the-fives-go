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
