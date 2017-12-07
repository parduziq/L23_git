import numpy as np

def swap(x, noSwaps=1):
    assert (noSwaps <= len(x))
    idx = np.random.choice(len(x), noSwaps, replace=False)
    y = np.copy(x)
    init_idx = np.copy(idx)
    np.random.shuffle(idx)

    for i in range(noSwaps):
        y[idx[i]] = x[init_idx[i]]
    return y
