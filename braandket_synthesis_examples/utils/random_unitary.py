import numpy as np


def random_generate_basis(n: int, dtype=np.complex128):
    bs = []

    for i in range(n):
        # random generate
        if dtype.__name__.startswith('complex'):
            x = np.random.uniform(0, 1, [n]) + np.random.uniform(0, 1, [n]) * 1j
            x = np.cast[dtype](x)
        elif dtype.__name__.startswith('float'):
            x = np.random.uniform(0, 1, [n])
            x = np.cast[dtype](x)
        else:
            raise ValueError(f"Not supported dtype: {dtype}")

        # wash
        for b in bs:
            x -= b * np.sum(x * np.conj(b))

        # normalize
        x = x / np.sqrt(np.sum(x * np.conj(x)))

        bs.append(x)

    return bs


def random_generate_unitary(n: int, dtype=np.complex128):
    bs = random_generate_basis(n, dtype)
    m = np.asarray(bs)
    return m
