import numpy as np


def to_binary(n: int, x: int):
    bs = np.zeros(n, dtype=np.int32)
    for k in range(n):
        bs[k] = (x % 2) == 1
        x //= 2
    return bs[::-1]


if __name__ == '__main__':
    print(to_binary(4, 6))
    print(to_binary(4, 13))
