import numpy as np
from matplotlib import pyplot as plt

from braandket import KetSpace
from braandket_synthesis import CNOT, Controlled, KetSpaces, QOperation, Remapped, Sequential, ToTensor
from braandket_synthesis_examples.qrd.qrd_synth import GivensRotationOperation
from braandket_synthesis_examples.qrd.utils import to_binary
from braandket_synthesis_examples.utils.to_code import ToCode
from braandket_synthesis_examples.utils.visualization import plot_matrix


def without_kth(items, k: int):
    if k < len(items):
        return *items[:k], *items[k + 1:]
    else:
        return items


def ctl_tgt(c, t):
    def m(qs: KetSpaces):
        qs: list[KetSpace]
        return qs[c], qs[t]

    return m


def givens_rotation_synth(operation: GivensRotationOperation) -> QOperation:
    if operation.n == 1:
        return operation

    n = operation.n
    a_bits = to_binary(n, operation.row_a)
    b_bits = to_binary(n, operation.row_b)
    print(f"a_bits={a_bits}")
    print(f"b_bits={b_bits}")

    # 先找一个不同的 bit （第 k 位）
    k = None
    for k in range(n):
        if a_bits[k] != b_bits[k]:
            break
    assert k is not None
    print(f"k={k}")

    # 以 k 为 control，把其他不同的 bit 都变相同
    ad_steps = []
    for l in range(k + 1, n):
        if a_bits[l] != b_bits[l]:
            ad_steps.append(Remapped(CNOT(), ctl_tgt(k, l)))

    # 把相同的 bits 都用作 control
    if a_bits[k] == 0:
        c_bits = without_kth(a_bits, k)
    else:
        c_bits = without_kth(b_bits, k)
    u_op = GivensRotationOperation(1, 0, 1, operation.a, operation.b)
    cu_op = Remapped(Controlled(u_op, c_bits), lambda qs: (without_kth(qs, k), qs[k]))

    if len(ad_steps) > 0:
        print(f"found p>1 p={len(ad_steps) + 1}")
        if len(ad_steps) > 1:
            ad_op = Sequential(ad_steps)
            rs_op = Sequential(reversed(ad_steps))
        else:
            ad_op = ad_steps[0]
            rs_op = ad_steps[0]
        return Sequential([ad_op, cu_op, rs_op])
    else:
        return cu_op


if __name__ == '__main__':
    n = 3
    row_a, row_b = 3, 6
    a, b = np.random.uniform(-1, 1, size=2)

    grot_op = GivensRotationOperation(n, row_a, row_b, a, b)
    synth_op = givens_rotation_synth(grot_op)

    spaces = [KetSpace(2, name=f"q{i}") for i in range(n)]

    print("\ngrot:")
    print(grot_op.trait(ToCode)(spaces))
    print("\nsynth:")
    print(synth_op.trait(ToCode)(spaces))

    plt.figure(figsize=(7, 4))

    plt.subplot(1, 2, 1)
    plot_matrix(grot_op.trait(ToTensor)(spaces).flattened_values(spaces))
    plt.title("grot")

    plt.subplot(1, 2, 2)
    plot_matrix(synth_op.trait(ToTensor)(spaces).flattened_values(spaces))
    plt.title("synth")

    plt.show()
