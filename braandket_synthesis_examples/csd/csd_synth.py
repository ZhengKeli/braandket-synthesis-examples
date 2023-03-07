import itertools

import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import cossin

from braandket import KetSpace
from braandket_synthesis import Apply, Controlled, QOperation, QubitsMatrixOperation, Remapped, Ry, Sequential, ToTensor
from braandket_synthesis_examples.utils.random_unitary import random_generate_unitary
from braandket_synthesis_examples.utils.to_code import ToCode
from braandket_synthesis_examples.utils.visualization import plot_matrix


def csd(matrix_op: QubitsMatrixOperation) -> QOperation:
    n = matrix_op.n
    Nh = 2 ** (n - 1)

    u, cs, v = cossin(matrix_op.matrix, Nh, Nh, separate=False)
    u00 = u[:Nh, :Nh]
    u11 = u[Nh:, Nh:]
    v00 = v[:Nh, :Nh]
    v11 = v[Nh:, Nh:]

    u00_op = QubitsMatrixOperation(u00)
    u11_op = QubitsMatrixOperation(u11)
    u_op = Sequential([
        Remapped(Controlled(u00_op, 0), lambda sps: (sps[0], sps[1:])),
        Remapped(Controlled(u11_op, 1), lambda sps: (sps[0], sps[1:])),
    ])

    v00_op = QubitsMatrixOperation(v00)
    v11_op = QubitsMatrixOperation(v11)
    v_op = Sequential([
        Remapped(Controlled(v00_op, 0), lambda sps: (sps[0], sps[1:])),
        Remapped(Controlled(v11_op, 1), lambda sps: (sps[0], sps[1:])),
    ])

    cs_ops = []
    for i, (i_keys) in enumerate(itertools.product(*[(0, 1) for _ in range(n - 1)])):
        csi00 = cs[i, i]
        csi10 = cs[i + Nh, i]
        theta = np.arctan2(csi10, csi00) * 2
        csi_op = Ry(theta)
        csi_op = Remapped(Controlled(csi_op, i_keys), lambda sps: (sps[1:], sps[0]))
        cs_ops.append(csi_op)
    cs_op = Sequential(cs_ops)

    return Sequential(reversed([v_op, cs_op, u_op]))


def csd_recursive(op: QOperation) -> QOperation:
    if isinstance(op, QubitsMatrixOperation):
        if op.n > 1:
            return csd_recursive(csd(op))
        else:
            return op
    elif isinstance(op, Controlled):
        return Controlled(csd_recursive(op.bullet), op.keys)
    elif isinstance(op, Remapped):
        return Remapped(csd_recursive(op.original), op.mapping)
    elif isinstance(op, Sequential):
        return Sequential([csd_recursive(step) for step in op.steps])
    else:
        return op


def flatten(op: QOperation) -> QOperation:
    if isinstance(op, Sequential):
        steps = []
        for step in op.steps:
            step = flatten(step)
            if isinstance(step, Sequential):
                steps.extend(step.steps)
            else:
                steps.append(step)
        return Sequential(steps)
    elif isinstance(op, Remapped):
        base = flatten(op.original)
        if isinstance(base, Sequential):
            return flatten(Sequential([
                Remapped(step, op.mapping)
                for step in base.steps
            ]))
        elif isinstance(base, Remapped):
            mapping1 = op.mapping
            mapping2 = base.mapping
            return Remapped(base.original, lambda sps: mapping2(mapping1(sps)))
        else:
            return Remapped(base, op.mapping)
    elif isinstance(op, Controlled):
        base = flatten(op.bullet)
        if isinstance(base, Sequential):
            return flatten(Sequential([
                Controlled(step, op.keys)
                for step in base.steps
            ]))
        elif isinstance(base, Remapped):
            mapping = base.mapping
            return flatten(
                Remapped(Controlled(base.original, op.keys), lambda sps: (sps[0], mapping(sps[1]))))
        elif isinstance(base, Controlled):
            return flatten(Remapped(Controlled(base.bullet, (op.keys, base.keys)),
                lambda sps: ((sps[0], sps[1][0]), sps[1][1])))
        else:
            return Controlled(base, op.keys)
    else:
        return op


if __name__ == '__main__':
    n = 3
    N = 2 ** n
    spaces = [KetSpace(2, name=f"q{i}") for i in range(n)]

    matrix = random_generate_unitary(N, dtype=np.float32)
    matrix_op = QubitsMatrixOperation(matrix)
    synth_op = csd_recursive(matrix_op)
    # synth_op = flatten(csd_recursive(matrix_op))

    print(synth_op.trait(Apply))

    print("\nmatrix:")
    print(matrix_op.trait(ToCode)(spaces))
    print("\nsynth:")
    print(synth_op.trait(ToCode)(spaces))

    plt.figure(figsize=(7, 4))

    plt.subplot(1, 2, 1)
    plot_matrix(matrix)
    plt.title("matrix")

    plt.subplot(1, 2, 2)
    plot_matrix(synth_op.trait(ToTensor).to_tensor(spaces).flattened_values(spaces))
    plt.title("synth")

    plt.show()
