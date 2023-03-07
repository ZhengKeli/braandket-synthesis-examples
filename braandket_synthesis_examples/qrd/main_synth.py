import numpy as np
from matplotlib import pyplot as plt

from braandket import KetSpace
from braandket_synthesis import QubitsMatrixOperation, Sequential, ToTensor
from braandket_synthesis_examples.csd.csd_synth import flatten
from braandket_synthesis_examples.qrd.givens_synth import givens_rotation_synth
from braandket_synthesis_examples.qrd.qrd_synth import final_phase_synth, qrd_synth
from braandket_synthesis_examples.utils.random_unitary import random_generate_unitary
from braandket_synthesis_examples.utils.to_code import ToCode
from braandket_synthesis_examples.utils.visualization import plot_matrix


def main_synth(matrix_op: QubitsMatrixOperation):
    synth_op = qrd_synth(matrix_op)
    synth_op = Sequential([
        *[givens_rotation_synth(step) for step in synth_op.steps[:-1]],
        final_phase_synth(synth_op.steps[-1])
    ])
    synth_op = flatten(synth_op)
    return synth_op


if __name__ == '__main__':
    n = 3
    N = 2 ** n
    spaces = [KetSpace(2, name=f"q{i}") for i in range(n)]

    matrix = random_generate_unitary(N, dtype=np.float32)
    matrix_op = QubitsMatrixOperation(matrix)
    synth_op = main_synth(matrix_op)

    print("\nmatrix:")
    print(matrix_op.trait(ToCode)(spaces))
    print("\nsynth:")
    print(synth_op.trait(ToCode)(spaces))

    plt.figure(figsize=(7, 4))

    plt.subplot(1, 2, 1)
    plot_matrix(matrix)
    plt.title("matrix")

    plt.subplot(1, 2, 2)
    plot_matrix(synth_op.trait(ToTensor)(spaces).flattened_values(spaces))
    plt.title("synth")

    plt.show()
