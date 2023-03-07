from typing import Optional, Union

import numpy as np
from matplotlib import pyplot as plt

from braandket import KetSpace
from braandket_synthesis import Controlled, QOperation, QubitsMatrixOperation, Remapped, Sequential, ToTensor
from braandket_synthesis_examples.utils.random_unitary import random_generate_unitary
from braandket_synthesis_examples.utils.to_code import ToCode
from braandket_synthesis_examples.utils.visualization import plot_matrix


class GivensRotationOperation(QubitsMatrixOperation):
    def __init__(self,
            n: int, row_a: int, row_b: int,
            a: Union[float, complex], b: Union[float, complex], *,
            name: Optional[str] = None
    ):
        if row_a == row_b:
            raise ValueError(f"row_a and row_b must be different, but got row_a={row_a} and row_b={row_b}")
        dtype = np.asarray([a, b]).dtype
        matrix = build_givens_rotation_matrix(2 ** n, row_a, row_b, a, b, dtype=dtype)
        super().__init__(matrix, name=name)
        self._row_a = row_a
        self._row_b = row_b
        self._a = a
        self._b = b

    @property
    def row_a(self) -> int:
        return self._row_a

    @property
    def row_b(self) -> int:
        return self._row_b

    @property
    def a(self) -> Union[float, complex]:
        return self._a

    @property
    def b(self) -> Union[float, complex]:
        return self._b


class FinalPhaseOperation(QubitsMatrixOperation):
    def __init__(self, n: int, p: Union[float, complex], *, name: Optional[str] = None):
        dtype = np.asarray(p).dtype
        matrix = np.eye(2 ** n, dtype=dtype)
        matrix[-1, -1] = p
        super().__init__(matrix, name=name)
        self._p = p

    @property
    def p(self) -> Union[float, complex]:
        return self._p


def extract_givens_factors(matrix: np.ndarray, col: int, row_a: int, row_b: int):
    a = matrix[row_a, col]
    b = matrix[row_b, col]
    c = np.sqrt(a * np.conj(a) + b * np.conj(b))

    a = a / c
    b = b / c
    return a, b


def build_givens_rotation_matrix(
        N: int, row_a: int, row_b: int,
        a: Union[float, complex], b: Union[float, complex],
        dtype=None
) -> np.ndarray:
    rotation = np.eye(N, dtype=dtype)
    rotation[row_a, row_a] = np.conj(a)
    rotation[row_a, row_b] = np.conj(b)
    rotation[row_b, row_a] = b
    rotation[row_b, row_b] = -a
    return rotation


def destruct_to_givens_rotations(matrix: np.ndarray):
    n = np.shape(matrix)[0]

    rotations = []
    for col in range(n - 1):
        row_a = col
        for row_b in range(row_a + 1, n):
            a, b = extract_givens_factors(matrix, col, row_a, row_b)
            rot_matrix = build_givens_rotation_matrix(n, row_a, row_b, a, b, matrix.dtype)
            matrix = rot_matrix @ matrix
            rotations.append(((row_a, row_b), (a, b)))
    phase = np.conj(matrix[-1, -1])
    return rotations, phase


def qrd_synth(matrix_op: QubitsMatrixOperation):
    n = matrix_op.n
    matrix = matrix_op.matrix

    rotations, phase = destruct_to_givens_rotations(matrix)

    steps = []
    for (row_a, row_b), (a, b) in rotations:
        steps.append(GivensRotationOperation(n, row_a, row_b, np.conj(a), b))  # conj: inverse
    steps.append(FinalPhaseOperation(n, np.conj(phase)))  # conj: inverse

    return Sequential(steps)


def final_phase_synth(operation: FinalPhaseOperation) -> QOperation:
    n = operation.n
    u = QubitsMatrixOperation(np.asarray([[1, 0], [0, operation.p]]))
    return Remapped(Controlled(u, (1,) * (n - 1)), lambda qs: (qs[:-1], qs[-1]))


if __name__ == '__main__':
    n = 2
    N = 2 ** n
    spaces = [KetSpace(2, name=f"q{i}") for i in range(n)]

    matrix = random_generate_unitary(N, dtype=np.float32)
    matrix_op = QubitsMatrixOperation(matrix)
    synth_op = qrd_synth(matrix_op)

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
