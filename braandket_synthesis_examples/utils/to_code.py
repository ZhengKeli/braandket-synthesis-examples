import abc
from typing import Optional

from braandket_synthesis import Controlled, KetSpaces, MatrixOperation, Op, QOperation, QOperationTrait, QuantumGate, \
    Remapped, Sequential
from braandket_synthesis.utils import iter_structure, iter_zip_structures


class ToCode(QOperationTrait[Op], abc.ABC):
    def __call__(self, spaces: KetSpaces) -> str:
        return self.to_code(spaces)

    def to_code(self, spaces: KetSpaces) -> str:
        head_str = type(self.operation).__name__
        args_str = ", ".join(sp.name for sp in iter_structure(spaces))

        try:
            body_code = self.body_to_code(spaces)
            if body_code is None:
                body_str = ""
            else:
                body_str = "{\n\t" + body_code.replace("\n", "\n\t") + "\n}"
        except NotImplementedError:
            body_str = "{ ... }"

        return f"{head_str}({args_str}) {body_str}"

    def body_to_code(self, spaces: KetSpaces) -> Optional[str]:
        raise NotImplementedError


class QOperationToCode(ToCode[QOperation]):
    def body_to_code(self, spaces: KetSpaces) -> Optional[str]:
        return None


class MatrixOperationToCode(ToCode[MatrixOperation]):
    def body_to_code(self, spaces: KetSpaces) -> Optional[str]:
        return str(self.operation.matrix)


class NotedQubitsMatrixOperationToCode(ToCode[QuantumGate]):
    def body_to_code(self, spaces: KetSpaces) -> Optional[str]:
        return None


class SequentialOperationToCode(ToCode[Sequential]):
    def to_code(self, spaces: KetSpaces) -> str:
        body_code = "\n".join(op.trait(ToCode)(spaces) for op in self.operation.steps)
        return "sequential {\n\t" + body_code.replace("\n", "\n\t") + "\n}"


class RemappedOperationToCode(ToCode[Remapped]):
    def to_code(self, spaces: KetSpaces) -> str:
        return self.operation.original.trait(ToCode)(self.operation.mapping(spaces))


class ControlOperationToCode(ToCode[Controlled]):
    def to_code(self, spaces: KetSpaces) -> str:
        ctl_spaces, tgt_spaces = spaces
        ctl_str = ", ".join(f"{sp.name}={k}" for sp, k in iter_zip_structures(ctl_spaces, self.operation.keys))

        body_code = self.operation.bullet.trait(ToCode)(tgt_spaces)
        return f"controlled({ctl_str}) {body_code}"
