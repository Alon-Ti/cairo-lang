import dataclasses
from os import walk
from os.path import realpath
from typing import Dict, Mapping, SupportsInt, Tuple, Union

from starkware.python.utils import Endianness

RELOCATABLE_OFFSET_LOWER_BOUND = -(2**63)
RELOCATABLE_OFFSET_UPPER_BOUND = 2**63

DEFAULT_PRIME = 2**31 - 1

def inv(x, prime=DEFAULT_PRIME):
    return pow(x, prime - 2, prime)

@dataclasses.dataclass
class CM31:
    """
    Represents a complex element a + bi in the Field F_(2^31-1)(i) where i^2 = -1.
    """

    a: int
    b: int

    def __add__(self, other):
        return CM31((self.a + other.a) % DEFAULT_PRIME, (self.b + other.b) % DEFAULT_PRIME)

    def __sub__(self, other):
        return CM31((self.a - other.a) % DEFAULT_PRIME, (self.b - other.b) % DEFAULT_PRIME)

    def __mul__(self, other):
        return CM31(
            (self.a * other.a - self.b * other.b) % DEFAULT_PRIME,
            (self.a * other.b + self.b * other.a) % DEFAULT_PRIME,
        )

    def inv(self):
        inv_norm2 = inv(self.a ** 2 + self.b ** 2)
        return CM31((self.a * inv_norm2) % DEFAULT_PRIME, (-self.b * inv_norm2) % DEFAULT_PRIME)

    def __div__(self, other):
        if other == 0:
            raise ZeroDivisionError("Division by zero.")
        return self * other.inv(other)

    def __neg__(self):
        return CM31(-self.a, -self.b)

    def __hash__(self) -> int:
        return hash(("CM31", self.a, self.b))

    def to_tuple(self):
        return (self.a, self.b)

    def is_zero(self):
        return self.to_tuple() == (0, 0)

    def __eq__(self, other):
        if isinstance(other, CM31):
            return self.a == other.a and self.b == other.b
        elif isinstance(other, int):
            return self == CM31(other, 0)
        return False

    def __lt__(self, other):
        if self.a == other.a:
            return self.b < other.b
        return self.a < other.a

    def __str__(self):
        def unoneify(x):
            return "" if x == 1 else str(x)

        if self.a == 0 and self.b == 0:
            return "0"
        if self.a == 0:
            return f"{unoneify(self.b)}i" if self.b != 1 else "i"
        if self.b == 0:
            return str(self.a)
        return f"{self.a} + {unoneify(self.b)}i"

R = CM31(2, 1)

@dataclasses.dataclass
class QM31:
    """
    Represents an element (a + bi) + (c + di)u in the Field F_(2^31-1)(i, u)
    where i^2 = -1 and u^2 = 2 + i.
    """
    a:  CM31
    b:  CM31
    
    def __add__(self, other):
        if isinstance(other, int):
            other = QM31.from_int(other)
        return QM31(
            self.a + other.a, self.b + other.b
        )

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        if isinstance(other, int):
            other = QM31.from_int(other)
        return QM31(
            self.a - other.a, self.b - other.b
        )

    def __rsub__(self, other):
        if isinstance(other, int):
            other = QM31.from_int(other)
        return other - self

    def __mul__(self, other):
        if isinstance(other, int):
            other = QM31.from_int(other)
        return QM31(
            self.a * other.a + R * self.b * other.b,
            self.a * other.b + self.b * other.a
        )

    def __rmul__(self, other):
        return self * other

    def __div__(self, other):
        if isinstance(other, int):
            other = QM31.from_int(other)
        return self * other.inv()

    def __rdiv__(self, other):
        if isinstance(other, int):
            other = QM31.from_int(other)
        return other / self # type: ignore

    def __hash__(self) -> int:
        return hash(("QM31", self.a, self.b))

    def inv(self):
        b2 = self.b * self.b
        ib2 = CM31(-b2.b, b2.a)
        denom_inv = inv(b2 + b2 + ib2)
        return QM31(
            self.a * denom_inv, -self.b * denom_inv
        )

    def __neg__(self):
        return QM31(-self.a, -self.b)

    def to_tuple(self):
        return (self.a.a, self.a.b, self.b.a, self.b.b)

    @classmethod
    def from_tuple(cls, t):
        assert len(t) == 4
        return cls(CM31(t[0], t[1]), CM31(t[2], t[3]))

    @classmethod
    def from_ints(cls, *ints):
        return cls.from_tuple(ints)

    @classmethod
    def from_int(cls, x):
        return cls.from_ints(x, 0, 0, 0)

    def to_bytes(self, n_bytes, byte_order):
        all_bytes = b"".join(
            x.to_bytes(n_bytes, byte_order) for x in self.to_tuple()
        )
        assert set(all_bytes[n_bytes:]) == {0}, f"{self} doesn't fit in {n_bytes} bytes."
        return all_bytes[:n_bytes]

    @classmethod
    def from_bytes(cls, data, byte_order):
        n_bytes = 4
        return cls.from_tuple(
            tuple(
            int.from_bytes(data[i * n_bytes : (i + 1) * n_bytes], byte_order)
            for i in range(4)
            )
        )

    def is_zero(self):
        return self.to_tuple() == (0, 0, 0, 0)

    def __lt__(self, other):
        if isinstance(other, int):
            other = QM31.from_int(other)
        if self.a == other.a:
            return self.b < other.b
        return self.a < other.a

    def __str__(self):
        def parenthify(expr):
            expr = str(expr)
            if "+" in expr:
                return f"({expr})"
            return expr

        def unoneify(x):
            return "" if x == 1 else str(x)

        if self.a.is_zero() and self.b.is_zero():
            return "0"
        if self.a.is_zero():
            return f"{parenthify(unoneify(self.b))}u"
        if self.b.is_zero():
            return parenthify(self.a)
        return f"{parenthify(self.a)} + {parenthify(unoneify(self.b))}u"

    def __eq__(self, other):
        if isinstance(other, QM31):
            return self.a == other.a and self.b == other.b
        if isinstance(other, int):
            return self == QM31.from_int(other)
        return False

    def __format__(self, format_spec):
        return str(self).__format__(format_spec)


@dataclasses.dataclass(frozen=True)
class RelocatableValue:
    """
    A value in the cairo vm representing an address in some memory segment. This is meant to be
    replaced by a real memory address (field element) after the VM finished.
    """

    segment_index: int
    offset: int

    SEGMENT_BITS = 8
    OFFSET_BITS = 22

    def is_zero(self):
        return False

    def __add__(self, other: "MaybeRelocatable") -> "RelocatableValue":
        if isinstance(other, int):
            return RelocatableValue(self.segment_index, self.offset + other)
        assert not isinstance(
            other, RelocatableValue
        ), f"Cannot add two relocatable values: {self} + {other}."
        return NotImplemented

    def __radd__(self, other: "MaybeRelocatable") -> "RelocatableValue":
        return self + other

    def __sub__(self, other: "MaybeRelocatable") -> "MaybeRelocatable":
        if isinstance(other, int):
            return RelocatableValue(self.segment_index, self.offset - other)
        assert self.segment_index == other.segment_index, (
            "Can only subtract two relocatable values of the same segment "
            f"({self.segment_index} != {other.segment_index})."
        )
        return self.offset - other.offset

    def __mod__(self, other: int):
        return RelocatableValue(self.segment_index, self.offset % other)

    def __lt__(self, other: "MaybeRelocatable"):
        if isinstance(other, int):
            # Integers are considered smaller than all relocatable values.
            return False
        if not isinstance(other, RelocatableValue):
            return NotImplemented
        return (self.segment_index, self.offset) < (other.segment_index, other.offset)

    def __le__(self, other: "MaybeRelocatable"):
        return self < other or self == other

    def __ge__(self, other: "MaybeRelocatable"):
        return not (self < other)

    def __gt__(self, other: "MaybeRelocatable"):
        return not (self <= other)

    def __hash__(self):
        return hash((self.segment_index, self.offset))

    def __format__(self, format_spec):
        return f"{self.segment_index}:{self.offset}".__format__(format_spec)

    def __str__(self):
        return f"{self.segment_index}:{self.offset}"

    @staticmethod
    def to_bytes(value: "MaybeRelocatable", n_bytes: int, byte_order: Endianness) -> bytes:
        """
        Serializes RelocatableValue as:
        1bit |   SEGMENT_BITS |   OFFSET_BITS
        1    |     segment    |   offset
        Serializes int as
        1bit | num
        0    | num
        """
        if isinstance(value, QM31):
            return value.to_bytes(n_bytes, byte_order)
        assert n_bytes * 8 > value.SEGMENT_BITS + value.OFFSET_BITS
        num = 2 ** (8 * n_bytes - 1) + value.segment_index * 2**value.OFFSET_BITS + value.offset
        return num.to_bytes(n_bytes, byte_order)

    @classmethod
    def from_bytes(cls, data: bytes, byte_order: Endianness) -> "MaybeRelocatable":
        n_bytes = len(data)
        num = int.from_bytes(data, byte_order)
        if num & (2 ** (8 * n_bytes - 1)):
            offset = num & (2**cls.OFFSET_BITS - 1)
            segment_index = (num >> cls.OFFSET_BITS) & (2**cls.SEGMENT_BITS - 1)
            return RelocatableValue(segment_index, offset)
        return QM31.from_bytes(data, byte_order)

    @staticmethod
    def to_tuple(value: "MaybeRelocatable") -> Tuple[int, ...]:
        """
        Converts a MaybeRelocatable to a tuple (which can be used to serialize the value in JSON).
        """
        if isinstance(value, RelocatableValue):
            return (value.segment_index, value.offset)
        elif isinstance(value, QM31):
            return value.to_tuple()
        else:
            raise NotImplementedError(f"Expected MaybeRelocatable, got: {type(value).__name__}.")

    @staticmethod
    def to_felt_or_relocatable(value: Union["RelocatableValue", SupportsInt]) -> "MaybeRelocatable":
        """
        Converts to int unless value is RelocatableValue, otherwise return value as is.
        """
        if isinstance(value, RelocatableValue):
            return value
        return QM31.from_int(value)

    @classmethod
    def from_tuple(cls, value: Tuple[int, ...]) -> "MaybeRelocatable":
        """
        Converts a tuple to a MaybeRelocatable. See to_tuple().
        """
        if len(value) == 2:
            return RelocatableValue(*value)
        elif len(value) == 4:
            return QM31.from_tuple(value)
        else:
            raise NotImplementedError(f"Expected a tuple of size 1 or 2, got: {value}.")


MaybeRelocatable = Union[QM31, RelocatableValue]
MaybeRelocatableDict = Dict[MaybeRelocatable, MaybeRelocatable]


def relocate_value(
    value: MaybeRelocatable,
    segment_offsets: Mapping[int, MaybeRelocatable],
    prime: int,
    allow_missing_segments: bool = False,
) -> MaybeRelocatable:
    if isinstance(value, QM31):
        return value
    elif isinstance(value, RelocatableValue):
        segment_offset = segment_offsets.get(value.segment_index)
        if segment_offset is None:
            assert allow_missing_segments, f"""\
Failed to relocate {value} with allow_missing_segments = False.
segment_offsets={segment_offsets}.
"""
            return value  # type: ignore

        value = value.offset + segment_offset
        if isinstance(value, int):
            assert value < prime, "Value must be less than prime"
        return QM31.from_int(value)
    else:
        raise NotImplementedError("Not relocatable")
