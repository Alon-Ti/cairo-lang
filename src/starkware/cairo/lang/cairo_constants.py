import dataclasses

DEFAULT_PRIME = 2**31-1

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
        return QM31(
            self.a + other.a, self.b + other.b
        )

    def __sub__(self, other):
        return QM31(
            self.a - other.a, self.b - other.b
        )

    def __mul__(self, other):
        return QM31(
            self.a * other.a + R * self.b * other.b,
            self.a * other.b + self.b * other.a
        )

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
        return cls(CM31(t[0], t[1]), CM31(t[2], t[3]))
