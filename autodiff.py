import math

class Scalar:
    def __init__(self, value, op='', children=()):
        self.value = value
        self.op = op
        self.children = children
        self.dot = 0.0

    def __repr__(self):
        return f'Scalar({self.value})'

    def __add__(self, other):
        return Scalar(self.value + other.value, op='+', children=(self, other))

    def __sub__(self, other):
        return Scalar(self.value - other.value, op='-', children=(self, other))

    def __mul__(self, other):
        return Scalar(self.value * other.value, op='*', children=(self, other))

    def __pow__(self, other):
        return Scalar(self.value ** other.value, op='**', children=(self, other))

    def __truediv__(self, other):
        return Scalar(self.value / other.value, op='/', children=(self, other))

    def autodiff(self, inner=False):
        if len(self.children) == 0:
            return
        if not inner:
            self.dot = 1.0

        if self.op == '+':
            self.children[0].dot += 1.0 * self.dot
            self.children[1].dot += 1.0 * self.dot
        elif self.op == '-':
            self.children[0].dot += 1.0 * self.dot
            self.children[1].dot += -1.0 * self.dot
        elif self.op == '*':
            self.children[0].dot += self.children[1].value * self.dot
            self.children[1].dot += self.children[0].value * self.dot
        elif self.op == '**':
            self.children[0].dot += self.children[1].value * self.children[0].value ** (self.children[1].value - 1) * self.dot
            self.children[1].dot += self.value * math.log(self.children[0].value) * self.dot
        elif self.op == '/':
            self.children[0].dot += self.dot / self.children[1].value
            self.children[1].dot += self.children[0].value * -1.0 * self.children[1].value ** -2.0

        self.children[0].autodiff(inner=True)
        self.children[1].autodiff(inner=True)
