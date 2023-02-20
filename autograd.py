import math

class Scalar:
    def __init__(self, value, op='', children=()):
        self.value = value
        self.op = op
        self.children = children
        self.grad = None

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

    def autograd(self):
        if len(self.children) == 0:
            return
        if self.grad is None:
            self.grad = 1.0

        if self.op == '+':
            self.children[0].grad = 1.0 * self.grad
            self.children[1].grad = 1.0 * self.grad
        elif self.op == '-':
            self.children[0].grad = 1.0 * self.grad
            self.children[1].grad = -1.0 * self.grad
        elif self.op == '*':
            self.children[0].grad = self.children[1].value * self.grad
            self.children[1].grad = self.children[0].value * self.grad
        elif self.op == '**':
            self.children[0].grad = self.children[1].value * self.children[0].value ** (self.children[1].value - 1) * self.grad
            self.children[1].grad = self.value * math.log(self.children[0].value) * self.grad
        elif self.op == '/':
            self.children[0].grad = self.grad / self.children[1].value
            self.children[1].grad = self.children[0].value * -1.0 * self.children[1].value ** -2.0

        self.children[0].autograd()
        self.children[1].autograd()
