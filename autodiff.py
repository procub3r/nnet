import math

class Scalar:
    def __init__(self, value, op='', children=()):
        self.value = value
        self.op = op
        self.children = children
        self.__dot = 0.0

    def __repr__(self):
        return f'Scalar({self.value})'

    @property
    def dot(self):
        return self.__dot

    def __add__(self, other):
        self.__dot = other.__dot = 0.0
        return Scalar(self.value + other.value, op='+', children=(self, other))

    def __sub__(self, other):
        self.__dot = other.__dot = 0.0
        return Scalar(self.value - other.value, op='-', children=(self, other))

    def __mul__(self, other):
        self.__dot = other.__dot = 0.0
        return Scalar(self.value * other.value, op='*', children=(self, other))

    def __pow__(self, other):
        self.__dot = other.__dot = 0.0
        return Scalar(self.value ** other.value, op='**', children=(self, other))

    def __truediv__(self, other):
        self.__dot = other.__dot = 0.0
        return Scalar(self.value / other.value, op='/', children=(self, other))

    def autodiff(self):
        self.__dot = 1.0
        self.__autodiff_rec()

    def __autodiff_rec(self):
        if len(self.children) == 0:
            return

        if self.op == '+':
            self.children[0].__dot += 1.0 * self.__dot
            self.children[1].__dot += 1.0 * self.__dot
        elif self.op == '-':
            self.children[0].__dot += 1.0 * self.__dot
            self.children[1].__dot += -1.0 * self.__dot
        elif self.op == '*':
            self.children[0].__dot += self.children[1].value * self.__dot
            self.children[1].__dot += self.children[0].value * self.__dot
        elif self.op == '**':
            self.children[0].__dot += self.children[1].value * self.children[0].value ** (self.children[1].value - 1) * self.__dot
            self.children[1].__dot += self.value * math.log(self.children[0].value) * self.__dot
        elif self.op == '/':
            self.children[0].__dot += self.__dot / self.children[1].value
            self.children[1].__dot += self.children[0].value * -1.0 * self.children[1].value ** -2.0

        for child in self.children:
            child.__autodiff_rec()
