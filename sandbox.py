from autograd import Scalar

# create scalars
m = Scalar(4.0)
b = Scalar(6.0)
x = Scalar(2.0)

# use scalars to compute y
t = m * x
y = t + b

# compute derivatives of y wrt all the
# variables involved in computing y
y.autograd()

# derivative of a function wrt a variable
# will be stored in that variable's .grad
print(x.grad)
