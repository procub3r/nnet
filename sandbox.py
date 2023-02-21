from autodiff import Scalar

# create scalars
a = Scalar(5.0)
b = Scalar(6.0)
c = Scalar(8.0)
x = Scalar(5.0)

# use scalars to compute y
y = a * x ** Scalar(2.0) + b * x + c

# compute derivatives of y wrt all the
# variables involved in computing y
y.autodiff()

# derivative of a function wrt a variable
# will be stored in that variable's .dot
print(x.dot)
