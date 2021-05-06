from sympy import *
x = Symbol('x')
y = x ** 100 - 3 * x ** 25 #(原始式子)
c = x**3 - x**2 - x**1 + 1 # (待除式子)
p = Poly(y, x)
#res = simplify(y / c)
N = 98 # （N 次 除法）
tmp = y
i = 0
print(p.as_dict())
m = 0
while i < N:
    p = Poly(tmp, x)
    tmp = tmp - ((p.as_dict()[(p.degree(),)] * x**(p.degree() - 3)) * c)
    m += (p.as_dict()[(p.degree(),)] * x**(p.degree() - 3))
    i += 1
    print(simplify(tmp), p.degree())
print(simplify(tmp))
print(simplify(m))
print(simplify(m * c + tmp))