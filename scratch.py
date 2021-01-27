
class A(object):
    x = 1

class B(A):
    pass

class C(A):
    pass



x = 10
def foo():
    x += 1
    print(x)

lst = [1,2,3]
def foo1():
    lst.append(5)


def foo2():
    lst += [5]


odd = lambda x : bool(x % 2)
numbers = [n for n in range(10)]

