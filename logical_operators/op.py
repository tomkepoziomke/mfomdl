def Zero(a, b):
    return a * b * 0

def One(a, b):
    return Zero(a, b) + 1

def And(a, b):
    return One(a, b) * (a * b != 0)

def Nand(a, b):
    return 1 - And(a, b)

def Or(a, b):
    return One(a, b) * (a + b != 0)

def Nor(a, b):
    return 1 - Or(a, b)

def Xor(a, b):
    return One(a, b) * (a + b == 1)

def Xnor(a, b):
    return 1 - Xor(a, b)

def Right(a, b):
    return Zero(a, b) + b

def Left(a, b):
    return Zero(a, b) + a

def Nright(a, b):
    return 1 - Right(a, b)

def Nleft(a, b):
    return 1 - Left(a, b)

def Impl(a, b):
    return Or(1 - a, b)

def Nimpl(a, b):
    return And(a, 1 - b)

def Lpmi(a, b):
    return Or(a, 1 - b)

def Nlpmi(a, b):
    return And(1 - a, b)
