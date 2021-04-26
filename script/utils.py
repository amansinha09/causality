#utils.py

import math

def k(s,f):
    s = len(s)
    f = len(f)
    return math.factorial(s)* math.factorial(f-s-1) / math.factorial(f)

