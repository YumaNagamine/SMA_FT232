import numpy as np
import matplotlib.pyplot as plt

def LinearFunc_coef(r0, r1):
    # r should be [x,y]
    a = (r1[1]-r0[1])/(r1[0]-r0[0])
    b = (r1[0]*r0[1] - r0[0]*r1[1])/(r1[0] - r0[0])
    return [a,b]

def triangle_func(x, leftroot, peak, rightroot):
    # root, peak = [x,y] 
    if x <= leftroot[0]:
        y = leftroot[1]
    elif leftroot[0]< x < peak[0] :
        [a,b] = LinearFunc_coef(leftroot, peak)
        y = a*x + b
    elif peak[0] <= x < rightroot[0]:
        [a,b] = LinearFunc_coef(peak, rightroot)
        y = a*x + b
    elif x >= rightroot[0]:
        y = rightroot[1]
    return y

def slope_func(x, left, right):
    if x <= left[0]:
        y = left[1]
    elif left[0]< x < right[0]:
        [a,b] = LinearFunc_coef(left, right)
        y = a*x + b
    elif x >= right[0]:
        y = right[1]
    return y