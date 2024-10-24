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

def normal_three_membership(x, tri_param, up_param, down_param): # consist of downhill, triangle, uphill
    # tri_param = [[][][]]
    # up_param, down_param = [[][]]
    y_tri = triangle_func(x, tri_param[0],tri_param[1],tri_param[2])
    y_up = slope_func(x, up_param[0],up_param[1])
    y_down = slope_func(x, down_param[0], down_param[1])
    y = np.array([y_tri, y_up, y_down])
    return y

def three_triangles(x, left_param, middle_param, right_param):
    pass
def calc_centroid(x, y0, y1, y2, dx): # y is array
    y = np.maximum.reduce([y0, y1, y2])
    num = np.sum(x*y*dx)
    den = np.sum(y*dx)
    centroid = num/den
    return centroid
def weighting(weights, membership_degree):
    weights = np.array(weights)
    membership_degree = np.array(membership_degree)

    diag = np.diag(weights@membership_degree)
    den = np.sum(diag)
    membership_degree = membership_degree / den

    return membership_degree
def test(x):
    y = - x**2 - 2*x + 3
    return y

if __name__ == '__main__':
    # x = np.linspace(-3,1,num = 100)
    # y = test(x)
 
    # center = calc_centroid(x, y, y, y)
    # print(center)

    x=10
    y = np.array([11,-11,12,1])
    y[1] = 1212
    print(y)
    pass
