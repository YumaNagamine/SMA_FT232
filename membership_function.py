import numpy as np
import matplotlib.pyplot as plt

def LinearFunc_coef(r0, r1):
    # r should be [x,y]
    a = (r1[1]-r0[1])/(r1[0]-r0[0])
    b = (r1[0]*r0[1] - r0[0]*r1[1])/(r1[0] - r0[0])
    return [a,b]

# triangle_func is no longer necessary?
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
    else:print('Value not found!')
    return y

def triangle_func_np(x, leftroot, peak, rightroot):

    a1, b1 = LinearFunc_coef(leftroot, peak)
    a2, b2 = LinearFunc_coef(peak, rightroot)

    mask1 = (x >= leftroot[0]) & (x < peak[0])
    mask2 = (x >= peak[0]) & (x <= rightroot[0])

    y = np.zeros_like(x)
    y[mask1] = a1 * x[mask1] + b1
    y[mask2] = a2 * x[mask2] + b2
    y[x < leftroot[0]] = leftroot[1]
    y[x > rightroot[0]] = rightroot[1]

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

def slope_func_np(x, left, right):
    
    a, b = LinearFunc_coef(left, right)  

    y = np.where(
        (x <= left[0]), left[1],
        np.where(
            (x >= right[0]), right[1],
            a * x + b
        )
    )

    return y

def normal_three_membership(x, tri_param, up_param, down_param): # consist of downhill, triangle, uphill
    # tri_param = [[][][]]
    # up_param, down_param = [[][]]
    y_tri = triangle_func(x, tri_param[0],tri_param[1],tri_param[2])
    y_up = slope_func(x, up_param[0],up_param[1])
    y_down = slope_func(x, down_param[0], down_param[1])
    y = np.array([y_tri, y_up, y_down])
    return y

def calc_centroid(x, y0, y1, y2, dx): # y is array
    y = np.maximum.reduce([y0, y1, y2])
    num = np.sum(x*y*dx)
    den = np.sum(y*dx)
    centroid = num/den
    return centroid

def weighting(weights, membership_degree):
    weights = np.array(weights)
    membership_degree = np.array(membership_degree)
    # diag = np.diag(weights@membership_degree)
    # den = np.sum(diag)
    mem = weights@membership_degree
    den = np.sum(mem)
    # new_membership_degree = diag / den
    new_membership_degree = mem/den

    return new_membership_degree

def get_processed_membershipfunc(x, param, membership_degree, order):
    y0 = triangle_func_np(x, param[0][0], param[0][1],param[0][2]) #left : du < 0 
    y1 = triangle_func_np(x, param[1][0], param[1][1],param[1][2]) #middle: du ~ 0
    y2 = triangle_func_np(x, param[2][0], param[2][1],param[2][2]) #right: du > 0

    # y0 = np.minimum(membership_degree[1],y0)
    # y1 = np.minimum(membership_degree[0],y1)
    # y2 = np.minimum(membership_degree[2],y2)

    for index, num in enumerate(order):
        if index == 0:
            y0 = np.minimum(membership_degree[num],y0)
        elif index == 1:
            y1 = np.minimum(membership_degree[num],y1)
        elif index == 2:
            y2 = np.minimum(membership_degree[num],y2)

    y = np.vstack((y0, y1, y2))

    return y


def test(x):
    y = - x**2 - 2*x + 3
    return y

if __name__ == '__main__':
    x = np.linspace(-1, 1, 18000)
    y = triangle_func_np(x, [-0.5, 0], [-0.25,1], [0,0])
    # y = slope_func_np(x, [-90,1],[0,0])
    plt.plot(x,y)
    plt.show()
    pass
