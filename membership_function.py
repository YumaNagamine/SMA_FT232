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

def seven_memdegree(x, param): # for former part of if-then rules. Return value of NB, NM, NS, ZE, PS, PM, PB
    # param should be [center of PS, center of PM, center of PB]
    value = np.zeros(7, dtype=np.float64)
    value[0] = slope_func_np(x, [-param[2],1], [-param[1],0])
    value[1] = triangle_func_np(x, [-param[2],0], [-param[1],1], [-param[0],0])
    value[2] = triangle_func_np(x, [-param[1],0], [-param[0],1], [0,0])
    value[3] = triangle_func_np(x, [-param[0],0], [0,1], [param[0],0])
    value[4] = triangle_func_np(x, [0,0], [param[0],1], [param[1],0])
    value[5] = triangle_func_np(x, [param[0],0], [param[1],1], [param[2],0])
    value[6] = slope_func_np(x, [param[1],0], [param[2],1])

    return value

def calc_centroid(x, y0, y1, y2, dx): # y is array
    y = np.maximum.reduce([y0, y1, y2])
    num = np.sum(x*y*dx)
    den = np.sum(y*dx)
    if den !=0:
        centroid = num/den
        return centroid
    else:
        return 0

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

def get_processed_membershipfunc_seven(x, param, membership_degree, order):
    # param should be [center of NB, NM, NS, ZE, PS, PM, PB]
    distance = param[1]-param[0]
    y0 = triangle_func_np(x, [param[0]-distance, 0], [param[0], 1], [param[1], 0]) # NB 
    y1 = triangle_func_np(x, [param[0], 0], [param[1], 1], [param[2], 0]) # NM
    y2 = triangle_func_np(x, [param[1], 0], [param[2], 1], [param[3], 0]) # NS
    y3 = triangle_func_np(x, [param[2], 0], [param[3], 1], [param[4], 0]) # ZE
    y4 = triangle_func_np(x, [param[3], 0], [param[4], 1], [param[5], 0]) # PS
    y5 = triangle_func_np(x, [param[4], 0], [param[5], 1], [param[6], 0]) # PM
    y6 = triangle_func_np(x, [param[5], 0], [param[6], 1], [param[6]+distance, 0]) # PB
    zeros = np.zeros(len(x))
    temp_list = []
    for index, num in enumerate(order):
        if index == 0:
            y0 = np.minimum(membership_degree[num],y0)
            if not np.all(y0 == 0): 
                temp_list.append(y0)
        elif index == 1:
            y1 = np.minimum(membership_degree[num],y1)
            if not np.all(y1 == 0): 
                temp_list.append(y1)
        elif index == 2:
            y2 = np.minimum(membership_degree[num],y2)
            if not np.all(y2 == 0): 
                temp_list.append(y2)
        elif index == 3:
            y3 = np.minimum(membership_degree[num],y3)
            if not np.all(y3 == 0): 
                temp_list.append(y3)
        elif index == 4:
            y4 = np.minimum(membership_degree[num],y4)
            if not np.all(y4 == 0): 
                temp_list.append(y4)
        elif index == 5:
            y5 = np.minimum(membership_degree[num],y5)
            if not np.all(y5 == 0): 
                temp_list.append(y5)
        elif index == 6:
            y6 = np.minimum(membership_degree[num],y6)
            if not np.all(y6 == 0): 
                temp_list.append(y6)
    
    if len(temp_list) < 3:
        for _ in range(3-len(temp_list)):
            temp_list.append(zeros)
    y = np.vstack((temp_list[0], temp_list[1], temp_list[2]))

    return y



def target_function(t, initial_target, target0=[],target1=[], target2=[],target3=[]):
    # target should be [time, target angle]
    
    if target0 == [] or t < target0[0]: return initial_target
    elif target1 == [] or t < target1[0]: return target0[1]
    elif target2 == [] or t < target2[0]: return target1[1]
    elif target3 == [] or t < target3[0]: return target2[1]
    else: return target3[1]

def test(x):
    y = - x**2 - 2*x + 3
    return y

if __name__ == '__main__':
    # x = np.linspace(-1, 1, 18000)
    # y = triangle_func_np(x, [-0.5, 0], [-0.25,1], [0,0])
    # # y = slope_func_np(x, [-90,1],[0,0])
    # plt.plot(x,y)
    # plt.show()
    # pass
    import time
    f = time.time()
    while True:
        t = time.time()-f 
        target = target_function(t, 10, [5,110],[10,200])
        print(t, "s")
        print(target)
        time.sleep(0.4)

