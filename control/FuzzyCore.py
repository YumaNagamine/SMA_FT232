# This file is created by Nagemine to put functions that fuzzy processes have in common, on March 24th 

import os, sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import numpy as np
from cv_angle_traking.angles_reader_joint_estimation import AngleTracker

def triangle_function(x, a,b,c):
    x = np.array(x)
    y = np.zeros_like(x,dtype=float)
    mask1 = (x >= a) & (x <= b)
    y[mask1] = (x[mask1]-a)/(b-a)
    
    mask2 = (x > b) & (x <= c)
    y[mask2] = (c-x[mask2])/(c-b)
    
    return y
def ramp_up(x,a,b): # a<b
    return np.clip((x-a)/(b-a), 0, 1)

def ramp_down(x,a,b): # a<b
    return np.clip((b-x)/(b-a), 0, 1)

def and_operation(y0,y1):
    return np.minimum(y0, y1)

def or_operaiton(*y):
    stacked_y = np.vstack(y)
    return np.max(stacked_y, axis=0)

def area(x,y):
    return np.trapz(y,x)

def centroid_x(x,y):
    try:
        centroid = np.sum(x*y)/np.sum(y)
        return centroid
    except ZeroDivisionError as e:
        return 0
    
def triangles(num, *centers):
    for i in range(num):
        pass