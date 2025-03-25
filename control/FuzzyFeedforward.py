# This class is for feedforward control created by Nagemine on March 24th 

import os, sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import numpy as np
from cv_angle_traking.angles_reader_joint_estimation import AngleTracker
import control.membership_functions as mf

class FuzzyFeedforward():
    def __init__(self):
        pass