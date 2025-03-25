# This file is for testing control process created by Nagemine on March 5th
# NN + fuzzy feedback
import os, sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import numpy as np
from cv_angle_traking.angles_reader_joint_estimation import AngleTracker
import control.membership_functions as mf
from FuzzyFeedback import FuzzyFeedback
from network import NeuralNetwork
from control.CameraSetting import Camera