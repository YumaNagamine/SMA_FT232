# processing frames to extract 3D joint position in realtime
import os, sys
import cv2
from cv_angle_traking.modify_markers_angles_reader import modified_marker_pos

class Extract3DPosition(modified_marker_pos):
    def predetermine_colorrange(self): # for existing(target) video
        pass
    def main_process(self, frame):
        
        pass