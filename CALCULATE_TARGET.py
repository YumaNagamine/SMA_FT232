import numpy as np
import cv2

class CALC_TARGERT:
    def __init__(self, frame):
        self.frame = frame
    
    def OnMouse(self, event, x, y, flags, param):
        key = cv2.waitKey(0)

        if event == cv2.EVENT_LBUTTONDOWN:
            self.fingertip = [x,y]
        if event == cv2.EVENT_RBUTTONDOWN:
            self.MP_joint = [x,y]
        if key == 27:
            return
            
        
    def calc_target(self):
        cv2.namedWindow("choose target", cv2.WINDOW_GUI_EXPANDED)
        cv2.MouseCallback("choose target", self.OnMouse)

        distance = np.sqrt((self.fingertip[0] - self.MP_joint[0])**2 
                           + (self.fingertip[1] - self.MP_joint[1])**2)
        
        