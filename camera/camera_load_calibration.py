# -*- coding: utf-8 -*-
# camera_calibration.py NOT READYET!!!!

import json
import cv2
import numpy as np

class CameraCalibration:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = json.load(f)

    def get_params(self, cam_name: str):
        """Return (camera_matrix, dist_coeffs, morphology) for given cam."""
        cam = self.config.get(cam_name)
        if cam is None:
            raise KeyError(f"No calibration for camera '{cam_name}'")
        K = np.array(cam['camera_matrix'], dtype=np.float64)
        D = np.array(cam['dist_coeffs'], dtype=np.float64)
        morph = cam['morphology']
        return K, D, morph

class CameraProcessor:
    def __init__(self, cam_index, cam_name, cali: CameraCalibration):
        # open camera
        self.cap = cv2.VideoCapture(cam_index)
        self.cam_name = cam_name
        self.K, self.D, self.morph = cali.get_params(cam_name)
        # precompute undistort maps
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            self.K, self.D, None, self.K, (w,h), cv2.CV_16SC2)

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        # 1. undistort
        frame = cv2.remap(frame, self.map1, self.map2, cv2.INTER_LINEAR)
        # 2. grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 3. blur
        blur = cv2.GaussianBlur(gray, (self.morph['blur_size'],)*2, 0)
        # 4. threshold to binary
        _, bin_img = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU)
        # 5. morphological open/close
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (self.morph['kernel_size'],)*2)
        morph = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, k)
        morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, k)
        return morph

    def release(self):
        self.cap.release()
def main():
    file_url = './CAL/cam/side/' 
    file_name = "intrinsics_side_20250519_014359.json"
    file_path = file_url + file_name
    cali = CameraCalibration(file_path)

    # side cam on index 0, top cam on index 1 (adjust as needed)
    side_proc = CameraProcessor(0, 'side', cali)
    top_proc  = CameraProcessor(1, 'top', cali)

    while True:
        side_m = side_proc.process_frame()
        top_m  = top_proc.process_frame()

        if side_m is None or top_m is None:
            break

        # display or process further
        cv2.imshow('side morph', side_m)
        cv2.imshow('top morph',  top_m)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    side_proc.release()
    top_proc.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
