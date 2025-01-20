import sys,os
print(sys.version)
os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin")

import time,cv2

def find_available_cameras():
    available_cameras = []
    for i in range(100000):  
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    return available_cameras

cameras = find_available_cameras()
if cameras:
    print(f"Available cameras: {cameras}")
else:
    print("No cameras found.")