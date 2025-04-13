import os
os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin")
import cv2
print(cv2.getBuildInformation())