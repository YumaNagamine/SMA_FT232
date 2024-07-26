import os
os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin")
#os.add_dll_directory(r"C:\OpenCV_Build4.10\install\x64\vc17\bin")
import cv2
print(cv2.getBuildInformation())