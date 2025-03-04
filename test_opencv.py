import sys,os
print(sys.version)
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,parentdir)

os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin")

import time,cv2

camera_index = 0
cap = cv2.VideoCapture(camera_index)

target_fps = 90
resolution =  (1600,1200)#(1920,1200)#q(800,600)# (800,600)#(1920,1200) (1280,720)#
width, height = resolution

cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0]) 
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
# Set FPS
cap.set(cv2.CAP_PROP_FPS,target_fps)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG')) # 'I420'
cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)  # 设置缓冲区大小为2


# Save video
fourcc = 'X264'#'MJPG' # 'I420' X264

if not cap.isOpened():
    print(f"Failed to open camera {camera_index}.")
else:
    print('Success!!')

# time.sleep(10)
while True:
    # time.sleep(10)
    try:
        ret, frame = cap.read()
        print(ret, frame)
    except KeyboardInterrupt:
        break

cap.release()