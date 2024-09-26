import cv2 
import matplotlib.pyplot as plt
import numpy as np
import time

from camera.NOGUI_ASYNCSAVER_with_ANGLESREADER import AsyncVideoSaver, AngleTracker
from collections import deque
from testFuzzy import FUZZY_CONTROL



target = []
for i in range(3):
    print("angle_", i , ':', sep = '', end = '')
    angle = int(input())
    target.append(angle)

target = np.array(target)

cam_num = 0

is_lightning = True
is_recod_video = True
cam_name = 'AR0234'

cap = cv2.VideoCapture(cam_num, cv2.CAP_DSHOW)
if cam_name == 'AR0234':
    target_fps = 90
    resolution = (1600,1200)
    width, height = resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

    cap.set(cv2.CAP_PROP_FPS, target_fps)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)

    if is_lightning:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        cap.set(cv2.CAP_DROP_GAIN, 0)
        cap.set(cv2.CAP_PROP_EXPOSURE, -11)
    else: 
        cap.set(cv2.CAP_PROP_GAIN, 0)
        cap.set(cv2.CAP_PROP_EXPOSURE, -3)

    fourcc = 'X264'

actual_fps = cap.get(cv2.CAP_PROP_FPS)
print("Target FPS: {target_fps}, Actual FPS: {actual_fps}")
if fourcc == 'X264':
    video_file_name = 'IMG/video/' +cam_name +'_' + time.strftime("%m%d-%H%M%S")  + '.mp4'


frame_id = 0
whether_first_frame = True

frame_times = deque(maxlen = 30)

cv_preview_wd_name = 'Video Preview'

cv2.nameWindow(cv_preview_wd_name, cv2.WINDOW_GUI_EXPANDED)
cv2.nameWindow("Mask", cv2.WINDOW_GUI_EXPANDED)

if is_recod_video : saver = AngleTracker(video_file_name, fourcc, target_fps, resolution, 'monocolor' )

while True:
    cur_time = time.perf_counter()
    ret, frame_raw = cap.read()

    if ret:
        if is_recod_video: saver.add_frame(frame_raw)

        if whether_first_frame:
            saver.acquire_marker_color()
            fuzzycontrol = FUZZY_CONTROL(target, firstangles?, False)
            whether_first_frame = False

        frame_id += 1
        frame_times.append(cur_time)

        if True: #read angles
            noneedframe, angle0, angle1, angle2 = saver.extract_angle(False)
            if angle0 > 180: angle0 = 360 - angle0

            if angle1 > 180: angle1 = 360 - angle1
            angles = [angle0, angle1, angle2]
            process_share_dict = angles
            print("angle: ", angles)
            cv2.imshow(cv_preview_wd_name, saver.frame)
        
        if True: # Fuzzy control 
            du = fuzzycontrol.Fuzzy_main(process_share_dict)

        if True:
            if frame_id > 45:
                cur_fps  = 30 / (cur_time - frame_times[0])
            else:
                cur_fps = -1
            cv2.putText(frame_raw, f'Time: {time.strftime("%Y%m%d-%H$M%S")},{cur_time}',
                            (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA) 
            cv2.putText(frame_raw, f'Current Frame {frame_id}; FPS: {int(cur_fps)}',
                            (10,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)


    else:
        print("cannot read video")
        break

    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
if is_recod_video: saver.finalize()
