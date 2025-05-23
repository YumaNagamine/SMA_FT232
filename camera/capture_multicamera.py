import os,sys, time
import cv2
from control.CameraSetting import Camera

retry = 5
while retry > 0:
    cam1 = Camera(0, cv2.CAP_DSHOW, cam_name='side')
    if cam1.isOpened():
        break
    cam1.release()
    time.sleep(1)
    retry -= 1

retry = 5
while retry > 0:
    cam2 = Camera(1, cv2.CAP_DSHOW, cam_name='top')
    if cam2.isOpened():
        break
    cam2.release()
    time.sleep(1)
    retry -= 1

cam1.load_calibration('./CAL/cam/side/intrinsics_flat_side_20250519_165804.json')
cam2.load_calibration('./CAL/cam/top/intrinsics_flat_side_20250519_165804.json')

cam1.realtime()
cam2.realtime()

win = 'sideview / topview'
cv2.namedWindow(win, cv2.WINDOW_NORMAL)

record_video = True

try:
    while True:
        ret1, frame1 = cam1.read()
        ret2, frame2 = cam2.read()

        if not ret1 or not ret2:
            continue
        
        if record_video:
            cam1.add_frame(frame1)
            cam2.add_frame(frame2)

        combined_frame = cv2.hconcat([frame1, frame2])
        cv2.imshow(win, combined_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break 

finally:
    cam1.release()
    time.sleep(1)
    cam2.release()
    time.sleep(1)
    cv2.destroyAllWindows()
    cam1.finalize()
    cam2.finalize()
