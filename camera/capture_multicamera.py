import os,sys, time
import cv2
from control.CameraSetting import Camera

retry = 5
while retry > 0:
    cam1 = Camera(0, cv2.CAP_MSMF, cam_name='side')
    if cam1.isOpened():
        break
    cam1.release()
    time.sleep(1)
    retry -= 1

retry = 5
while retry > 0:
    cam2 = Camera(1, cv2.CAP_MSMF, cam_name='top')
    if cam2.isOpened():
        break
    cam2.release()
    time.sleep(1)
    retry -= 1

cam1.load_calibration('./CAL/cam/side/intrinsics_flat_side_20250527_204730.json')
cam2.load_calibration('./CAL/cam/top/intrinsics_flat_top_20250527_204845.json')

cam1.realtime()
cam2.realtime()

win = 'sideview / topview'
cv2.namedWindow(win, cv2.WINDOW_NORMAL)
cv2.resizeWindow(win, 1280, 600)

record_video = True

try:
    while True:
        ret1, frame1 = cam1.read()
        ret2, frame2 = cam2.read()
        processedf1 = cam1.process_frame(frame1)
        processedf2 = cam2.process_frame(frame2)
        # processedf1 = cv2.cvtColor(processedf1, cv2.COLOR_HSV2BGR)
        # processedf2 = cv2.cvtColor(processedf2, cv2.COLOR_HSV2BGR)

        if not ret1 or not ret2:
            continue
        
        if record_video:
            cam1.add_frame(processedf1)
            cam2.add_frame(processedf2)

        combined_frame = cv2.hconcat([processedf1, processedf2])
        # frame_to_show = cv2.cvtColor(combined_frame, cv2.COLOR_HSV2BGR)
        # cv2.imshow(win, combined_frame)
        cv2.imshow(win, combined_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break 

finally:
    print('Releasing cameras and closing windows...')
    # if record_video:
    #     cam1.save_video()
    #     cam2.save_video()
    # cam1.stop()
    # cam2.stop()
    print('Captured frames:')
    # print(f'Camera 1: {cam1.frame_count} frames')
    # print(f'Camera 2: {cam2.frame_count} frames')
    # print('Processed frames:')
    # print(f'Camera 1: {cam1.processed_frame_count} frames')
    # print(f'Camera 2: {cam2.processed_frame_count} frames')
    print('Frame shapes:')
    print(f'Camera 1: {frame1.shape}')
    print(f'Camera 2: {frame2.shape}')
    print(f'Camera 1: {processedf1.shape}')
    print(f'Camera 2: {processedf2.shape}')
    # print(frame1)
    # print(processedf1)

    cam1.release()
    time.sleep(1)
    cam2.release()
    time.sleep(1)
    cv2.destroyAllWindows()
    cam1.finalize()
    cam2.finalize()
    print('Camera capture finished.')
