# This file is created by Nagamine on May 16
# to measure 3D-marker-positions from existing video
# Place the target video files on sc01/
import sys, os, time
os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin")

from control.CameraSetting import Camera
import cv2
from modify_markers_angles_reader import ModifiedMarkers
import numpy as np

if __name__ == "__main__":
    
    # window setting
    windowname = 'camera1/camera2'
    cv2.namedWindow(windowname, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(windowname, 1920, 600)
    windowname_choose = 'Choose' # ここをChooseにしないとクリックできないようになっているので，後で修正
    threshold_area_size = [200, 50, 50, 10]# [80, 20, 10, 40]
    output_video_fps = 30

    # create objects
    video_name1 = "test_side.mp4" # example
    video_name2 = "test_top.mp4"
    tracker_side = ModifiedMarkers(video_name1, threshold_area_size, 'monocolor')
    tracker_top = ModifiedMarkers(video_name2, threshold_area_size, 'monocolor')
    
    our_camera_setting = True
    if our_camera_setting:
        cap1 = Camera(tracker_side.video_path)
        cap2 = Camera(tracker_top.video_path)
        cap1.existingvideo(tracker_side.video_path, output_fps=output_video_fps)
        cap2.existingvideo(tracker_top.video_path, output_fps=output_video_fps)
    else:
        frame_shift = 0
        cap1 = cv2.VideoCapture(tracker_side.video_path)
        cap2 = cv2.VideoCapture(tracker_top.video_path)
        cap1.set(cv2.CAP_PROP_POS_FRAMES, frame_shift)
        cap1.set(cv2.CAP_PROP_FPS, output_video_fps)
        cap2.set(cv2.CAP_PROP_POS_FRAMES, frame_shift)
        cap2.set(cv2.CAP_PROP_FPS, output_video_fps)

    # cv2.namedWindow('Video Preview', cv2.WINDOW_GUI_EXPANDED)
    # cv2.namedWindow('Mask', cv2.WINDOW_GUI_EXPANDED)


    if not cap1.isOpened() or not cap2.isOpened():
        print('cannot open the camera')
        sys.exit()

    # parameters
    theta = 0.55
    distance = -30
    theta_top = 0.20
    tracker_side.set_params(theta, distance)
    tracker_top.set_params(theta_top=theta_top)


    # constants
    colors = [(43,74,134), (0,0,255), (255,0,0), (0,255,255)]
    line_padding = [0.7, 1.5,1.5,1.5]
    font_scale = 1
    text_position_cnt = (100, 100)
    text_position_time = (100, 120)
    
    frame_jump = 0    
    frame_shift = 0
    frame_id = 0
    basepoint = np.array([830, 600])
   
    cv2.namedWindow('Mask Preview', cv2.WINDOW_NORMAL)
    # to store
    measure = []
    frames_to_store_side = []
    frames_to_store_top = []

    is_record = True
    
    try:
        while True:
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()

            if not ret1 or not ret2:
                print('Missed the frame!')
                # continue
                break

            ########################################

            # image processing
            if frame_id == frame_shift: 
                tracker_side.acquire_marker_color(frame1, windowname_choose)
                tracker_top.fingertip_range = tracker_side.fingertip_range
            frame_id += 1


            try:
                frame1, angle0, angle1, angle2, raw_marker_pos, modified_marker_pos = tracker_side.extract_angle_improved(frame1, colors, modify=True)
                # frame1, angle0, angle1, angle2, raw_marker_pos, modified_marker_pos = tracker_side.extract_angle(frame1, colors, modify=True)
            except:
                print('frame1 process error')
                continue
            try:
                frame2, angle_top, marker_z, fingertip_z = tracker_top.extract_single_angle(frame2, basepoint)
                print('marker_z:', marker_z)
                print('fingertip_z:', fingertip_z)
            except:
                    print('frame2 process error')
                    continue
            
            cv2.imshow('Mask Preview', tracker_top.binary_mask)
            cv2.waitKey(1) 
            ########################################

            combined_frames = cv2.hconcat([frame1, frame2])
            # combined_frames = cv2.vconcat([frame1, frame2])
            cv2.imshow(windowname, combined_frames)
            print('rangers sideview;', tracker_side.marker_rangers)
            print('rangers topview;', tracker_top.fingertip_range)
            if is_record:
                frames_to_store_side.append(frame1)
                frames_to_store_top.append(frame2)
                measure.append([frame_id, angle0, angle1, angle2, angle_top,
                                tuple(raw_marker_pos[0][0]),
                                tuple(raw_marker_pos[0][1]), tuple(raw_marker_pos[1][0]), tuple(raw_marker_pos[1][1]),
                                tuple(raw_marker_pos[2][0]), tuple(raw_marker_pos[2][1]), tuple(raw_marker_pos[3][0]),
                                tuple(modified_marker_pos[0][0]),
                                tuple(modified_marker_pos[0][1]),tuple(modified_marker_pos[1][0]),tuple(modified_marker_pos[1][1]),
                                tuple(modified_marker_pos[2][0]),tuple(modified_marker_pos[2][1]),tuple(modified_marker_pos[3][0]),
                                tuple(tracker_side.DIP), tuple(tracker_side.PIP),tuple(tracker_side.MCP),
                                tuple(tracker_top.fingertip), tuple(basepoint)
                                ])
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    finally:
        cap1.release()
        cap2.release()
        cv2.destroyAllWindows()
        columns = ["frame","angle0", "angle1", "angle2", "angle_top",
                "marker pos0","marker pos1","marker pos2","marker pos3","marker pos4","marker pos5","marker pos6",
                "modified marker pos0","modified marker pos1","modified marker pos2","modified marker pos3","modified marker pos4","modified marker pos5","modified marker pos6",
                "DIP_side","PIP_side", "MCP_side", "fingertip_top", "MCP_top"]
        tracker_side.store_video(frames_to_store_side, output_video_fps)
        tracker_top.store_video(frames_to_store_top, output_video_fps)
        tracker_side.store_data_disignated_columns(measure, output_video_fps ,columns=columns)


