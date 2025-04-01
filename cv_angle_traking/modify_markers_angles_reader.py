# This file is created by Nagamine on March 31.(copy of angles_reader)
# to analize new-finger movement from exisxting video 
import time,os
os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin")
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from cv_angle_traking.angles_reader_multicolor import AngleTracker

class ModifiedMarkers(AngleTracker):
    def __init__(self,video_name=[], denoising_mode='monocolor'):
        super.__init__(video_name=[], denoising_mode='monocolor')

    def set_params(self, theta, distance, shift):
        self.theta = theta
        self.distance = distance
        self.shift = shift

    def multiply_vector(self, vector, rate):
        return rate * vector
    
    def rotate_vector(self, vector, theta): #To rotate vector by theta(rad). vector must be [a,b], dont give two dots.
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
        new_vec = rotation_matrix @ vector
        return new_vec
    
    def shift_markers(self, markers, d): # markers should be [distal, proximal], each element: [x,y]
        markers = np.array(markers)
        vector = markers[0] - markers[1] # 近位から遠位へのベクトル
        rotate_matrix = np.array([[0,-1],[1,0]])
        vertical_vector = rotate_matrix @ vector
        shifter = vertical_vector * (d/np.linalg.norm(vertical_vector))
        modified_markers = [markers[0]+shifter, markers[1]+shifter]
        modified_markers = np.array(modified_markers)
        return modified_markers
    
    def marker_discriminator_distalis(self): #末節骨のマーカーを区別する　[遠位のマーカー　近位のマーカー]にする
        pass
    def marker_discriminator(self): #中節骨と基節骨のマーカーを区別する　上と同様
        pass
    # def calculate_angle(self, line1, line2, index): #over ride
    #     pass
    @staticmethod
    def calculate_distance(point0, point1):
        point0 = np.array(point0)
        point1 = np.array(point1)
        return np.linalg.norm(point0-point1)

    def extract_angle(self, frame, swap, colors, modify=True): #over ride
        # Convert the input frame to the CIELAB color space

        cielab_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2Lab)

        # Segment markers by color in the CIELAB color space
        [marker_blue, marker_pink, marker_green, marker_yellow] = self.segment_marker_by_color(cielab_frame)
        # marker_blue, marker_pink, marker_green, marker_yellow = segment_marker_by_color(cielab_frame)


        # Create a stack of masks for each color marker
        masks = np.stack([marker_blue, marker_pink, marker_green, marker_yellow], axis=0)

        # Define color names for visualization
        colors_name = ["blue", "pink", "green", "yellow"]

        # Initialize a list to store points per frame
        makerset_per_frame = []

        # Set the line padding value
        line_pad = 5  # Adjust this value as needed

        # Initialize the direction vector for the first line
        direction_vector_0_1 = None
        
        if 1:
            # Iterate over each color marker
            for mask, thr, color, color_name, direction_vector in zip(masks, threshold_area_size, colors, colors_name, [direction_vector_0_1, None, None, None]        ):
                # Convert the mask to uint8
                
                mask = np.uint8(mask) # True/False -> 0/1
        
                # Find connected components in the mask
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

                # Filter regions based on area threshold
                filtered_regions = [index for index, stat in enumerate(stats[1:]) if stat[4] >= thr]
                
                    

                # Initialize a list to store points per mask
                point_per_mask = []
                if modify: modified_point_per_mask = []

                # If missed point, go next mask
                if len(filtered_regions) < 2: 
                    point_per_mask.extend([(-1,-1),(-1,-1)]) 
                    makerset_per_frame.append(point_per_mask)
                    continue

                # Iterate over filtered regions in the mask
                for idx, index in enumerate(filtered_regions):
                    # Access region properties from the stats array
                    left, top, width, height, area = stats[index + 1]

                    # Calculate the centroid
                    centroid_x, centroid_y = int(left + width / 2), int(top + height / 2)

                    # Append the centroid to the list of points for the mask
                    point_per_mask.append((centroid_x, centroid_y))


                    #この辺にpoint_per_maskの順序を並べ替えるコードを書いたほうがいいかも？
                    #マーカーの順序はpoint_per_mask=[遠位,近位]

                    if modify:
                        marker_vec = []
                        # このへんにマーカー位置を修正するコードを書く
                        # point_per_maskがマーカーの点なのでidxかindexの値に応じて処理を分ける
                        # 修正前のマーカーの位置と修正後のマーカーの位置両方を保存する
                        
                        if idx == 0: 
                            modified_distal = point_per_mask[0]
                            modified_proximal = point_per_mask[1]
                        if idx == 1:
                            # marker_vec = point_per_mask[1] - point_per_mask[0]
                            marker_vec = self.calculate_vector(point_per_mask[0], point_per_mask[1]) #遠位から近位へのベクトル
                            rotated_vec = self.rotate_vector(marker_vec, self.theta)
                            modified_distal = np.array(point_per_mask[0]) #遠位
                            modified_proximal = modified_distal + rotated_vec #近位
                            
                        if idx == 2:
                            marker_vec = self.calculate_vector(point_per_mask[0],point_per_mask[1])
                            self.shift_markers 


                        modified_point_per_mask.append(tuple(modified_distal))
                        modified_point_per_mask.append(tuple(modified_proximal))
                        print(f'for debug:\n marker index:{idx}, modified_point_per_mask: {modified_point_per_mask}')


                if modify:
                    for idx, point in enumerate(modified_point_per_mask):
                        cv2.circle(frame, (point[0], point[1]), radius=idx * 10, color=color, thickness=2)
                    for idx, point in enumerate(modified_point_per_mask):
                        cv2.circle(frame, (point[0], point[1]), radius=idx * 10, color=color, thickness=2)
                else: # visualize circles on raw marker positions
                    for idx, point in enumerate(point_per_mask):
                        cv2.circle(frame, (point[0], point[1]), radius=idx * 10, color=color, thickness=2)

                    # Visualize circles for each point with increased radius
                    for idx, point in enumerate(point_per_mask):
                        cv2.circle(frame, (point[0], point[1]), radius=idx * 10 + 10, color=color, thickness=3)

                if len(point_per_mask) <2:
                    continue
                # If direction vector is not initialized, calculate it from the first two points
                if direction_vector is None:
                    direction_vector = self.calculate_vector(point_per_mask[1], point_per_mask[0])

                # Calculate points for the line based on the direction vector and line padding
                point1 = (int(point_per_mask[1][0] - line_pad * direction_vector[0]),
                        int(point_per_mask[1][1] - line_pad * direction_vector[1]), )
                
                point2 = (int(point_per_mask[0][0] + line_pad * direction_vector[0]),
                        int(point_per_mask[0][1] + line_pad * direction_vector[1]), )

                # Visualize the line connecting the two points
                cv2.line(frame, point1, point2, color, 3)

                # Append the points for the current mask to the list of points per frame
                makerset_per_frame.append(point_per_mask)

                # if len(filtered_regions)<2:
                #     print("filtered_regions: ",filtered_regions)
                #     print("num_labels: ",num_labels)
                #     print("point_per_mask: ",point_per_mask)
                #     # print("labels: ",labels)                    
                #     return frame,[],[],[]
                # else: 
                #     print("filtered_regions: ",filtered_regions)
                #     print("point_per_mask: ",point_per_mask)

            # Calculate angles between consecutive lines
            # print(makerset_per_frame)
            angle_0 = self.calculate_angle(makerset_per_frame[0], makerset_per_frame[1])
            angle_1 = self.calculate_angle(makerset_per_frame[1], makerset_per_frame[2])
            angle_2 = self.calculate_angle(makerset_per_frame[2], makerset_per_frame[3])

            _text_pos_x = 100
            # Add text annotations to the frame with calculated angles
            frame = self.add_text_to_frame(frame, "ANGLE 0: {}".format((angle_0)), position=(_text_pos_x, 210), font_scale=1, thickness=2, color=(255, 255, 0))
            frame = self.add_text_to_frame(frame, "ANGLE 1: {}".format((angle_1)), position=(_text_pos_x, 240), font_scale=1, thickness=2, color=(255, 255, 0))
            frame = self.add_text_to_frame(frame, "ANGLE 2: {}".format((angle_2)), position=(_text_pos_x, 270), font_scale=1, thickness=2, color=(255, 255, 0))
            
        # except Exception as err:
        #     print(color_name,' Failed!:',err)
        #     return frame,[],[],[]


        return frame, angle_0, angle_1, angle_2
        pass
    def store_raw_data(self, measure, set_fps=30): # save time, each angles, frame id, 6 x raw marker position data to CSV file
        df_angle = pd.DataFrame(data=measure, columns=["frame","angle0", "angle1", "angle2", 
                                                    "marker pos0","marker pos1","marker pos2","marker pos3","marker pos4","marker pos5","marker pos6",])
        df_angle["time"] = df_angle["frame"]/set_fps
        df_angle.to_csv(os.path.join(self.output_folder_path, f"{video_name.split('.')[0]}_extracted.csv"),index=False)
        np_data = np.array(measure)[:, ::-1]
        print("measure:", type(measure))
        print(type(np_data), np_data)
        saveFigure(np_data, f"{video_name.split('.')[0]}_extracted.csv", ["angle_2","angle_1","angle_0","frame"], show_img=False, figure_mode='Single')
        
    def store_video(self,):
        pass


if __name__ == '__main__':
    import os,sys,json
    parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0,parentdir)

    from lib.GENERALFUNCTIONS import *

    # Constants
    ## For styling
    colors = [(255,0,0), (127,0,255), (0,127,0), (0,127,255)]
    line_padding = [0.7, 1.5,1.5,1.5]

    font_scale = 1
    text_position_cnt = (100, 100)
    text_position_time = (100, 120)
    
    video_name = "sc01.mp4"
    frame_jump = 5

    ## For algorithm tuning
    # Are for optime
    kernel = np.ones((5,5),np.uint8)
    threshold_area_size = [10, 10, 10, 10]# [80, 20, 10, 40]
    frame_shift = 0
    output_video_fps = 30 # I dont know if its work

    tracker = ModifiedMarkers(video_name,denoising_mode = 'monocolor')
    # Main logic
    cap = cv2.VideoCapture(tracker.video_path) 
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_shift)
    cap.set(cv2.CAP_PROP_FPS, output_video_fps)

    enable_gpu_acc = True
    if not cv2.cuda.getCudaEnabledDeviceCount():
        print("Cuda accelaration is not supported, working on CPU")
        enable_gpu_acc = False

    if not cap.isOpened():
        print("Error: Could not open the video file.")
        exit()

    # Create a window to display the frames
    cv_preview_wd_name = 'Video Preview'
    cv_choose_wd_name = 'Choose'

    cv2.namedWindow(cv_preview_wd_name, cv2.WINDOW_GUI_EXPANDED)
    cv2.namedWindow("Mask",cv2.WINDOW_GUI_EXPANDED)

    measure = [] # for storing angles
    frames_to_store = []
    cnt = frame_shift # for storing frame count

    # Videos capture cycles
    while True:
        strt = time.time()
        ret, frame = cap.read()
        if not ret: break
        
        if cnt==frame_shift: tracker.acquire_marker_color(frame)
 
        frame, angle_0, angle_1, angle_2  = tracker.extract_angle(frame, False)
        # # Use the original frame instead of creating a copy
        # try: frame, angle_0, angle_1, angle_2  = tracker.extract_angle(frame, False)
        # except Exception as err: continue
        # if frame is None: continue
        # cv2.imshow('Video Preview', frame)

        # Add text to the frame
        frame = tracker.add_text_to_frame(frame, str(cnt), position=text_position_cnt, font_scale=font_scale)

        # Calculate and add time information
        end = time.time()
        frame = tracker.add_text_to_frame(frame, str(end - strt), position=text_position_time, font_scale=font_scale)
        measure.append([cnt, angle_0,angle_1,angle_2])
        
        frames_to_store.append(frame.copy())
        cnt += 1

        if frame_jump == 0:
            pass
        elif not cnt % frame_jump ==0 :
            cnt += 1;continue
        else: print(cnt)
        # if cnt > 1000:break
        # print(cnt)
        cv2.imshow('Video Preview', frame)
        if cv2.waitKey(1) & 0xFF == 27: # cv2.waitKey(1000) & 0xFF == ord('q')
            break
    cap.release()
    cv2.destroyAllWindows()

    print("\nFinished video extraction")

    ## Store processed video
    # Store the video with updated frames
    # Set the desired output video path
    
    tracker.store_video(frames_to_store,output_video_fps)
    tracker.store_data(measure,output_video_fps)
    print(tracker.video_pos_file_url)