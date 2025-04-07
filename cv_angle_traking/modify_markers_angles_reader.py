# This file is created by Nagamine on March 31.(copy of angles_reader)
# to analize new-finger movement from exisxting video 
import time,os
os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin")
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from angles_reader_multicolor import AngleTracker

class ModifiedMarkers(AngleTracker):
    def __init__(self,video_name=None, denoising_mode='monocolor'):
        super().__init__(video_name, denoising_mode)

    def set_params(self, theta, distance):
        self.theta = theta
        self.distance = distance

    def multiply_vector(self, vector, rate):
        return rate * vector
    
    def rotate_vector(self, vector, theta): #To rotate vector by theta(rad). vector must be [a,b], dont give two dots.
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
        new_vec = rotation_matrix @ vector
        new_vec = np.array(new_vec, dtype=int)
        return new_vec
    
    def shift_markers(self, markers, d): # markers should be [distal, proximal], each element: [x,y]
        markers = np.array(markers)
        vector = markers[0] - markers[1] # 近位から遠位へのベクトル
        rotate_matrix = np.array([[0,-1],[1,0]])
        vertical_vector = rotate_matrix @ vector
        shifter = vertical_vector * (d/np.linalg.norm(vertical_vector))
        modified_markers = [markers[0]+shifter, markers[1]+shifter]
        modified_markers = np.array(modified_markers, dtype=int)
        return modified_markers
    
            
    def marker_discriminator(self, markers): #中節骨と基節骨のマーカーを区別する　上と同様
        markers = np.array(markers)
        distance0 = self.calculate_distance(self.palm_marker_position, markers[0])
        distance1 = self.calculate_distance(self.palm_marker_position, markers[1])
        if distance0 > distance1:
            return markers
        elif distance0 < distance1:
            markers[0], markers[1] = markers[1], markers[0]
            return markers

    def marker_discriminator_distalis(self, markers): #末節骨のマーカーを区別する　[遠位のマーカー　近位のマーカー]にする
        # distalmarker_mediaは中節骨の遠位のマーカー
        distance0 = self.calculate_distance(self.media_distalis, markers[0])
        distance1 = self.calculate_distance(self.media_distalis, markers[1])
        if distance0 > distance1:
            return markers
        elif distance0 < distance1:
            markers[0], markers[1] = markers[1], markers[0]
            return markers
        
    # def calculate_angle(self, line1, line2, index): #over ride
    #     pass
    @staticmethod
    def calculate_distance(point0, point1):
        point0 = np.array(point0)
        point1 = np.array(point1)
        return np.linalg.norm(point0-point1)

    def frame_trimer(self, frame, x,y):
        return frame[0:y, 0:x]

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
        markerset_per_frame = []
        if modify:modified_markerset_per_frame=[]

        # Set the line padding value
        line_pad = 5  # Adjust this value as needed

        # Initialize the direction vector for the first line
        direction_vector_0_1 = None
        
        if 1:
            # Iterate over each color marker
            for mask, thr, color, color_name, direction_vector, color_num in zip(masks, threshold_area_size, colors, colors_name, [direction_vector_0_1, None, None, None],[0,1,2,3]):
                # Convert the mask to uint8
                try:
                    mask = np.uint8(mask) # True/False -> 0/1
            
                    # Find connected components in the mask
                    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

                    # Filter regions based on area threshold
                    filtered_regions = [index for index, stat in enumerate(stats[1:]) if stat[4] >= thr]
                    
                        

                    # Initialize a list to store points per mask
                    point_per_mask = []
                    if modify: modified_point_per_mask = []

                    # If missed point, go next mask
                    if len(filtered_regions)<2 and (not color_num == 3): 
                        point_per_mask.extend([(-1,-1),(-1,-1)]) 
                        markerset_per_frame.append(point_per_mask)
                        continue

                    # Iterate over filtered regions in the mask
                    for idx, index in enumerate(filtered_regions):
                        # Access region properties from the stats array
                        left, top, width, height, area = stats[index + 1]

                        # Calculate the centroid
                        centroid_x, centroid_y = int(left + width / 2), int(top + height / 2)

                        # Append the centroid to the list of points for the mask
                        point_per_mask.append((centroid_x, centroid_y))

                        #この辺にpoint_per_maskの順序を並べ替えるコードを書いたほうがいいかも -> done
                        #マーカーの順序はpoint_per_mask=[遠位,近位]
                        if color_num == 0 and idx == 1:
                            point_per_mask = self.marker_discriminator_distalis(point_per_mask)

                        
                        elif (color_num == 1 and idx == 1) or (color_num == 2 and idx == 1):
                            point_per_mask = self.marker_discriminator(point_per_mask)
                            self.media_distalis = point_per_mask[0]
                            point_per_mask = point_per_mask.tolist()
                        elif color_num == 3 and idx == 0:
                            self.palm_marker_position = np.array([centroid_x, centroid_y])
                            #self.palm_marker_positionとself.media_distalisをクリックで指定した位置とするコードの設定が必要 -> done

                    if modify:
                        # このへんにマーカー位置を修正するコードを書く -> done
                        # 修正前のマーカーの位置と修正後のマーカーの位置両方を保存する
                        if color_num  == 0: 
                            modified_distal = point_per_mask[0]
                            modified_proximal = point_per_mask[1]
                            modified_point_per_mask.append(tuple(modified_distal))
                            modified_point_per_mask.append(tuple(modified_proximal))
                        if color_num == 1:
                            # marker_vec = point_per_mask[1] - point_per_mask[0]
                            marker_vec = self.calculate_vector(point_per_mask[0], point_per_mask[1]) #遠位から近位へのベクトル
                            rotated_vec = self.rotate_vector(marker_vec, self.theta)
                            modified_distal = np.array(point_per_mask[0]) #遠位
                            modified_proximal = modified_distal + rotated_vec #近位
                            modified_point_per_mask.append(tuple(modified_distal))
                            modified_point_per_mask.append(tuple(modified_proximal))
                            
                        if color_num == 2:
                            modified_markers = self.shift_markers(point_per_mask, self.distance)
                            modified_distal, modified_proximal = modified_markers[0], modified_markers[1]
                            modified_point_per_mask.append(tuple(modified_distal))
                            modified_point_per_mask.append(tuple(modified_proximal))
                        if color_num == 3:
                            modified_point_per_mask=point_per_mask.copy()
                            modified_point_per_mask.append((point_per_mask[0][0]+100, point_per_mask[0][1]))
                            
                    

                    if modify: #visualize circles on modified marker positions
                        for idx, point in enumerate(modified_point_per_mask):
                            cv2.circle(frame, (point[0], point[1]), radius=idx * 10, color=color, thickness=2)
                        for idx, point in enumerate(modified_point_per_mask):
                            cv2.circle(frame, (point[0], point[1]), radius=idx*10 + 5, color=color, thickness=2)
                    else: # visualize circles on raw marker positions
                        for idx, point in enumerate(point_per_mask):
                            cv2.circle(frame, (point[0], point[1]), radius=idx * 10, color=color, thickness=2)
                        # Visualize circles for each point with increased radius
                        for idx, point in enumerate(point_per_mask):
                            cv2.circle(frame, (point[0], point[1]), radius=idx * 10 + 5, color=color, thickness=3)

                    if len(point_per_mask) <2 and (not color_num == 3):
                        continue
                    

                    # If direction vector is not initialized, calculate it from the first two points
                    

                    if modify: #修正後のマーカーの位置に線を書くコード
                        if direction_vector is None and color_num != 3:
                            direction_vector = self.calculate_vector(modified_point_per_mask[1],modified_point_per_mask[0])
                            point1 = (int(modified_point_per_mask[1][0] - line_pad * direction_vector[0]),
                                    int(modified_point_per_mask[1][1] - line_pad * direction_vector[1]), )
                    
                            point2 = (int(modified_point_per_mask[0][0] + line_pad * direction_vector[0]),
                                    int(modified_point_per_mask[0][1] + line_pad * direction_vector[1]), )

                        if direction_vector is None and color_num == 3:
                            direction_vector = np.array([100,0])
                            point1 = (int(modified_point_per_mask[0][0] - line_pad*direction_vector[0]),
                                    int(modified_point_per_mask[0][1] - line_pad*direction_vector[1]))
                            point2 = (int(modified_point_per_mask[0][0] + line_pad*direction_vector[0]),
                                    int(modified_point_per_mask[0][1] + line_pad*direction_vector[1]))
                    # Calculate points for the line based on the direction vector and line padding
                    else:
                        if direction_vector is None and color_num != 3:
                            direction_vector = self.calculate_vector(point_per_mask[1], point_per_mask[0])
                    
                        point1 = (int(point_per_mask[1][0] - line_pad * direction_vector[0]),
                                int(point_per_mask[1][1] - line_pad * direction_vector[1]), )
                        
                        point2 = (int(point_per_mask[0][0] + line_pad * direction_vector[0]),
                                int(point_per_mask[0][1] + line_pad * direction_vector[1]), )
                    
                    # Visualize the line connecting the two points
                    cv2.line(frame, point1, point2, color, 3)

                    # Append the points for the current mask to the list of points per frame
                    markerset_per_frame.append(point_per_mask)
                    if modify:
                        modified_markerset_per_frame.append(modified_point_per_mask)


                    # if len(filtered_regions)<2:
                    #     print("filtered_regions: ",filtered_regions)
                    #     print("num_labels: ",num_labels)
                    #     print("point_per_mask: ",point_per_mask)
                    #     # print("labels: ",labels)                    
                    #     return frame,[],[],[]
                    # else: 
                    #     print("filtered_regions: ",filtered_regions)
                    #     print("point_per_mask: ",point_per_mask)

                except: # if error occurs, go to next mask 本当にこれで機能する？
                    continue
                # Calculate angles between consecutive lines
                # print(makerset_per_frame)


            # この辺の修正が必要
            if modify:
                # print('raw marker positions', markerset_per_frame)
                # print('modified marker positions', modified_markerset_per_frame)
                try:
                    angle_0 = self.calculate_angle(modified_markerset_per_frame[0], modified_markerset_per_frame[1])[2]
                    angle_0 = int(10*angle_0)/10
                except IndexError:
                    angle_0 = []
                try:
                    angle_1 = self.calculate_angle(modified_markerset_per_frame[1], modified_markerset_per_frame[2])[2]
                    angle_1 = int(10*angle_1)/10
                except IndexError:
                    angle_1 = []
                try:
                    angle_2 = self.calculate_angle(modified_markerset_per_frame[2], modified_markerset_per_frame[3])[2]
                    angle_2 = int(10*angle_2)/10
                except:
                    angle_2 = []

                _text_pos_x = 100
            # Add text annotations to the frame with calculated angles
                # frame = self.add_text_to_frame(frame, "ANGLE 0: {}".format((angle_0)), position=(_text_pos_x, 210), font_scale=1, thickness=2, color=(255, 255, 0))
                # frame = self.add_text_to_frame(frame, "ANGLE 1: {}".format((angle_1)), position=(_text_pos_x, 240), font_scale=1, thickness=2, color=(255, 255, 0))
                # frame = self.add_text_to_frame(frame, "ANGLE 2: {}".format((angle_2)), position=(_text_pos_x, 270), font_scale=1, thickness=2, color=(255, 255, 0))
                frame = self.add_text_to_frame(frame, "ANGLE 0: {}".format(angle_0), position=(_text_pos_x, 210), font_scale=1, thickness=2, color=(255, 255, 0))
                frame = self.add_text_to_frame(frame, "ANGLE 1: {}".format(angle_1), position=(_text_pos_x, 240), font_scale=1, thickness=2, color=(255, 255, 0))
                frame = self.add_text_to_frame(frame, "ANGLE 2: {}".format(angle_2), position=(_text_pos_x, 270), font_scale=1, thickness=2, color=(255, 255, 0))
            
                # except Exception as err:
                #     print(color_name,' Failed!:',err)
                #     return frame,[],[],[]
                return frame, angle_0, angle_1, angle_2, markerset_per_frame, modified_markerset_per_frame
            
            else: return frame, None, None, None, markerset_per_frame, None

    def calculate_angle(self, distalis_markers, proximal_markers): #each markers is list, [(遠位),(近位)]
            try:
                distalis_markers = np.array(distalis_markers)
                proximal_markers = np.array(proximal_markers)
                distalis_vec = distalis_markers[0] - distalis_markers[1]
                proximal_vec = proximal_markers[0] - proximal_markers[1] #近位から遠位へのベクトル

                dot_product = np.dot(proximal_vec, distalis_vec)
                cross_product = np.cross(proximal_vec, distalis_vec)
                norm_distalis = np.linalg.norm(distalis_vec)
                norm_proximal = np.linalg.norm(proximal_vec)
                angle_rad = np.arctan2(cross_product, dot_product)
                angle_degree = np.degrees(angle_rad)
                if angle_rad < 0:
                    joint_angle = abs(angle_degree) + 180
                else:
                    joint_angle = 180 - angle_degree
                return angle_degree, angle_rad, joint_angle
            except:
                return [],[],[]

    def acquire_marker_color(self, frame, cv_choose_wd_name):
        marker_rangers = super().acquire_marker_color(frame, cv_choose_wd_name)
        self.media_distalis = self.marker_position_frame0[1]
        self.palm_marker_position = self.marker_position_frame0[3]
        return marker_rangers
    
    
    def store_raw_data(self, measure, set_fps=30): # save time, each angles, frame id, 6 x raw marker position data to CSV file
        df_angle = pd.DataFrame(data=measure, columns=["frame","angle0", "angle1", "angle2", 
                                                    "marker pos0","marker pos1","marker pos2","marker pos3","marker pos4","marker pos5","marker pos6",])
        df_angle["time"] = df_angle["frame"]/set_fps
        df_angle.to_csv(os.path.join(self.output_folder_path, f"{self.video_name.split('.')[0]}_extracted.csv"),index=False)
        np_data = np.array(measure)[:, 0:4 ,::-1]
        print("measure:", type(measure))
        print(type(np_data), np_data)
        saveFigure(np_data, f"{self.video_name.split('.')[0]}_extracted.csv", ["angle_2","angle_1","angle_0","frame"], show_img=False, figure_mode='Single')
        
    # def store_video(self,):
    #     pass


if __name__ == '__main__':
    import os,sys,json
    print(os.getcwd())
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
    
    video_name = "output_20250407_174902.mp4"
    frame_jump = 5

    ## For algorithm tuning
    # Are for optime
    kernel = np.ones((5,5),np.uint8)
    threshold_area_size = [20, 70, 20, 10]# [80, 20, 10, 40]
    frame_shift = 0
    output_video_fps = 90 # I dont know if its work

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

    #parameters
    theta = 0.60
    distance = -30
    tracker.set_params(theta, distance)

    # Videos capture cycles
    try:
        while True:
            strt = time.time()
            ret, frame = cap.read()
            if not ret: break
            frame = tracker.frame_trimer(frame, 1300, 1200)
            
            if cnt==frame_shift: tracker.acquire_marker_color(frame, cv_choose_wd_name)
            frame, angle_0, angle_1, angle_2, raw_marker_pos, modified_marker_pos  = tracker.extract_angle(frame,False, colors, modify=True)
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
            measure.append([cnt, angle_0,angle_1,angle_2, 
                            tuple(raw_marker_pos[0][0]), tuple(raw_marker_pos[0][1]), tuple(raw_marker_pos[1][0]), tuple(raw_marker_pos[1][1]),
                            tuple(raw_marker_pos[2][0]), tuple(raw_marker_pos[2][1]), tuple(raw_marker_pos[3][0])])

            # print("for debug:\nappended measure=", measure[-1])
            
            frames_to_store.append(frame.copy())
            cnt += 1

            if frame_jump == 0:
                pass
            elif not cnt % frame_jump ==0 :
                cnt += 1;continue
            else: 
                pass
                # print(cnt)
            # if cnt > 1000:break
            # print(cnt)
            cv2.imshow('Video Preview', frame)
            if cv2.waitKey(1) & 0xFF == 27: # cv2.waitKey(1000) & 0xFF == ord('q')
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

        print("\nFinished video extraction")

        ## Store processed video
        # Store the video with updated frames
        # Set the desired output video path
        
        tracker.store_video(frames_to_store,output_video_fps)
        print(tracker.marker_position_frame0)
        tracker.store_raw_data(measure, output_video_fps)
        print(tracker.video_pos_file_url)