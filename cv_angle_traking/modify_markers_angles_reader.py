# This file is created by Nagamine on March 31.(copy of angles_reader)
# to analize new-finger movement from exisxting video 
import os
os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin") # be careful that it won't work if it's normal opencv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from angles_reader_multicolor import AngleTracker

class ModifiedMarkers(AngleTracker):
    def __init__(self,video_name=None, threshold_area_size=None, denoising_mode='monocolor'):
        super().__init__(video_name, denoising_mode)
        self.video_path = './sc01/' + video_name
        print('video_path:',self.video_path)

        self.threshold_area_size = threshold_area_size

    def set_params(self, theta=0, distance=0, theta_top=0):
        self.theta = theta
        self.distance = distance
        self.theta_top = theta_top

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
    
            
    def marker_discriminator(self, markers): #中節骨と基節骨のマーカーを区別する　[遠位のマーカー　近位のマーカー]にする
        # markers = np.array(markers)
        distance0 = self.calculate_distance(self.palm_marker_position, markers[0])
        distance1 = self.calculate_distance(self.palm_marker_position, markers[1])
        # print('distances:', distance0, distance1)
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
    
    def segment_marker_by_color(self,frame_tmp, num_maker_sets): 
        # Input must be a frame in the cielab color model from the OpenCV function

        if self.denoising_mode == 'color':
            # Single img denoising
            frame_tmp = cv2.fastNlMeansDenoisingColored(frame_tmp,None, 7, 7, 3, 5)# 
            # _mask = _mask>0.5
        
        if num_maker_sets != 1:
            # Extract color channels
            L_channel = frame_tmp[:, :, 0] # lightness
            a_channel = frame_tmp[:, :, 1] # red -> green
            b_channel = frame_tmp[:, :, 2] # Yellow -> Blue

            marker_rangers = self.marker_rangers
            markers_masks = []
            # print(marker_rangers)
            for i in range(num_maker_sets):# Color segmentation using NumPy array operations
                _mask =((L_channel > marker_rangers[i][0][0]) & (L_channel < marker_rangers[i][0][1]) &
                        (a_channel > marker_rangers[i][1][0]) & (a_channel < marker_rangers[i][1][1]) &
                        (b_channel > marker_rangers[i][2][0]) & (b_channel < marker_rangers[i][2][1]) )
                
                
                if self.denoising_mode == 'monocolor':
                    # Single img denoising
                    _mask = cv2.fastNlMeansDenoising(np.uint8(_mask),None, 5, 3, 5)# 
                    _mask = _mask>0.5

                markers_masks.append( _mask )

        else:
            L_channel = frame_tmp[:, :, 0] # lightness
            a_channel = frame_tmp[:, :, 1] # red -> green
            b_channel = frame_tmp[:, :, 2] # Yellow -> Blue
            markers_masks = []
            # print(marker_rangers)
            for i in range(num_maker_sets):# Color segmentation using NumPy array operations
                _mask =((L_channel > self.fingertip_range[0][0][0]) & (L_channel < self.fingertip_range[0][1][0]) &
                        (a_channel > self.fingertip_range[0][0][1]) & (a_channel < self.fingertip_range[0][1][1]) &
                        (b_channel > self.fingertip_range[0][0][2]) & (b_channel < self.fingertip_range[0][1][2]) )
                
                if self.denoising_mode == 'monocolor':
                    # Single img denoising
                    _mask = cv2.fastNlMeansDenoising(np.uint8(_mask),None, 5, 3, 5)# 
                    _mask = _mask>0.5

                markers_masks.append( _mask )

        if False: # Display masks in a cv high gui window
            mask_in_one = np.vstack((markers_masks[0],markers_masks[1],markers_masks[2],markers_masks[3]))
            # mask_in_one = np.vstack((markers_masks[2]))
            # print("!!!!:",type(_mask))
            # cv2.namedWindow("Mask",cv2.WINDOW_GUI_EXPANDED)
            # while True:

            cv2.imshow("Mask",255*np.uint8(mask_in_one))    

            # cv2.waitKey(1) 
                
        return markers_masks

    def extract_angle(self, frame, colors, modify=True): #over ride
        # Convert the input frame to the CIELAB color space

        cielab_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2Lab)

        # Segment markers by color in the CIELAB color space
        [marker_blue, marker_pink, marker_green, marker_yellow] = self.segment_marker_by_color(cielab_frame, 4)

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
            for mask, thr, color, color_name, direction_vector, color_num in zip(masks, self.threshold_area_size, colors, colors_name, [direction_vector_0_1, None, None, None],(0,1,2,3)):
                # Convert the mask to uint8
                try:
                    # print('maskdebugA;', mask.shape, mask.dtype)
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
                        if modify:
                            modified_point_per_mask.extend([(-1,-1),(-1,-1)])
                            modified_markerset_per_frame.append(modified_point_per_mask)
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
                            # point_per_mask = point_per_mask.tolist()
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
                     # visualize circles on raw marker positions
                        
                        for idx, point in enumerate(point_per_mask):
                            cv2.circle(frame, (point[0], point[1]), radius=idx * 10, color=[255,255,255], thickness=2)
                        # Visualize circles for each point with increased radius
                        for idx, point in enumerate(point_per_mask):
                            cv2.circle(frame, (point[0], point[1]), radius=idx * 10 + 5, color=[255,255,255], thickness=3)

                    if len(point_per_mask) <2 and (not color_num == 3):
                        continue

                    print('point_per_mask:',point_per_mask)

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

            if modify:
                # print('raw marker positions', markerset_per_frame)
                self.estimate_joint(modified_markerset_per_frame)
                cv2.circle(frame, center=self.DIP, radius=10, color=[0,255,0], thickness=10)
                cv2.circle(frame, center=self.PIP, radius=10, color=[0,255,0], thickness=10)
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

    def segment_marker_by_color_opencv(self, frame_tmp, side=True):
        cielab = cv2.cvtColor(frame_tmp, cv2.COLOR_RGB2Lab)
        self.marker_rangers = np.array(self.marker_rangers)
        if side: # side camera
            masks = []
            for rng in self.marker_rangers:
                rng = rng.T
                lowerb = tuple(int(x) for x in rng[0])
                upperb = tuple(int(x) for x in rng[1])
                mask = cv2.inRange(cielab, lowerb, upperb)
                masks.append(mask > 0)

            # masks = np.stack(mask)
            return masks
        else: # top camera
            # rng = self.fingertip_range[0].T # brown marker
            lowerb = tuple(int(x) for x in self.fingertip_range[0][0])
            upperb = tuple(int(x) for x in self.fingertip_range[0][1])
            mask = cv2.inRange(cielab, lowerb, upperb)
            return [mask > 0]

    def extract_angle_improved(self, frame, colors, modify=True): # BUGS here! faster than upper one, but cannot display masks
        
        cielab_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2Lab)
        masks = self.segment_marker_by_color_opencv(frame)
        colors_name = ["blue", "pink", "green", "yellow", "brown"]

        # Initialize a list to store points per frame
        markerset_per_frame = []
        if modify:modified_markerset_per_frame=[]
        joint_pos = []

        # Set the line padding value
        line_pad = 5  # Adjust this value as needed

        # Initialize the direction vector for the first line
        direction_vector_0_1 = None

        
        if 1:
            # Iterate over each color marker
            for mask, thr, color, color_name, direction_vector, color_num in zip(masks, self.threshold_area_size, colors, colors_name, [direction_vector_0_1, None, None, None],(0,1,2,3)):
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
                        if modify:
                            modified_point_per_mask.extend([(-1,-1),(-1,-1)])
                            modified_markerset_per_frame.append(modified_point_per_mask)
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
                            # point_per_mask = point_per_mask.tolist()
                        elif color_num == 3 and idx == 0:
                            self.palm_marker_position = np.array([centroid_x, centroid_y])
                            #self.palm_marker_positionとself.media_distalisをクリックで指定した位置とするコードの設定が必要 -> done

                    if modify:
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
                        
                     # visualize circles on raw marker positions
                        
                        # for idx, point in enumerate(point_per_mask):
                        #     cv2.circle(frame, (point[0], point[1]), radius=idx * 10, color=[255,255,255], thickness=2)
                        # # Visualize circles for each point with increased radius
                        # for idx, point in enumerate(point_per_mask):
                        #     cv2.circle(frame, (point[0], point[1]), radius=idx * 10 + 5, color=[255,255,255], thickness=3)

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

            if modify:
                # print('raw marker positions', markerset_per_frame)
                self.estimate_joint(modified_markerset_per_frame)
                cv2.circle(frame, center=self.DIP, radius=10, color=[0,255,0], thickness=10)
                cv2.circle(frame, center=self.PIP, radius=10, color=[0,255,0], thickness=10)
                cv2.circle(frame, center=self.MCP, radius=10, color=[0,255,0], thickness=10)
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
            
    def estimate_joint(self, modified_markerset_per_frame, shifters=[15,110]):
        # print('modified in estimate joint', modified_markerset_per_frame)
        print('modi', modified_markerset_per_frame)
        modified_markerset_per_frame_vec = np.array(modified_markerset_per_frame)
        direction_vector_0 = modified_markerset_per_frame_vec[1][0] - modified_markerset_per_frame_vec[1][1] #近位から遠位へのベクトル
        direction_vector_1 = modified_markerset_per_frame_vec[2][0] - modified_markerset_per_frame_vec[2][1]

        self.DIP = modified_markerset_per_frame_vec[1][0] + (shifters[0]/np.linalg.norm(direction_vector_0))*direction_vector_0
        self.PIP = modified_markerset_per_frame_vec[2][0] + (shifters[1]/np.linalg.norm(direction_vector_1))*direction_vector_1
        self.DIP = self.DIP.astype(np.int32)
        self.PIP = self.PIP.astype(np.int32)
        self.MCP = (modified_markerset_per_frame[3][0][0]-130, modified_markerset_per_frame[3][0][1])

        return self.DIP, self.PIP, self.MCP

    def acquire_marker_color(self, frame, cv_choose_wd_name):
        marker_rangers = super().acquire_marker_color(frame, cv_choose_wd_name)
        self.media_distalis = self.marker_position_frame0[1]
        self.palm_marker_position = self.marker_position_frame0[3]
        self.make_ranger_topview()
        return marker_rangers
    
    
    def store_raw_data(self, measure, set_fps=30): # save time, each angles, frame id, 6 x raw marker position data to CSV file
        # df_angle = pd.DataFrame(data=measure, columns=["frame","angle0", "angle1", "angle2", 
        #                                             "marker pos0_x","marker pos0_y","marker pos1","marker pos2","marker pos3","marker pos4","marker pos5","marker pos6",
        #                                             "DIP_x", "DIP_y", "PIP_x", "PIP_y", "MCP_x", "MCP_y"])
        df_angle = pd.DataFrame(data=measure, columns=["frame","angle0", "angle1", "angle2", 
                                                    "marker pos0","marker pos1","marker pos2","marker pos3","marker pos4","marker pos5","marker pos6",
                                                    "modified marker pos0","modified marker pos1","modified marker pos2","modified marker pos3","modified marker pos4","modified marker pos5","modified marker pos6",
                                                    "DIP","PIP"])
        df_angle["time"] = df_angle["frame"]/set_fps
        df_angle.to_csv(os.path.join(self.output_folder_path, f"{self.video_name.split('.')[0]}_extracted.csv"),index=False)
        np_data = np.array(measure)[:, 0:4 ,::-1]
        print("measure:", type(measure))
        print(type(np_data), np_data)
        saveFigure(np_data, f"{self.video_name.split('.')[0]}_extracted.csv", ["angle_2","angle_1","angle_0","frame"], show_img=False, figure_mode='Single')

    def store_data_disignated_columns(self, measure, set_fps=30, columns=[]):
        df_angle = pd.DataFrame(data=measure, columns=columns)
        df_angle["time"] = df_angle["frame"]/set_fps
        df_angle.to_csv(os.path.join(self.output_folder_path, f"{self.video_name.split('.')[0]}_extracted.csv"),index=False)
        np_data = np.array(measure)[:, 0:4 ,::-1]
        print("measure:", type(measure))
        print(type(np_data), np_data)
        # saveFigure(np_data, f"{self.video_name.split('.')[0]}_extracted.csv", ["angle_2","angle_1","angle_0","frame"], show_img=False, figure_mode='Single')    
    
# --------------↓　for single joint tracking　↓----------------

    def extract_single_angle(self, frame, basepoint): # topview
        
        cielab_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2Lab)
        mask = self.segment_marker_by_color_opencv(frame, False)[0]
        # mask = self.segment_marker_by_color(cielab_frame, 1)[0]

        # Initialize a list to store points per frame
        markerpos_per_frame = []
        modified_pos_per_frame=[]

        # Set the line padding value
        # line_pad = 5  # Adjust this value as needed
        line_pad = 0  # Adjust this value as needed

        thr = 100

        # Initialize the direction vector for the first line
        direction_vector = None

        if 1:
            try:
                mask = np.uint8(mask) # True/False -> 0/1
                self.binary_mask = mask * 255
                # Find connected components in the mask
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

                # Filter regions based on area threshold
                filtered_regions = [index for index, stat in enumerate(stats[1:]) if stat[4] >= thr]
                # If missed point, go to next frame
                if len(filtered_regions) <= 0 : 
                    markerpos_per_frame.append((-1,-1)) 
                    modified_pos_per_frame.append((-1, -1))
                    return frame, [], None, None
                
                # extract only a ball joint
                max_area = 0
                for idx, index in enumerate(filtered_regions):
                    left, top, width, height, area = stats[index + 1]
                    if area > max_area:
                        area_max_index = index
                        ball_joint_stats = {'left':left, 'top':top, 'width':width, 'height':height, 'area':area }
                centroid_x, centroid_y = int(ball_joint_stats['left'] + ball_joint_stats['width'] / 2), int(ball_joint_stats['top'] + ball_joint_stats['height'] / 2)
                markerpos_per_frame.append((centroid_x, centroid_y))

                        
                # visualize circles on raw marker positions
                print('markerpos per frame', markerpos_per_frame)
                cv2.circle(frame, (markerpos_per_frame[0][0], markerpos_per_frame[0][1]), radius=10, color=[255,255,255], thickness=2)
                cv2.circle(frame, basepoint, radius=10, color=[255,255,255], thickness=-1)
                
                vector_to_rotate = np.array([centroid_x, centroid_y]) - basepoint
                # calculate modified position and angle 
                if markerpos_per_frame[0][0] <= basepoint[0] :# the case fingertip is left of MCP
                    rotated_vec = self.rotate_vector(vector_to_rotate, self.theta_top)
                    point_on_fingerline = basepoint - np.array([1000,0])
                else:
                    rotated_vec = self.rotate_vector(vector_to_rotate, -self.theta_top)
                    point_on_fingerline = basepoint + np.array([1000,0])

                print('rotated_fjdklfjdkl', rotated_vec)
                modified_pos = basepoint + rotated_vec
                print('modified fingertip', modified_pos)
                modified_pos_per_frame.append(tuple(modified_pos))
                angle = self.calculate_single_angle(modified_pos_per_frame[0], point_on_fingerline, basepoint)
                angle = int(10*np.degrees(angle))/10
                    
                
                #visualize circles on modified marker position
                cv2.circle(frame, (modified_pos_per_frame[0][0], modified_pos_per_frame[0][1]), radius= 10, color=[0,255,0], thickness=-1)
                
                if direction_vector is None: # draw line between basepoint and modified position
                    direction_vector = self.calculate_vector(basepoint, modified_pos_per_frame[0]) 

                point1 = (int(basepoint[0] - line_pad * direction_vector[0]),
                        int(basepoint[1] - line_pad * direction_vector[1]), )
        
                point2 = (int(modified_pos_per_frame[0][0] + line_pad * direction_vector[0]),
                        int(modified_pos_per_frame[0][1] + line_pad * direction_vector[1]), )
                
                # Visualize the line connecting the two points
                cv2.line(frame, point1, point2, color=[255,255,255], thickness=3)
                cv2.line(frame,point_on_fingerline, basepoint, color=[255,255,255], thickness=3)
                self.fingertip = (int(modified_pos_per_frame[0][0]), int(modified_pos_per_frame[0][1]))

            except: # if error occurs, go to next mask 本当にこれで機能する？
                return frame, [], None, None

            _text_pos_x = 100
            frame = self.add_text_to_frame(frame, "ANGLE: {}".format(angle), position=(_text_pos_x, 210), font_scale=1, thickness=2, color=(255, 255, 0))
            
            return frame, angle, markerpos_per_frame, modified_pos_per_frame
            
    def calculate_single_angle(self, A, B, C): # calculate angle ACB
        # Convert points to numpy arrays
        A = np.array(A)
        B = np.array(B)
        C = np.array(C)
        # Calculate vectors AB and BC
        CA = A - C 
        CB = B - C
        dot = np.dot(CB, CA)
        det = CB[0]*CA[1] - CB[1]*CA[0]  
        angle = np.arctan2(det, dot)
        return angle
    
    def make_ranger_topview(self):
        upperlimit = self.fingertip_segment + [30,15,15]
        upperlimit = upperlimit.tolist()
        lowerlimit = self.fingertip_segment - [30,15,15]
        lowerlimit = lowerlimit.tolist()
        marker_rangers = [[lowerlimit, upperlimit]]
        self.fingertip_range = marker_rangers


if __name__ == '__main__':
    import os,sys,json
    print(os.getcwd())
    parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0,parentdir)
    from lib.GENERALFUNCTIONS import *

    # Constants
    ## For styling
    colors = [(43,74,134), (0,0,255), (255,0,0), (0,255,255)]
    line_padding = [0.7, 1.5,1.5,1.5]

    font_scale = 1
    text_position_cnt = (100, 100)
    text_position_time = (100, 120)
    
    video_name = "FDP.mp4"
    frame_jump = 0

    ## For algorithm tuning
    # Are for optime
    kernel = np.ones((5,5),np.uint8)
    threshold_area_size = [200, 50, 50, 10]# [80, 20, 10, 40]
    frame_shift = 0
    output_video_fps = 90 

    tracker = ModifiedMarkers(video_name,threshold_area_size,denoising_mode = 'monocolor')
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
    theta = 0.55
    distance = -30
    tracker.set_params(theta, distance)

    # Videos capture cycles
    try:
        while True:
            strt = time.time()
            ret, frame = cap.read()
            if not ret: break
            # frame = tracker.frame_trimer(frame, 1300, 1200)
            
            if cnt==frame_shift: tracker.acquire_marker_color(frame,cv_choose_wd_name)
            # frame, angle_0, angle_1, angle_2, raw_marker_pos, modified_marker_pos  = tracker.extract_angle(frame, colors, modify=True)
            frame, angle_0, angle_1, angle_2, raw_marker_pos, modified_marker_pos  = tracker.extract_angle_improved(frame,colors, modify=True)

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
            try:
                measure.append([cnt, angle_0,angle_1,angle_2, 
                                # tuple(raw_marker_pos[0][0][0]), tuple(raw_marker_pos[0][0][1]), # fingertip
                                tuple(raw_marker_pos[0][0]),
                                tuple(raw_marker_pos[0][1]), tuple(raw_marker_pos[1][0]), tuple(raw_marker_pos[1][1]),
                                tuple(raw_marker_pos[2][0]), tuple(raw_marker_pos[2][1]), tuple(raw_marker_pos[3][0]),
                                tuple(modified_marker_pos[0][0]),
                                tuple(modified_marker_pos[0][1]),tuple(modified_marker_pos[1][0]),tuple(modified_marker_pos[1][1]),
                                tuple(modified_marker_pos[2][0]),tuple(modified_marker_pos[2][1]),tuple(modified_marker_pos[3][0]),
                                tuple(tracker.DIP), tuple(tracker.PIP), tuple(tracker.MCP)])
            except:
                cv2.imshow('Video Preview', frame)
                continue
                # print("for debug:\nappended measure=", measure[-1])
            
            frames_to_store.append(frame.copy())
            cnt += 1
            # if cnt % 20 == 0: print(cnt)
            print(cnt)


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