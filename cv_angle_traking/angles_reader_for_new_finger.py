# %%
import time,os
os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin")
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import sys


class AngleTracker(object): # TODO
   
    def __init__(self, video_name=[],denoising_mode='monocolor'): # TODO
        # colors_name = ["blue", "pink", "green", "yellow"]
        if video_name==[]:
            print("Empty video name, exiting...");exit()
             
        self.video_name = video_name
        self.video_path = DATA_FOLDER + video_name #"../IMG_7102.MOV"
        self.output_folder_path = DATA_FOLDER + self.video_name.split('.')[0] +'/'# "../output/video"
        # output_folder_csv = output_folder_path #"../output/csv"
        self.video_pos_file_url = self.output_folder_path + self.video_name.split('.')[0] +'.json'

        if not os.path.exists(self.output_folder_path):
            os.makedirs(self.output_folder_path)

        self.color_mode = 0 # 0: Lab,1: Rgb
        self.num_maker_sets = 4
        self.denoising_mode = denoising_mode# 'monocolor'
        self.threshold_area_size = [50,30,50,150]
        self.colors = [(255,0,0), (127,0,255), (255,0,255),(0,127,255)]
        if self.color_mode ==0: # Lab
            self.maker_tolerance_L = [75,50,20,20]#int(0.08 * 255)
            self.maker_tolerance_a = [30,45,17,17]# int(0.09 * 255)# red -> green
            self.maker_tolerance_b = [40,20,15,30]# int(0.09 * 255)# Yellow -> Blue
        else : # RGB
            self.maker_tolerance_L = int(0.5 * 255)
            self.maker_tolerance_a = int(0.2 * 255)# red -> green
            self.maker_tolerance_b = int(0.2 * 255)# Yellow -> Blue

        # self.marker_rangers = [ #[Low Lhigh alow ahigh blow bhigh]] #  SC02
        #                [ [100,255],[150,180],[100,150]], # Marker A
        #                [ [130,155],[100,200],[20,80]], # Marker B
        #                [ [180,220],[55,85],[100,180]], # Marker C
        #                [ [150,215],[80,100],[90,110]], # Marker D 
        #                 ]
        self.marker_rangers = [ #[Low Lhigh alow ahigh blow bhigh]] # SC01
                       [ [100,220],[160,220],[60,160]], # Marker A
                       [ [100,215],[70,190],[30,80]], # Marker B
                       [ [180,210],[55,85],[120,180]], # Marker C
                       [ [150,235],[80,90],[100,120]], # Marker D 
                        ]
        # self.marker_rangers = [ #[Low Lhigh alow ahigh blow bhigh]] # Default
        #                [ [0,255],[140,170],[160,255]], # Marker A
        #                [ [0,255],[175,255],[0,80]], # Marker B
        #                [ [0,255],[110,120],[130,180]], # Marker C
        #                [ [0,255],[80,120],[90,110]], # Marker D 

        #                 ]
        self._point_counter = 0
        self.maker_position_frame0 = []
        for _ in range(self.num_maker_sets):self.maker_position_frame0.append([0,0])

        self.enable_maker_pos_acquirement = False
        self.load_point_pos()
        self.angle_pos = []

        # self.fingertip_pos = []
        # self.angle0_pos = []
        # self.angle1_pos = []
        # self.angle2_pos = []
        self.fingertip_pos_list = np.array([0,0])
        self.angle0_pos_list = np.array([0,0])
        self.angle1_pos_list = np.array([0,0])
        self.angle2_pos_list = np.array([0,0])

        pass
        
    def add_text_to_frame(self,frame, text, position=(30, 30), font=cv2.FONT_HERSHEY_DUPLEX, font_scale=0.2, color=(0, 255, 0), thickness=2):
           # Add text overlay into video frame
        """
        Add text to a frame.

        Parameters:
        - frame (numpy.ndarray): Input frame.
        - text (str): Text to be added to the frame.
        - position (tuple): Position of the text (x, y).
        - font (int): Font type. FONT_HERSHEY_PLAIN FONT_HERSHEY_SIMPLEX
        - font_scale (float): Font scale.
        - color (tuple): Text color (B, G, R).
        - thickness (int): Text thickness.

        Returns:
        - numpy.ndarray: Frame with added text.
        """
        frame_with_text = frame.copy()
        cv2.putText(frame_with_text, text, position, font, font_scale, color, thickness)
        return frame_with_text

    def calculate_angle(self,line1, line2, index):
        try:
            # Convert lines to numpy arrays
            line1 = np.array(line1)
            line2 = np.array(line2)

            # Calculate the vectors corresponding to the lines
            vector1 = line1[1] - line1[0]
            vector2 = line2[1] - line2[0]

            # Calculate the dot product and cross product of the vectors
            dot_product = np.dot(vector1, vector2)
            cross_product = np.cross(vector1, vector2)

            # Calculate the magnitudes of the vectors
            magnitude1 = np.linalg.norm(vector1)
            magnitude2 = np.linalg.norm(vector2)

            # Calculate the cosine of the angle between the vectors
            cosine_theta = dot_product / (magnitude1 * magnitude2)

            # Determine the sign of the dot product to determine the direction
            if index == 0 or index == 1:
                if dot_product > 0:
                    angle_radians = np.arccos(cosine_theta)
                    # Convert the angle to degrees
                    angle_degrees = 180 - np.degrees(angle_radians)
                    # Adjust angle for the cross product sign
                    if cross_product < 0:
                        angle_degrees = 360 - angle_degrees
                else:
                    angle_radians = np.arccos(cosine_theta)
                    # Convert the angle to degrees
                    angle_degrees = np.degrees(angle_radians)
            elif index == 2:
                if self.temp_green_y < self.temp_yellow_y: # 90 < angle2 < 180
                    if dot_product > 0:
                        angle_radians = np.arccos(cosine_theta)
                        # Convert the angle to degrees
                        angle_degrees = 180 - np.degrees(angle_radians)
                        # Adjust angle for the cross product sign
                        # if cross_product < 0:
                        #     angle_degrees = 360 - angle_degrees
                    else: 
                        angle_radians = np.arccos(cosine_theta)
                        # Convert the angle to degrees
                        angle_degrees = np.degrees(angle_radians) 

                else: #angle2 > 180
                    if dot_product <= 0:
                        angle_radians = np.arccos(cosine_theta)
                        angle_degrees = 360 - np.degrees(angle_radians) 
                    else:
                        angle_radians = np.arccos(cosine_theta)
                        angle_degrees = 180 + np.degrees(angle_radians)
        except Exception as err:
            return []


        if index == 0 or index == 1 :#or index == 2:
            if angle_degrees > 180:
                angle_degrees = 360 - angle_degrees

        return angle_degrees
    
    @staticmethod
    def calculate_vector(point1, point2):
        return np.array(point2) - np.array(point1)

    def segment_marker_by_color_Maros(self,frame_tmp):
        # Input must be a frame in the cielab color model from the OpenCV function
        num_maker_sets = 4
        if self.denoising_mode == 'color':
            frame_tmp = cv2.fastNlMeansDenoisingColored(frame_tmp, None, 7,7,3,5)
        # Extract color channels
        L_channel = frame_tmp[:, :, 0]
        a_channel = frame_tmp[:, :, 1]
        b_channel = frame_tmp[:, :, 2]
        marker_rangers = self.marker_rangers
        markers_masks = []

        # Color segmentation using NumPy array operations
        for i in range(num_maker_sets):
            _mask = ((L_channel > marker_rangers[i][0][0]) & (L_channel < marker_rangers[i][0][0]) &
                     (a_channel > marker_rangers[i][1][0]) & (a_channel < marker_rangers[i[1][1]]) &
                     (b_channel > marker_rangers[i][2][0]) & (b_channel < marker_rangers[i][2][1]))
            if self.denoising_mode == 'monocolor':
                _mask = cv2.fastNlMeansDenoising(np.uint8(_mask), None, 5,3,5)
                _mask = _mask>0.5
            
            markers_masks.append( _mask )

        if True:
            mask_in_one = np.vstack((markers_masks[0],markers_masks[1],markers_masks[2],markers_masks[3]))
            cv2.imshow("Mask", 255 * np.uint8(mask_in_one))

        return markers_masks

    def segment_marker_by_color(self,frame_tmp): # Askar.L
        # Input must be a frame in the cielab color model from the OpenCV function
        num_maker_sets = 4

        if self.denoising_mode == 'color':
            # Single img denoising
            frame_tmp = cv2.fastNlMeansDenoisingColored(frame_tmp,None, 7, 7, 3, 5)# 
            # _mask = _mask>0.5

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
        
        if True: # Display masks in a cv high gui window
            mask_in_one = np.vstack((markers_masks[0],markers_masks[1],markers_masks[2],markers_masks[3]))
            # mask_in_one = np.vstack((markers_masks[2]))
            # print("!!!!:",type(_mask))
            # cv2.namedWindow("Mask",cv2.WINDOW_GUI_EXPANDED)
            # while True:
            cv2.imshow("Mask",255*np.uint8(mask_in_one))         
            # cv2.waitKey(1) 
                
        return markers_masks

    def acquire_marker_color(self,frame): #TODO
        marker_rangers_old = self.marker_rangers
        marker_rangers = []
        num_marker_sets = self.num_maker_sets

        cv2.namedWindow(cv_choose_wd_name, cv2.WINDOW_GUI_EXPANDED)
        cv2.setMouseCallback(cv_choose_wd_name, tracker.mouse_event)

        if self.color_mode == 0:
            frame_to_segment = cv2.cvtColor(frame, cv2.COLOR_RGB2Lab)
        else: frame_to_segment = frame
        
        if self.enable_maker_pos_acquirement:
            # frame = add_text_to_frame(frame,'Please choose',position=(40, 50),color=(255, 255,255),font_scale=1)
            _meassage = 'Choose '+ str(num_marker_sets)+' position for the marker'
        else:
            _meassage = "Loaded exsist maker position:"

            print(self.maker_position_frame0)
            for [x,y] in self.maker_position_frame0:
                self._disp_marker_pos(x,y,frame)
                # self.maker_position_frame0[self._point_counter] = [x,y]
                self._point_counter = self._point_counter + 1 if self._point_counter < self.num_maker_sets-1 else 0

            pass
        self._point_counter = 0

        cv2.putText(frame,_meassage, (40, 50), cv2.FONT_HERSHEY_DUPLEX,
                    1.8, (255, 255,255), thickness =2)
        cv2.putText(frame, 'Press Esc to continue on point extraction', (40, 80), cv2.FONT_HERSHEY_DUPLEX,
                    1, (255, 255,255), thickness =1)
        while not frame is None:
            cv2.imshow('Choose', frame)
            _key_pressed = cv2.waitKey(1)

            if  _key_pressed & 0xFF == 27: break
            elif _key_pressed == ord('s'):
                _meassage = 'Saving:'
                cv2.putText(frame, _meassage, (50, 200), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), thickness = 1)
                self.store_point_pos()# TODO
                time.sleep(0.6);break

        for _i,_pos in enumerate(self.maker_position_frame0):
            # Get color dara from lab img

            # Cal tolerance range
            upper_limit = frame_to_segment[_pos[1]][_pos[0]] + [
                self.maker_tolerance_L[_i], self.maker_tolerance_a[_i], self.maker_tolerance_b[_i]]  
            
            lower_limit = frame_to_segment[_pos[1]][_pos[0]] - [
                self.maker_tolerance_L[_i], self.maker_tolerance_a[_i], self.maker_tolerance_b[_i]]
            
            # print(upper_limit,lower_limit);exit() # [146 171  82] [122 147  58]

            marker_rangers_ch = []
            # Save to variable
            for _j in range(3):
                marker_rangers_ch.append([lower_limit[_j],upper_limit[_j]])
                # self.marker_rangers[_i][_j:_j+1] = [lower_limit[_j],upper_limit[_j]]
                # self.marker_rangers[_i][_j+1] = 
            pass
            marker_rangers.append(marker_rangers_ch)

        print(marker_rangers)
        self.marker_rangers = marker_rangers
        
        cv2.destroyWindow("Choose")

        return marker_rangers 
    

    def _disp_marker_pos(self,x,y,frame):
        _meassage = str(self._point_counter) + ":%d,%d" % (x, y) # _point_counter
        cv2.circle(frame, (x, y), 1, (255, 255, 255), thickness = -1)
        cv2.putText(frame, _meassage, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (255, 255, 255), thickness = 1)
        
        return frame

    def mouse_event(self,event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:
            if self.enable_maker_pos_acquirement:
                self._disp_marker_pos(x, y,frame)
                self.maker_position_frame0[self._point_counter] = [x,y]
                self._point_counter = self._point_counter + 1 if self._point_counter < self.num_maker_sets-1 else 0
            # print(self.maker_position_frame0)
            else:
                _meassage = "Please right click to start"
                cv2.putText(frame, _meassage, (50, 200), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), thickness = 1)
                
        if event == cv2.EVENT_RBUTTONDOWN:
            _meassage = 'Please Choose target points by left click:'
            cv2.putText(frame, _meassage, (50, 200), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), thickness = 1)
            self.enable_maker_pos_acquirement = True
            pass
        

        cv2.imshow("Choose", frame)
        return []

    def extract_angle(self, frame, whether_firstframe, calc_intersection):
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
        blue_pos = []
        
        if 1:
            # Iterate over each color marker
            for mask, thr, color, color_name, direction_vector in zip(masks, self.threshold_area_size, self.colors, colors_name, [direction_vector_0_1, None, None, None]        ):
                # Convert the mask to uint8
                
                mask = np.uint8(mask) # True/False -> 0/1
        
                # Find connected components in the mask
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

                # Filter regions based on area threshold
                filtered_regions = [index for index, stat in enumerate(stats[1:]) if stat[4] >= thr]
                
                    

                # Initialize a list to store points per mask
                point_per_mask = []

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
                    if color == (255,0,0): #blue
                        blue_pos.append((centroid_x, centroid_y))
                    if color == (0,127,0): #green 
                        self.temp_green_y = centroid_y
                    if color == (0,127,255):#yellow
                        self.temp_yellow_y = centroid_y                    
                
                # Visualize circles for each point in the mask
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
            angle_0 = self.calculate_angle(makerset_per_frame[0], makerset_per_frame[1], 0)
            angle_1 = self.calculate_angle(makerset_per_frame[1], makerset_per_frame[2], 1)
            angle_2 = self.calculate_angle(makerset_per_frame[2], makerset_per_frame[3], 2)
            
            #Find angles position by calculating intersections of lines 
            if calc_intersection:
                angle0_pos = self.calculate_intersection(makerset_per_frame[0], makerset_per_frame[1])
                angle1_pos = self.calculate_intersection(makerset_per_frame[1], makerset_per_frame[2])
                angle2_pos = self.calculate_intersection(makerset_per_frame[2], makerset_per_frame[3])
                self.b0 = blue_pos[0]
                self.b1 = blue_pos[1]
                d0 = AngleTracker.calc_distance(angle1_pos, blue_pos[0])
                d1 = AngleTracker.calc_distance(angle1_pos, blue_pos[1])
                print("d0 :", d0)
                print("d1 :", d1)

                if d0 < d1: #then blue_pos[1] is fingertip
                    fingertip_pos = np.array(blue_pos[1])
                    self.referenceFingertip = fingertip_pos
                    self.referenceAngle0 = blue_pos[0]
                    # angle0_pos = blue_pos[0]
                elif d0 > d1: #then blue_pos[0] is fingertip
                    fingertip_pos = np.array(blue_pos[0])
                    self.referenceFingertip = fingertip_pos
                    self.referenceAngle0 = blue_pos[1]

                    # angle0_pos = blue_pos[1]
                self.angle_pos = AngleTracker.relativise_AnglePos_toMCP(fingertip_pos, angle0_pos, angle1_pos, angle2_pos)
                # self.angle_pos = AngleTracker.rotate_frame_tracked_points(self.angle_pos[3], self.angle_pos[0], self.angle_pos[1], self.angle_pos[2])
                # self.fingertip_pos.append(self.angle_pos[0])
                # self.angle0_pos.append(self.angle_pos[1])
                # self.angle1_pos.append(self.angle_pos[2])
                # self.angle2_pos.append(self.angle_pos[3])
                self.fingertip_pos_list = np.vstack((self.fingertip_pos_list, self.angle_pos[0]))
                self.angle0_pos_list = np.vstack((self.angle0_pos_list, self.angle_pos[1]))
                self.angle1_pos_list = np.vstack((self.angle1_pos_list, self.angle_pos[2]))
                self.angle2_pos_list = np.vstack((self.angle2_pos_list, self.angle_pos[3]))
                print("fingertip pos:", fingertip_pos)
                print("angle0 pos:", angle0_pos)
                print("angle1 pos:", angle1_pos)
                print("angle2 pos:", angle2_pos)

            _text_pos_x = 100
            # Add text annotations to the frame with calculated angles
            frame = self.add_text_to_frame(frame, "ANGLE 0: {}".format((angle_0)), position=(_text_pos_x, 210), font_scale=1, thickness=2, color=(255, 255, 0))
            frame = self.add_text_to_frame(frame, "ANGLE 1: {}".format((angle_1)), position=(_text_pos_x, 240), font_scale=1, thickness=2, color=(255, 255, 0))
            frame = self.add_text_to_frame(frame, "ANGLE 2: {}".format((angle_2)), position=(_text_pos_x, 270), font_scale=1, thickness=2, color=(255, 255, 0))

        # except Exception as err:
        #     print(color_name,' Failed!:',err)
        #     return frame,[],[],[]
            if False:
                if whether_firstframe:
                    self.prev_angles = [angle_0, angle_1, angle_2]
                    self.current_angles = [angle_0, angle_1, angle_2]
                else: frame = self.detect_cv_error(frame, 30, angle_0, angle_1, angle_2)
                angle_0, angle_1, angle_2 = self.current_angles[0], self.current_angles[1], self.current_angles[2]

        return frame, angle_0, angle_1, angle_2
    
    def load_point_pos(self,):
        try:
            # JSON到字典转化
            _pos_file = open(self.video_pos_file_url, 'r')
            _pos_data = json.load(_pos_file)
            print(type(_pos_data),_pos_data)
            self.maker_position_frame0 = _pos_data['maker_position_frame0'] 
            if len(_pos_data) == 0: raise Exception("")        
            else: print("\tSuccessfully load calibration data!:",self.maker_position_frame0,"\n")

        except Exception as Err: 
            print("\tErr occurs when loading maker_position_frame0, Please calibrate angle sensor: \n",Err)
            # self.calibrateRange()
        pass

    def store_point_pos(self,):
        # save position of markers in video to json
        _data = self.maker_position_frame0
        _data = {'maker_position_frame0':_data}
        info_json = json.dumps(_data,sort_keys=True, indent=4, separators=(',', ': '))

        f = open(self.video_pos_file_url, 'w')
        f.write(info_json)
        # exit()
        pass

    def store_data(self,measure,set_fps=30):
        ## Store csv - raw_angles
        df_angle = pd.DataFrame(data=measure, columns=["frame", "angle_0", "angle_1", "angle_2"])
        df_angle["time"] = df_angle["frame"] / set_fps

        df_angle.to_csv(os.path.join(self.output_folder_path,f"{video_name.split('.')[0]}_extracted.csv"), 
                        index=False)

        np_data = np.array(measure)[:,::-1]
        print("measure:",type(measure))
        print(type(np_data),np_data)
        saveFigure(np_data,f"{video_name.split('.')[0]}_extracted.csv",["angle_2", "angle_1", "angle_0","frame"],
                   show_img=False, figure_mode='Single' )
        # saveData()

        pass

    def store_video(self,frames, fps):
        self.output_video_url = os.path.join(self.output_folder_path,f"{video_name.split('.')[0]}_extracted.mp4") 
        # Function to store the video with updated frames
        fourcc = cv2.VideoWriter_fourcc('M','J','P','G') # Win
        # fourcc = cv2.VideoWriter_fourcc(*'x264')# # 'avc1' # Mac
        print('fourcc built')
        height, width, _ = frames[0].shape
        out = cv2.VideoWriter(self.output_video_url, fourcc, fps, (width, height))
        for frame in frames:out.write(frame)
        out.release()
    
    def trim_frame(self, frame):
        # frame = frame[0:975, 0:1300]
        frame = frame[0:1200 , 0:1300]
        return frame
    
    def detect_cv_error(self,frame, error_detector, angle_0, angle_1, angle_2):
        self.current_angles = [angle_0, angle_1, angle_2]

        for i in range(len(self.current_angles)):
            err = self.current_angles[i] - self.prev_angles[i]
            if err < error_detector:
                self.prev_angles[i] = self.current_angles[i]
            else:
                self.current_angles[i] = self.prev_angles[i]
                frame = self.add_text_to_frame(frame, "cv error!", position=(150, 300), font_scale=1, thickness=2, color = (255, 255, 0))
        return frame
    
    def calculate_intersection(self, line0, line1): #Each line should be given as two coordinates like :line0 = [(x0, y0),(x1, y1)]
        # print("line0:", line0)
        # print("line1:", line1)
        
        x0 = line0[0][0]
        y0 = line0[0][1]
        x1 = line0[1][0]
        y1 = line0[1][1]
        a00 = y1 - y0
        a01 = x1 - x0
        b0 = x0*y1 - x1*y0

        x2 = line1[0][0]
        y2 = line1[0][1]
        x3 = line1[1][0]
        y3 = line1[1][1]
        a10 = y3 - y2
        a11 = x3 - x2
        b1 = x2*y3 - x3*y2
        A = np.array([[a00, -a01],
                      [a10, -a11]])
        B = np.array([b0, b1])

        x = np.linalg.solve(A, B) #intersection of lines

        # return 10*np.round(x/10)
        return x    
    
    @staticmethod
    def relativise_AnglePos_toMCP(fingertip_pos, angle0_pos, angle1_pos, angle2_pos):
        fingertip_pos = np.array(fingertip_pos - angle2_pos)
        fingertip_pos[1] = - fingertip_pos[1]
        angle0_pos = np.array(angle0_pos - angle2_pos)
        angle0_pos[1] = - angle0_pos[1]
        angle1_pos = np.array(angle1_pos - angle2_pos)
        angle1_pos[1] = - angle1_pos[1]

        angle2_pos = np.array([0, 0])
        # fingertip_pos = list(fingertip_pos - angle2_pos)
        # angle0_pos = list(angle0_pos - angle2_pos)
        # angle1_pos = list(angle1_pos - angle2_pos)
        # angle2_pos = [0, 0]
        return [fingertip_pos, angle0_pos, angle1_pos, angle2_pos]


    def save_trajectory(self):
        self.fingertip_pos_list = np.vstack(self.fingertip_pos_list)
        self.angle0_pos_list = np.vstack(self.angle0_pos_list)
        self.angle1_pos_list = np.vstack(self.angle1_pos_list)
        self.angle2_pos_list = np.vstack(self.angle2_pos_list)
        fig = plt.figure(layout="tight")
        scatter = True
        if scatter == True:
            L = len(self.fingertip_pos_list) - 1 
            value = np.linspace(0, 1, L)
            plt.scatter(self.fingertip_pos_list.T[0][1:], self.fingertip_pos_list.T[1][1:], label = 'fingertip', s=15)#, c=value, cmap='winter')
            plt.scatter(self.angle0_pos_list.T[0][1:], self.angle0_pos_list.T[1][1:], label ='angle0', s=15)#, c=value, cmap='winter')
            plt.scatter(self.angle1_pos_list.T[0][1:], self.angle1_pos_list.T[1][1:], label ='angle1', s=15)#, c=value, cmap='winter')
            plt.scatter(self.angle2_pos_list.T[0][1:], self.angle2_pos_list.T[1][1:], label ='angle2', s=20)#, c=value, cmap='winter')
        else:    
            plt.plot(self.fingertip_pos_list.T[0][1:], self.fingertip_pos_list.T[1][1:], label = 'fingertip')
            plt.plot(self.angle0_pos_list.T[0][1:], self.angle0_pos_list.T[1][1:], label ='angle0')
            plt.plot(self.angle1_pos_list.T[0][1:], self.angle1_pos_list.T[1][1:], label ='angle1')
            plt.plot(self.angle2_pos_list.T[0][1:], self.angle2_pos_list.T[1][1:], label ='angle2')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True)
        plt.show()


    @staticmethod
    def rotate_coordinates(points, angle, center=(0, 0)):
        """
        Rotate a set of coordinates around a specified center.

        Parameters:
        - points: A list of (x, y) coordinates to be rotated.
        - angle: The rotation angle in degrees.
        - center: The center of rotation. Default is (0, 0).

        Returns:
        - Rotated coordinates as a numpy array.
        """
        angle_rad = np.radians(angle)
        cos_theta = np.cos(angle_rad)
        sin_theta = np.sin(angle_rad)

        # Translate to the origin, rotate, and translate back
        centered_points = np.array(points) - np.array(center)
        rotated_points = np.dot(centered_points, np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]]))
        rotated_points += np.array(center)

        return rotated_points

    @staticmethod
    def rotate_frame_tracked_points(point_0_a, point_0_b, point_1_b, point_2_b, point_3_b):
        """
        Rotate a set of tracked points based on a reference point and a calculated angle.

        Parameters:
        - point_0_a (tuple): Coordinates of the reference point.
        - point_0_b (tuple): Coordinates of the first tracked point.
        - point_1_b (tuple): Coordinates of the second tracked point.
        - point_2_b (tuple): Coordinates of the third tracked point.
        - point_3_b (tuple): Coordinates of the fourth tracked point.

        Returns:
        - np.ndarray: An array containing the rotated coordinates of the tracked points.
        """
        # Create a new point along a horizontal line with the first point
        marker = (point_0_a[0] + 100, point_0_a[1])

        # Calculate the angle of rotation using CvHelper.calculate_angle function
        shift_angle = AngleTracker.calculate_angle(
            (
                point_0_a, point_0_b
            ),
            (
                point_0_a, marker
            )
        )
        # print(shift_angle)

        # Rotate points using CvHelper.rotate_coordinates function
        rotated_points = AngleTracker.rotate_coordinates([point_0_b, point_1_b, point_2_b, point_3_b], (180 - shift_angle) * -1,
                                                     point_0_a)

        # Shift points to central zero
        rotated_points[:, 0] = rotated_points[:, 0] - rotated_points[0, 0]
        rotated_points[:, 1] = rotated_points[:, 1] - rotated_points[0, 1]

        return rotated_points
    
    @staticmethod
    def calc_distance(point1, point2):
        """
        points shold be like: [x, y]
        
        this function returns distance of two points 
        """
        print("point1", point1)
        print("point2", point2)

        x0, y0 = point1[0], point1[1]
        x1, y1 = point2[0], point2[1]
        d = np.sqrt(((x0-x1)**2) + ((y0-y1)**2)  )
        return d
    


# %%
if __name__ == '__main__':
    import os,sys,json
    parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0,parentdir)

    from lib.GENERALFUNCTIONS import *

    # Constants
    ## For styling
    # colors = [(255,0,0), (127,0,255), (0,127,0), (0,127,255)]
    line_padding = [0.7, 1.5,1.5,1.5]

    font_scale = 1
    text_position_cnt = (100, 100)
    text_position_time = (100, 120)
    
    video_name = "sc01.mp4"
    frame_jump = 5
    ## For algorithm tuning
    # Are for optime
    kernel = np.ones((5,5),np.uint8)
    # threshold_area_size = [10, 10, 10, 10]# [80, 20, 10, 40]
    frame_shift = 0
    output_video_fps = 30 # I dont know if its work

    tracker = AngleTracker(video_name,denoising_mode = 'monocolor')
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
    whether_firstframe = True
    calc_intersection = True

    # Videos capture cycles
    while True:
        strt = time.time()
        ret, frame = cap.read()
        if not ret: break
        
        frame = tracker.trim_frame(frame)

        if cnt==frame_shift: tracker.acquire_marker_color(frame)

        if whether_firstframe:
            frame, angle_0, angle_1, angle_2  = tracker.extract_angle(frame, whether_firstframe, calc_intersection)
            whether_firstframe = False
        else:
            frame, angle_0, angle_1, angle_2  = tracker.extract_angle(frame, whether_firstframe, calc_intersection)
            if calc_intersection: print("fingertip(abs):", tracker.referenceFingertip)
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
    if calc_intersection:
        print("fingertip(to MCP):", tracker.fingertip_pos_list)
        print("angle1(to MCP):",tracker.angle1_pos_list)
        tracker.save_trajectory()
    else:
        tracker.store_data(measure,output_video_fps)
    print(tracker.video_pos_file_url)