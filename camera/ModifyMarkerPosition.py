# This file is created by Nagamine on 24th March, copied NOGUI_ASYNCSAVER_with_ANGLESREADER
# to modify marker positions and also save raw marker position data to CSV file for a new finger
# FOR 

if __name__=='__main__': # Test codes # Main process
    import os,sys
    # import pyftdi.spi as spi
    parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0,parentdir)

    from lib.GENERALFUNCTIONS import *

import os
os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin")

import cv2
import ttkbootstrap as ttk
from PIL import Image, ImageTk
from lib.GENERALFUNCTIONS import *
import tkinter as tk
from collections import deque
import concurrent.futures
import pandas as pd
 

class AsyncVideoSaver(object):
    def __init__(self, filename, fourcc, fps, frame_size):
        self.filename = filename
        self.fourcc =  fourcc  # XVID I420 3IVD
        self.fps = fps
        self.frame_size = frame_size
        self.maxlen = 30
        self.frame_queue = deque()

        try:
            self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)
            self.out = cv2.VideoWriter(self.filename, cv2.VideoWriter_fourcc(*fourcc), self.fps, self.frame_size)
            print("Saving Video file:",filename," in ", )
        except Exception as err :
            print('cv2.VideoWeiter creation failed!!')
            print('Err occurs:',err)
            exit()


    def save_frame_batch(self, frames):
        for frame in frames:
            # _frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            self.out.write(frame)

    def add_frame(self, frame):
        self.frame_queue.append(frame)
        
        if len(self.frame_queue) >= self.maxlen-2:  # 批量处理
            frames_to_save = list(self.frame_queue)
            self.frame_queue.clear()
            self.executor.submit(self.save_frame_batch, frames_to_save)
        
        self.frame = frame 

    def input_frame(self, frame):
        self.frame = frame

    def finalize(self):
        if self.frame_queue:
            self.save_frame_batch(list(self.frame_queue))
        self.executor.shutdown(wait=True)
        self.out.release()

class AngleTracker(AsyncVideoSaver):
    def __init__(self, filename, fourcc, fps, frame_size, denoising_mode='monocolor'):
        super().__init__(filename, fourcc, fps, frame_size)

        self.color_mode = 0 # 0: Lab,1: Rgb
        self.num_maker_sets = 4
        self.denoising_mode = 0# denoising_mode# 'monocolor'
        self.cv_choose_wd_name = "Choose"
        self.threshold_area_size = [50, 30, 50, 150]
        self.colors = [(255,0,0), (127,0,255), (0,127,0), (0,127,255)] 
        self.video_pos_file_url = "C:/Users/SAWALAB/Desktop/projects/SMA_FT232/LOG/realtime/pos.json"

        if self.color_mode == 0: # Lab
            self.maker_tolerance_L = [75,50,20,20]#int(0.08 * 255)q
            self.maker_tolerance_a = [30,45,17,17]# int(0.09 * 255)# red -> green
            self.maker_tolerance_b = [40,20,15,30]# int(0.09 * 255)# Yellow -> Blue
        else : # RGB
            self.maker_tolerance_L = int(0.5 * 255)
            self.maker_tolerance_a = int(0.2 * 255)# red -> green
            self.maker_tolerance_b = int(0.2 * 255)# Yellow -> Blue  

        self.marker_rangers = [ #[Low Lhigh alow ahigh blow bhigh]] # SC01
                       [ [35,142],[132,163],[56,92]], # Marker A
                       [ [118,189],[139,187],[137,212]], # Marker B
                       [ [87,134],[76,108],[106,132]], # Marker C
                       [ [104,175],[109,127],[149,232]], # Marker D 
                        ] # under the condition of light: 25%
        self._point_counter = 0
        self.maker_position_frame0 = []
        for _ in range(self.num_maker_sets):self.maker_position_frame0.append([0,0])

        self.enable_maker_pos_acquirement = False
        self.load_point_pos()

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
                if self.temp_purple_y < self.temp_yellow_y: # 90 < angle2 < 180
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
        if index == 0 or index == 1:
            if angle_degrees > 180:
                angle_degrees = 360 - angle_degrees

        return angle_degrees
    
    @staticmethod
    def calculate_vector(point1, point2):
        return np.array(point2) - np.array(point1)    

    def segment_marker_by_color(self,frame_tmp): # Askar.L # used in extract_angle
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

    def acquire_marker_color(self): #TODO
        marker_rangers_old = self.marker_rangers
        marker_rangers = []
        num_marker_sets = self.num_maker_sets
        
        cv2.namedWindow(self.cv_choose_wd_name, cv2.WINDOW_GUI_EXPANDED)
        cv2.setMouseCallback(self.cv_choose_wd_name, self.mouse_event)
    
        if self.color_mode == 0:
            frame_to_segment = cv2.cvtColor(self.frame, cv2.COLOR_RGB2Lab)
        else: frame_to_segment = self.frame
        
        if self.enable_maker_pos_acquirement:
            # frame = add_text_to_frame(frame,'Please choose',position=(40, 50),color=(255, 255,255),font_scale=1)
            _meassage = 'Choose '+ str(num_marker_sets)+' position for the marker'
        else:
            _meassage = "Loaded exsist maker position:"

            print(self.maker_position_frame0)
            for [x,y] in self.maker_position_frame0:
                self._disp_marker_pos(x,y)
                # self.maker_position_frame0[self._point_counter] = [x,y]
                self._point_counter = self._point_counter + 1 if self._point_counter < self.num_maker_sets-1 else 0

            pass
        self._point_counter = 0

        cv2.putText(self.frame,_meassage, (40, 50), cv2.FONT_HERSHEY_DUPLEX,
                    1.8, (255, 255,255), thickness =2)
        cv2.putText(self.frame, 'Press Esc to continue on point extraction', (40, 80), cv2.FONT_HERSHEY_DUPLEX,
                    1, (255, 255,255), thickness =1)
        while not self.frame is None:
            cv2.imshow('Choose', self.frame)
            _key_pressed = cv2.waitKey(1)

            if  _key_pressed & 0xFF == 27: break
            elif _key_pressed == ord('s'):
                _meassage = 'Saving:'
                cv2.putText(self.frame, _meassage, (50, 200), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), thickness = 1)
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

        print('marker_rangers:',marker_rangers)
        self.marker_rangers = marker_rangers
        # os.system('pause')
        cv2.destroyWindow("Choose")

        self.first_angles = self.extract_angle()

        return marker_rangers 

    def extract_angle(self):
        # Convert the input frame to the CIELAB color space

        cielab_frame = cv2.cvtColor(self.frame, cv2.COLOR_RGB2Lab)

        # Segment markers by color in the CIELAB color space
        [marker_blue, marker_pink, marker_green, marker_yellow] = self.segment_marker_by_color(cielab_frame)


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
            for mask, thr, color, color_name, direction_vector in zip(masks, self.threshold_area_size, self.colors, colors_name, [direction_vector_0_1, None, None, None]        ):
                # Convert the mask to uint8
                
                mask = np.uint8(mask) # True/False -> 0/1
        
                # Find connected components in the mask
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

                # Filter regions based on area threshold
              
                filtered_regions = [index for index, stat in enumerate(stats[1:]) if stat[4] >= thr]
                # stat ;(左上の x 座標, 左上の y 座標, 幅, 高さ, 面積)
                    

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
 
                    if color == (0,127,0): #purple
                        self.temp_purple_y = centroid_y
                    if color == (0,127,255): #yellow
                        self.temp_yellow_y = centroid_y
                
                # Visualize circles for each point in the mask
                for idx, point in enumerate(point_per_mask):
                    cv2.circle(self.frame, (point[0], point[1]), radius=idx * 10, color=color, thickness=2)

                # Visualize circles for each point with increased radius
                for idx, point in enumerate(point_per_mask):
                    cv2.circle(self.frame, (point[0], point[1]), radius=idx * 10 + 10, color=color, thickness=3)

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
                cv2.line(self.frame, point1, point2, color, 3)

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

            _text_pos_x = 100
            # Add text annotations to the frame with calculated angles
            self.frame = self.add_text_to_frame(self.frame, "ANGLE 0: {}".format((angle_0)), position=(_text_pos_x, 210), font_scale=1, thickness=2, color=(255, 255, 0))
            self.frame = self.add_text_to_frame(self.frame, "ANGLE 1: {}".format((angle_1)), position=(_text_pos_x, 240), font_scale=1, thickness=2, color=(255, 255, 0))
            self.frame = self.add_text_to_frame(self.frame, "ANGLE 2: {}".format((angle_2)), position=(_text_pos_x, 270), font_scale=1, thickness=2, color=(255, 255, 0))

        # except Exception as err:
        #     print(color_name,' Failed!:',err)
        #     return frame,[],[],[]


        return self.frame, angle_0, angle_1, angle_2

    def _disp_marker_pos(self,x,y):
        _meassage = str(self._point_counter) + ":%d,%d" % (x, y) # _point_counter
        cv2.circle(self.frame, (x, y), 1, (255, 255, 255), thickness = -1)
        cv2.putText(self.frame, _meassage, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (255, 255, 255), thickness = 1)
        
        return self.frame

    def mouse_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.enable_maker_pos_acquirement:
                self._disp_marker_pos(x, y)
                self.maker_position_frame0[self._point_counter] = [x,y]
                self._point_counter = self._point_counter + 1 if self._point_counter < self.num_maker_sets-1 else 0
                print(self.maker_position_frame0)
            else:
                _meassage = "Please right click to start"
                cv2.putText(self.frame, _meassage, (50, 200), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), thickness = 1)
                
        if event == cv2.EVENT_RBUTTONDOWN:
            _meassage = 'Please Choose target points by left click:'
            cv2.putText(self.frame, _meassage, (50, 200), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), thickness = 1)
            self.enable_maker_pos_acquirement = True
            pass
    
        cv2.imshow("Choose", self.frame)
        return []
    
    def load_point_pos(self,):
        import json
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
        import json
        # save position of markers in video to json
        _data = self.maker_position_frame0
        _data = {'maker_position_frame0':_data}
        info_json = json.dumps(_data,sort_keys=True, indent=4, separators=(',', ': '))

        f = open(self.video_pos_file_url, 'w')
        f.write(info_json)
        # exit()
        pass

    # def store_data(self,measure,set_fps=30):
    #     ## Store csv - raw_angles
    #     df_angle = pd.DataFrame(data=measure, columns=["frame", "angle_0", "angle_1", "angle_2"])
    #     df_angle["time"] = df_angle["frame"] / set_fps

    #     df_angle.to_csv(os.path.join(self.output_folder_path,f"{video_name.split('.')[0]}_extracted.csv"), 
    #                     index=False)

    #     np_data = np.array(measure)[:,::-1]
    #     print("measure:",type(measure))
    #     print(type(np_data),np_data)
    #     saveFigure(np_data,f"{video_name.split('.')[0]}_extracted.csv",["angle_2", "angle_1", "angle_0","frame"],
    #                show_img=False, figure_mode='Single' )
    #     # saveData()

    #     pass

    # def store_video(self,frames, fps):
    #     self.output_video_url = os.path.join(self.output_folder_path,f"{video_name.split('.')[0]}_extracted.mp4") 
    #     # Function to store the video with updated frames
    #     fourcc = cv2.VideoWriter_fourcc('M','J','P','G') # Win
    #     # fourcc = cv2.VideoWriter_fourcc(*'x264')# # 'avc1' # Mac
    #     print('fourcc built')
    #     height, width, _ = frames[0].shape
    #     out = cv2.VideoWriter(self.output_video_url, fourcc, fps, (width, height))
    #     for frame in frames:out.write(frame)
    #     out.release()

class ModifyMarkerPositions(AngleTracker):
    def __init__(self, filename, fourcc, fps, frame_size, denoising_mode='monocolor'):
        super.__init__(filename, fourcc, fps, frame_size, denoising_mode='monocolor')
        pass
    def set_params(self, theta, distance, shift):
        self.theta = theta
        self.distance = distance
        self.shift = shift

    def rotate_vector(self):
        pass
    def shift_vector(self):
        pass
    
    def marker_discriminator_distalis(): #末節骨のマーカーを区別する　[遠位のマーカー　近位のマーカー]にする
        pass
    def marker_discriminator(): #中節骨と基節骨のマーカーを区別する　上と同様
        pass
    def calculate_angle(self, line1, line2, index): #over ride
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
if __name__ == "__main__":

    cam_num =  0
    
    is_lighting = True
    is_recod_video = True    
    cam_name = 'AR0234' # 'OV7251' #  
    
    cap = cv2.VideoCapture(cam_num,cv2.CAP_DSHOW)  #cv2.CAP_DSHOW  CAP_WINRT

    
    if cam_name == 'AR0234': # Aptina AR0234
        target_fps = 90
        resolution =  (1600,1200)#(1920,1200)#q(800,600)# (800,600)#(1920,1200) (1280,720)#
        width, height = resolution

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0]) 
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        # Set FPS
        cap.set(cv2.CAP_PROP_FPS,target_fps)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG')) # 'I420'
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)  # 设置缓冲区大小为2
        
        if is_lighting:            # 曝光控制
            # 设置曝光模式为手动
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # 0.25表示手动模式，0.75表示自动模式
            cap.set(cv2.CAP_PROP_GAIN, 0)  # 调整增益值，具体范围取决于摄像头
            cap.set(cv2.CAP_PROP_EXPOSURE, -11)  # 设置曝光值，负值通常表示较短的曝光时间
        else:            
            cap.set(cv2.CAP_PROP_GAIN, 0)  # 调整增益值，具体范围取决于摄像头
            cap.set(cv2.CAP_PROP_EXPOSURE, -3)  # 设置曝光值，负值通常表示较短的曝光时间
        # Save video
        fourcc = 'X264'#'MJPG' # 'I420' X264

    elif cam_name == 'OV7251': # Grayscale
        target_fps = 120
        resolution =  (640,480) # (640,480)
        width, height = resolution
        # cap.set(cv2.CAP_PROP_CONVERT_RGB,0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0]) 
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

        # Set FPS
        cap.set(cv2.CAP_PROP_FPS,target_fps)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG')) # 'I420'
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)  # 设置缓冲区大小为2
        
        # # 曝光控制
        cap.set(cv2.CAP_PROP_GAIN, -0.5)  # 调整增益值，具体范围取决于摄像头
        cap.set(cv2.CAP_PROP_EXPOSURE, -20)  # 设置曝光值，负值通常表示较短的曝光时间

        fourcc = 'MJPG' 

    elif cam_name == 'Oneplus': 
       
        target_fps = 480
        resolution = (1280,720) #q(800,600)# (800,600)#(1920,1200) (1280,720)#
        width, height = resolution 
        cap = cv2.VideoCapture(2)
 
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0]) 
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        # Set FPS
        cap.set(cv2.CAP_PROP_FPS,target_fps)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG')) # 'I420'
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)  # 设置缓冲区大小为2
        
        # 曝光控制
        cap.set(cv2.CAP_PROP_GAIN, 4)  # 调整增益值，具体范围取决于摄像头
        cap.set(cv2.CAP_PROP_EXPOSURE, -10)  # 设置曝光值，负值通常表示较短的曝光时间

        fourcc = 'X264'

        pass

    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Target FPS: {target_fps}, Actual FPS: {actual_fps}")
    if fourcc == 'MJPG':
        video_file_name = 'IMG/video/' +cam_name +'_' + time.strftime("%m%d-%H%M%S")  + '.avi'
    elif fourcc == 'X264':
        video_file_name = 'IMG/video/' +cam_name +'_' + time.strftime("%m%d-%H%M%S")  + '.mp4'
    elif fourcc == 'XVID':
        video_file_name = 'IMG/video/' +cam_name +'_' + time.strftime("%m%d-%H%M%S")  + '.avi'
    elif fourcc == 'H265': # BUG
        video_file_name = 'IMG/video/' +cam_name +'_' + time.strftime("%m%d-%H%M%S")  + '.mp4'

    if is_recod_video: saver = AngleTracker(video_file_name,fourcc, target_fps, resolution, 'monocolor')
    
    frame_id = 0
    whether_firstframe = True
    
    # 初始化时间戳队列^^^^---^
    frame_times = deque(maxlen=30)  # 保持最近30帧的时间戳
    
    cv_preview_wd_name = 'Video Preview'
    # cv_choose_wd_name = 'Choose'
    # colors = [(255,0,0), (127,0,255), (0,127,0), (0,127,255)]
    cv2.namedWindow(cv_preview_wd_name, cv2.WINDOW_GUI_EXPANDED)
    cv2.namedWindow("Mask",cv2.WINDOW_GUI_EXPANDED)

    while True: # Video Loop // 90 Hz
        cur_time = time.perf_counter()
        ret, frame_raw = cap.read()   

        if ret:

            if is_recod_video: saver.add_frame(frame_raw)
            
            if whether_firstframe: 
                saver.acquire_marker_color()
                whether_firstframe = False

            frame_id += 1
            frame_times.append(cur_time)

            if True: #read angles
                noneedframe, angle_0, angle_1, angle_2  = saver.extract_angle()
                angles = [angle_0, angle_1, angle_2]
                # print(angles)
                # process_share_dict['angles'] = [angle_0, angle_1, angle_2]
                # print(process_share_dict['angles'])
                cv2.imshow('Video Preview', saver.frame)

            if True: #frame_id % int(actual_fps // 20) == 0:  # 每S两次
                if frame_id > 45: cur_fps = 30 / (cur_time - frame_times[0])
                else : cur_fps = -1
                cv2.putText(frame_raw, f'Time: {time.strftime("%Y%m%d-%H%M%S")},{cur_time}',
                             (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                
                cv2.putText(frame_raw, f'Current Frame {frame_id}; FPS: {int(cur_fps)}',
                             (10,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                
            # if frame_id % int(actual_fps // 20) == 0:  # 每S两次
                # deepcopy. frame_raw(800,600) 
                # frame_raw_copied = frame_raw.deepcopy()
                
                # copied_frame = copy.deepcopy(frame_raw)
                # cv2.resize(frame_raw.copy(), (800,600))

                # copied_frame = cv2.resize(copied_frame,(720,480))
                
                # image_to_send = Image.fromarray(cv2.cvtColor(frame_raw.copy(), cv2.COLOR_BGR2RGB))
                # image_to_send = Image.fromarray(cv2.cvtColor(copied_frame, cv2.COLOR_BGR2RGB))
                
                # process_share_dict['photo'] = image_to_send
                # process_share_dict['photo_acquired_t'] = time.time()
                
                
            pass        
                 

        if cv2.waitKey(1) & 0xFF == ord('q'):  # 按'q'键退出
            break
        
    cap.release()
    cv2.destroyAllWindows()
    if is_recod_video: saver.finalize()
