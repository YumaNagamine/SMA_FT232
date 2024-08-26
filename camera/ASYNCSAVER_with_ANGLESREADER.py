# Debugged @ 20240726
# Created by Askar.Liu
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

    def finalize(self):
        if self.frame_queue:
            self.save_frame_batch(list(self.frame_queue))
        self.executor.shutdown(wait=True)
        self.out.release()

class AngleTracker(AsyncVideoSaver):
    def __init__(self, filename, fourcc, fps, frame_size, denoising_mode='monocolor'):
        super().__init__(self, filename, fourcc, fps, frame_size)

        self.color_mode = 0 # 0: Lab,1: Rgb
        self.num_maker_sets = 4
        self.denoising_mode = denoising_mode# 'monocolor'
        self.cv_preview_wd_name = "Video Preview"
        self.cv_choose_wd_name = "Choose"
        

        if self.color_mode ==0: # Lab
            self.maker_tolerance_L = 20#int(0.08 * 255)
            self.maker_tolerance_a = 30# int(0.09 * 255)# red -> green
            self.maker_tolerance_b = 30# int(0.09 * 255)# Yellow -> Blue
        else : # RGB
            self.maker_tolerance_L = int(0.5 * 255)
            self.maker_tolerance_a = int(0.2 * 255)# red -> green
            self.maker_tolerance_b = int(0.2 * 255)# Yellow -> Blue  

        self.marker_rangers = [ #[Low Lhigh alow ahigh blow bhigh]] # SC01
                       [ [100,220],[160,220],[60,160]], # Marker A
                       [ [100,215],[70,190],[30,80]], # Marker B
                       [ [180,210],[55,85],[120,180]], # Marker C
                       [ [150,235],[80,90],[100,120]], # Marker D 
                        ]
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
    
    def calculate_angle(self,line1, line2):
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
        except Exception as err:
            return []
        return angle_degrees
    
    @staticmethod
    def calculate_vector(point1, point2):
        return np.array(point2) - np.array(point1)    

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

    def acquire_marker_color(self): #TODO
        marker_rangers_old = self.marker_rangers
        marker_rangers = []

        num_marker_sets = self.num_maker_sets

        cv2.namedWindow(self.cv_choose_wd_name, cv2.WINDOW_GUI_EXPANDED)
        cv2.setMouseCallback(self.cv_choose_wd_name, object.mouse_event)

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
                self._disp_marker_pos(x,y,self.frame)
                # self.maker_position_frame0[self._point_counter] = [x,y]
                self._point_counter = self._point_counter + 1 if self._point_counter < self.num_maker_sets-1 else 0

            pass
        self._point_counter = 0

        cv2.putText(self.frame,_meassage, (40, 50), cv2.FONT_HERSHEY_DUPLEX,
                    1.8, (255, 255,255), thickness =2)
        cv2.putText(self.frame 'Press Esc to continue on point extraction', (40, 80), cv2.FONT_HERSHEY_DUPLEX,
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
            upper_limit = frame_to_segment[_pos[1]][_pos[0]] + [self.maker_tolerance_L, self.maker_tolerance_a, self.maker_tolerance_b]  
            lower_limit = frame_to_segment[_pos[1]][_pos[0]] - [self.maker_tolerance_L, self.maker_tolerance_a, self.maker_tolerance_b]
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

    def _disp_marker_pos(self,x,y):
        _meassage = str(self._point_counter) + ":%d,%d" % (x, y) # _point_counter
        cv2.circle(self.frame, (x, y), 1, (255, 255, 255), thickness = -1)
        cv2.putText(self.frame, _meassage, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (255, 255, 255), thickness = 1)
        
        return self.frame

    def mouse_event(self,event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:
            if self.enable_maker_pos_acquirement:
                self._disp_marker_pos(x, y)
                self.maker_position_frame0[self._point_counter] = [x,y]
                self._point_counter = self._point_counter + 1 if self._point_counter < self.num_maker_sets-1 else 0
            # print(self.maker_position_frame0)
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



if __name__ == "__main__":

    print('Running on env: ',sys.version_info)
    
    ## Create CAM obj
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

    if is_recod_video: saver = AsyncVideoSaver(video_file_name, fourcc, target_fps, resolution)
    frame_id = 0
    time_cv_st = time.perf_counter()
    
    # 初始化时间戳队列
    frame_times = deque(maxlen=30)  # 保持最近30帧的时间戳
  
    # Video Loop
    while True:
        cur_time = time.perf_counter()
        ret, frame_raw = cap.read()

        if ret:
            if is_recod_video: saver.add_frame(frame_raw)
            # Convert the frame to PIL format
            # frame = cv2.cvtColor(frame_BGR, cv2.COLOR_BGR2RGB)
            # frame = Image.fromarray(frame)

            # Resize the image to fit the label
            # frame = frame.resize((640, 360)) #640, 360 1280,720
            pass
        else: continue
        frame_id += 1
        frame_times.append(cur_time)

        if True: #frame_id % int(actual_fps // 20) == 0:  # 每示两次
 
            if frame_id>30: cur_fps = len(frame_times) / (frame_times[-1] - frame_times[0])
            else : cur_fps = -1

            cv2.putText(frame_raw, f'Time: {time.strftime("%Y%m%d-%H%M%S")},{frame_times[-1] }', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame_raw, f'Current Frame {frame_id}; FPS: {int(cur_fps)}', (10,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow('frame', frame_raw)  # 显示图像
            

        if cv2.waitKey(1) & 0xFF == ord('q'):  # 按'q'键退出
                break
 
    cap.release()
    cv2.destroyAllWindows()
    if is_recod_video: saver.finalize()
