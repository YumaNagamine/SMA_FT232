import numpy as np
import time
import membership_function as mf
# from cv_angle_traking.angles_reader_for_new_finger import AngleTracker
from camera.NOGUI_ASYNCSAVER_with_ANGLESREADER import AsyncVideoSaver, AngleTracker
# By searching for "adjust" (Ctrl + F) you can find parameters to adjust

# output vector du assumes [flexor0, flexor1, mainextensor, subextensor0, subextensor1, ooo, ooo]

class FUZZYCONTROL():
    def __init__(self):
        pass

    def input_target(self, current_angles):
        remain_upper_limit = 5 #adjust these parameters
        remain_lower_limit = -5
        flag = True
        err = [0,0,0]
        #setting target of angle0
        while True:
            print("Satisfy 90 < angle0 < 180")
            angle0 = int(input("angle0: "))
            err[0] = angle0 - current_angles[0]
            print('err',err)
            if not 90 < angle0 < 180:
                print("This angle is impossible. Try again...")
                print()
                continue
            if remain_lower_limit < err[0] < remain_upper_limit:
                print("Target angle is considered as the same as current angle. ")
                print("Skip setting target angle1... ")
                print()
                flag = False
                DIP_PIP_mode = 'remain'
                break 
            if 90 < angle0 < 180:
                if err[0] > remain_upper_limit:
                    DIP_PIP_mode = 'extend'
                elif err[0] < remain_lower_limit:
                    DIP_PIP_mode = 'flex'
                else:
                    print("Find some bags... Force-quiting..")
                    return
                print("error01 mode: ", DIP_PIP_mode)
                break
        #setting target of angle1
        if flag:
            if DIP_PIP_mode == 'flex':
                while True:
                    print("Satisfy 90 < angle1 < ", current_angles[1]+remain_lower_limit)
                    angle1 = int(input("angle1 :"))
                    if 90  < angle1 < current_angles[1] + remain_lower_limit:
                        err[1] = angle1 - current_angles[1]
                        break
                    else:
                        print("This angle is impossible. Try again...")
                        print()

            elif DIP_PIP_mode == 'extend':
                while True:
                    print(f"Satisfy {current_angles[1]+remain_upper_limit} < angle1 < 180")
                    angle1 = int(input("angle1 :"))
                    if current_angles[1] + remain_upper_limit < angle1 < 180:
                        err[1] = angle1 - current_angles[1]
                        break
                    else: 
                        print("This angle is impossible. Try again...")
                        print()
            
        #setting target of angle2
        while True:
            print("Satisfy 90 < angle2 < 250")
            angle2 = int(input("angle2: "))
            err[2] = angle2 - current_angles[2]
            if remain_lower_limit < err[2] < remain_upper_limit:
                print("Target angle is considered as the same as current angle. ")
                MP_mode = 'remain'                
                break
            if 90 < angle2 < 250:
                if err[2] < remain_lower_limit:
                    MP_mode = 'flex'
                elif err[2] > remain_upper_limit:
                    MP_mode = 'extend'
                break
            else:
                print("This angle is impossible. Try again...")
                print()
        
        self.mode = [DIP_PIP_mode, MP_mode]
        self.err = err
        print("mode=", self.mode)
        print("errors= ", err)
        return np.array([angle0, angle1, angle2])

    def control_method(self, err):
        if self.mode[0] == 'flex' and self.mode[1] == 'flex':
            self.du = self.controlmethod_FF(err)
            
        elif self.mode[0] == 'flex' and self.mode[1] == 'extend':
            self.du = self.controlmethod_FE(err)

        elif self.mode[0] == 'flex' and self.mode[1] == 'remain':
            self.du = self.controlmethod_FR(err)

        elif self.mode[0] == 'extend' and self.mode[1] == 'flex':
            self.du = self.controlmethod_EF(err)

        elif self.mode[0] == 'extend' and self.mode[1] == 'extend':
            self.du =  self.controlmethod_EE(err)

        elif self.mode[0] == 'extend' and self.mode[1] == 'remain':
            self.du = self.controlmethod_ER(err)
            
        elif self.mode[0] == 'remain' and self.mode[1] == 'flex':
            self.du = self.controlmethod_RF(err)

        elif self.mode[0] == 'remain' and self.mode[1] == 'extend':
            self.du = self.controlmethod_RE(err)

        elif self.mode[0] == 'remain' and self.mode[1] == 'remain':
            self.du = self.controlmethod_RR(err)
        
    def controlmethod_FF(self,err):
        # err nust be: [angle0, angle1, angle2]
        err = np.array(err)        
        du = np.array([0,0,0,0,0,0,0])
        # adjust following parameters
        du_min = -1
        du_max = 1
        param_tri = np.array([[], [], []], # for angle0,1
                             [[], [], []]) # for angle2
        param_up = np.array([[], []], # for angle0,1
                            [[], []]) # for angle2
        param_down = np.array([[],[]], # for angle0,1
                              [[],[]]) # for angle2
        membership_degree_angle0 = mf.normal_three_membership(err[0], param_tri[0], param_up[0], param_down[0])
        membership_degree_angle1 = mf.normal_three_membership(err[1], param_tri[0], param_up[0], param_down[0])
        membership_degree_angle2 = mf.normal_three_membership(err[2], param_tri[1], param_up[1], param_down[1])
        membership_degree = np.vstack(membership_degree_angle0, membership_degree_angle1, membership_degree_angle2)
        
        # adjust following parameters
        # for flexor0
        param_output_0 = np.array([[],[],[]], # left triangle
                                  [[],[],[]], # middle triangle
                                  [[],[],[]]) # right triangle
        # for flexor1
        param_output_1 = np.array([[],[],[]], # left triangle
                                  [[],[],[]], # middle triangle
                                  [[],[],[]]) # right triangle
        weights_flexor0 = np.array([])
        weights_flexor1 = np.array([])
        membership_degree_flexor0 = mf.weighting(weights_flexor0, membership_degree)
        membership_degree_flexor1 = mf.weighting(weights_flexor1, membership_degree)

        
        # get arrays of output_membership
        fine = 100 # can be adjusted. The bigger, the greater calculation cost. 
        number_of_step = (du_max-du_min)*fine  
        x = np.linspace(du_min, du_max, num=number_of_step)
        y0 = np.vectorize(mf.triangle_func)(x, param_output_0[0][0], param_output_0[0][1],param_output_0[0][2]) #left : du < 0 
        y1 = np.vectorize(mf.triangle_func)(x, param_output_0[1][0], param_output_0[1][1],param_output_0[1][2]) #middle: du ~ 0
        y2 = np.vectorize(mf.triangle_func)(x, param_output_0[2][0], param_output_0[2][1],param_output_0[2][2]) #right: du > 0
        y0 = np.minimum(membership_degree_flexor0[1],y0)
        y1 = np.minimum(membership_degree_flexor0[0],y1)
        y2 = np.minimum(membership_degree_flexor0[2],y2)

        du[0] = mf.calc_centroid(x, y0, y1, y2) # flexor0 output
        
        y0 = np.vectorize(mf.triangle_func)(x, param_output_1[0][0], param_output_1[0][1],param_output_1[0][2])
        y1 = np.vectorize(mf.triangle_func)(x, param_output_1[1][0], param_output_1[1][1],param_output_1[1][2])
        y2 = np.vectorize(mf.triangle_func)(x, param_output_1[2][0], param_output_1[2][1],param_output_1[2][2])
        y0 = np.minimum(membership_degree_flexor1[1],y0)
        y1 = np.minimum(membership_degree_flexor1[0],y1)
        y2 = np.minimum(membership_degree_flexor1[2],y2)

        du[1] = mf.calc_centroid(x, y0, y1, y2) # flexor1 output

        return du
        


    def deFuzzy_process(x,):
        
        return
    
    @staticmethod
    def limit_dutyratio(dutyratio, upperlimit):
        return np.clip(dutyratio, None, upperlimit)
    
if __name__ == "__main__":
    import os,sys
    import cv2
    parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0,parentdir)
    from lib.GENERALFUNCTIONS import *
    from collections import deque


    os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin")

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

    if fourcc == 'X264':  
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
    filename = 'no meaning'
    fuzzy = FUZZYCONTROL()
    tracker = AngleTracker(video_file_name, fourcc, target_fps, resolution, 'monocolor')
    control = True


    while True:
        cur_time = time.perf_counter()
        ret, frame_raw = cap.read()
        if ret:
            if is_recod_video: saver.add_frame(frame_raw)

            if whether_firstframe:
                saver.acquire_marker_color()
                firstangle = [saver.first_angles[1], saver.first_angles[2], saver.first_angles[3]]
                print('first angle:',firstangle)
                target = fuzzy.input_target(firstangle)
                whether_firstframe = False
            
            if True:
                noneedframe, angle_0, angle_1, angle_2 = saver.extract_angle()
                angles = np.array([angle_0, angle_1, angle_2])
                currrent_error =  angles - target
                cv2.imshow('Video Preview', saver.frame)
            
            frame_id += 1
            frame_times.append(cur_time)


            if True:
                if frame_id > 45: cur_fps = 30 / (cur_time - frame_times[0])
                else : cur_fps = -1
                cv2.putText(frame_raw, f'Time: {time.strftime("%Y%m%d-%H%M%S")},{cur_time}',
                             (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                
                cv2.putText(frame_raw, f'Current Frame {frame_id}; FPS: {int(cur_fps)}',
                             (10,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                
            if control:
                du = fuzzy.control_method(err)

        if cv2.waitKey(1) & 0xFF == ord('q'): break
 

    cap.release()
    cv2.destroyAllWindows()
    if is_recod_video: saver.finalize()