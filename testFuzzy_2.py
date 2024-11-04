# I would like to thank following libraries:
import numpy as np
import time
import membership_function as mf
# from cv_angle_traking.angles_reader_for_new_finger import AngleTracker
from camera.NOGUI_ASYNCSAVER_with_ANGLESREADER import AsyncVideoSaver, AngleTracker
from SMA_finger.SMA_finger_MP import *

# By searching for "adjust" (Ctrl + F) you can find parameters to adjust

# output vector du assumes [flexor0, flexor1, mainextensor, subextensor0, subextensor1, ooo, ooo]

class FUZZYCONTROL():

    def __init__(self):
        self.actuator_device = []
        self.channels = PWMGENERATOR.CH_EVEN

        self.output_levels = np.zeros(7)

        self.connect()
        self.angle_history = np.zeros(4)

        print("\n\nFuzzy contorller established")

        # test update speed
        _st_ = time.perf_counter()
        for _i in range(100):
            for channels, ch_DR in zip(self.channels, self.output_levels):
                self.actuator_device.setDutyRatioCH(channels, ch_DR, relax=False)
        _t = 1/(-(_st_ - time.perf_counter())/100)
        print("Tested PWM output update rate:",_t ,"Hz")

        pass
    def connect(self):
        url_test_len = 4
        actuator_device = []
        for _i in range(url_test_len):
            _url = os.environ.get('FTDI_DEVICE', 'ftdi:///1')
            try:
                print("\nConnecting:")
                actuator_device = ctrlProcess(_url, 'ADC001')
            except Exception as err:
                print(err)
            if actuator_device:
                self.actuator_device = actuator_device
                break

    def apply_DR(self, retry=True):
        # for ch_DR in self.output_levels:
        #     val = ch_DR
        #     ch_DR.set(val)
        try:
            for channels, ch_DR in zip(self.channels, self.output_levels):
                self.actuator_device.setDutyRatioCH(channels, ch_DR, relax=False)
        except AttributeError:
            print("NO CONNECTION!")
            if retry: print('Auto connecting!'); self.connect()

    def stop_DR(self):
        print('set all DR zero!')
        for i in range(len(self.output_levels)): self.output_levels[i] = 0
        # for ch in self.out_levels: ch.set(0)

        self.apply_DR(retry=False)


    def input_target(self, current_angles):
        remain_upper_limit = 1 #adjust these parameters
        remain_lower_limit = -1
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
        self.target = np.array([angle0, angle1, angle2])
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

        self.output_levels = np.array(self.output_levels + self.du)
        self.output_levels = self.limit_dutyratio(self.output_levels, 0.4)
        

    def controlmethod_FF(self,err):
        # err nust be: [angle0, angle1, angle2]
        err = np.array(err)        
        du = np.zeros(7, dtype=np.float32)
        # adjust following parameters
        du_min = -0.1
        du_max = 0.1
        param_tri = np.array([[[-45,0], [0,1], [45,0]], # for angle0,1
                             [[-80,0], [0,1], [80,0]]]) # for angle2
        param_up = np.array([[[0,0], [90,1]], # for angle0,1
                            [[0,0], [160,1]]]) # for angle2
        param_down = np.array([[[-90,1],[0,0]], # for angle0,1
                              [[-160,1],[0,0]]]) # for angle2
        membership_degree_angle0 = mf.normal_three_membership(err[0], param_tri[0], param_up[0], param_down[0])
        membership_degree_angle1 = mf.normal_three_membership(err[1], param_tri[0], param_up[0], param_down[0])
        membership_degree_angle2 = mf.normal_three_membership(err[2], param_tri[1], param_up[1], param_down[1])
        membership_degree = np.vstack((membership_degree_angle0, membership_degree_angle1, membership_degree_angle2))


        # adjust following parameters
        # for flexor0
        param_output_0 = np.array([[[-0.05, 0],[-0.025, 1],[0,0]], # left triangle
                                  [[-0.03,0],[0,1],[0.03, 0]], # middle triangle
                                  [[0,0],[0.025, 1],[0.05, 0]]]) # right triangle
        # for flexor1
        param_output_1 = np.array([[[-0.05, 0],[-0.025, 1],[0,0]], # left triangle
                                  [[-0.03,0],[0,1],[0.03, 0]], # middle triangle
                                  [[0,0],[0.025, 1],[0.05, 0]]]) # right triangle
        
        weights_flexor0 = np.array([0,1,1])
        weights_flexor1 = np.array([1,1,1])
        membership_degree_flexor0 = mf.weighting(weights_flexor0, membership_degree)
        membership_degree_flexor1 = mf.weighting(weights_flexor1, membership_degree)

        
        # get arrays of output_membership
        fine = 1000 
        number_of_step = (du_max-du_min)*fine  
        dx =  (du_max-du_min)/number_of_step
            
        x = np.linspace(du_min, du_max, num=int(number_of_step))

        self.x = x
        """
        y0 = np.vectorize(mf.triangle_func)(x, param_output_0[0][0], param_output_0[0][1],param_output_0[0][2]) #left : du < 0 
        y1 = np.vectorize(mf.triangle_func)(x, param_output_0[1][0], param_output_0[1][1],param_output_0[1][2]) #middle: du ~ 0
        y2 = np.vectorize(mf.triangle_func)(x, param_output_0[2][0], param_output_0[2][1],param_output_0[2][2]) #right: du > 0
        """
        y0 = mf.triangle_func_np(x, param_output_0[0][0], param_output_0[0][1],param_output_0[0][2]) #left : du < 0 
        y1 = mf.triangle_func_np(x, param_output_0[1][0], param_output_0[1][1],param_output_0[1][2]) #middle: du ~ 0
        y2 = mf.triangle_func_np(x, param_output_0[2][0], param_output_0[2][1],param_output_0[2][2]) #right: du > 0

        y0 = np.minimum(membership_degree_flexor0[1],y0)
        y1 = np.minimum(membership_degree_flexor0[0],y1)
        y2 = np.minimum(membership_degree_flexor0[2],y2)
        self.y1 = np.vstack((y0,y1,y2))

        du[0] = mf.calc_centroid(x, y0, y1, y2, dx) # flexor0 output
        '''
        y0 = np.vectorize(mf.triangle_func)(x, param_output_1[0][0], param_output_1[0][1],param_output_1[0][2])
        y1 = np.vectorize(mf.triangle_func)(x, param_output_1[1][0], param_output_1[1][1],param_output_1[1][2])
        y2 = np.vectorize(mf.triangle_func)(x, param_output_1[2][0], param_output_1[2][1],param_output_1[2][2])
        '''
        y0 = mf.triangle_func_np(x, param_output_1[0][0], param_output_1[0][1],param_output_1[0][2])
        y1 = mf.triangle_func_np(x, param_output_1[1][0], param_output_1[1][1],param_output_1[1][2])
        y2 = mf.triangle_func_np(x, param_output_1[2][0], param_output_1[2][1],param_output_1[2][2])

        y0 = np.minimum(membership_degree_flexor1[1],y0)
        y1 = np.minimum(membership_degree_flexor1[0],y1)
        y2 = np.minimum(membership_degree_flexor1[2],y2)
        self.y2 = np.vstack((y0,y1,y2))

        # FUZZYCONTROL.visualize_functions('flexor1',x,y0,y1,y2)

        
        du[1] = mf.calc_centroid(x, y0, y1, y2, dx) # flexor1 output

        # print('du',du)
  
        return du
    
    def Fuzzy_process(self, current_angles, firstframe):
        current_angles = np.array(current_angles)
        if firstframe:
            self.control_method(self.err)
        else:
            err = self.target - current_angles
            self.control_method(err)

        self.apply_DR()
    
    @staticmethod
    def limit_dutyratio(dutyratio, upperlimit):
        return np.clip(dutyratio, None, upperlimit)
    
    @staticmethod
    def visualize_functions(title, x, y0,y1=0,y2=0):
        plt.plot(x,y0)
        plt.plot(x,y1)
        plt.plot(x,y2)
        plt.title(title)
        plt.show()

    def setting_visualize_functions_realtime(self): # To visualize output
        self.fig, self.axes = plt.subplots(1,2)
            

    def visualize_functions_realtime(self, interval, x, y_1, y_2): # To visualize output, used in while loooooop
        # y_i must consist of 3 data of y
        if self.mode == ['flex', 'flex']:
            for ax in self.axes: ax.clear()
            self.axes[0].set_title('flexor0')
            self.axes[1].set_title('flexor1')
            lines = [self.axes[0].plot(x, y_1[0])[0], self.axes[1].plot(x, y_2[0])[0]]
            lines = [self.axes[0].plot(x, y_1[1])[0], self.axes[1].plot(x, y_2[1])[0]]
            lines = [self.axes[0].plot(x, y_1[2])[0], self.axes[1].plot(x, y_2[2])[0]]
            self.axes[0].axvline(x=self.du[0], linestyle='--')
            self.axes[1].axvline(x=self.du[1], linestyle='--')


        plt.pause(interval)

    def angle_recorder(self, current_time, current_angles):
        current_angles = np.array(current_angles)
        temp = np.hstack(current_time, current_angles)
        self.angle_history = np.vstack(self.angle_history, temp)


    def angle_plotter(self):
        plt.plot(self.angle_history[1:, 0], self.angle_history[1:, 1], label='angle0')
        plt.plot(self.angle_history[1:, 0], self.angle_history[1:, 2], label='angle1')
        plt.plot(self.angle_history[1:, 0], self.angle_history[1:, 3], label='angle2')
        plt.xlabel = 'time'
        plt.ylabel = 'angle'
        plt.grid()
        plt.legend()
        plt.show()

    

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
    is_recod_video = False    
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
    control_interval = 1 # duty ratio is added once per control_interval
    visualize = True

    
    while True:
        cur_time = time.perf_counter()
        ret, frame_raw = cap.read()
        try:
            if ret:
                if is_recod_video: tracker.add_frame(frame_raw)
                else: tracker.input_frame(frame_raw)            

                if whether_firstframe:
                    tracker.acquire_marker_color()
                    firstangle = [tracker.first_angles[1], tracker.first_angles[2], tracker.first_angles[3]]
                    print('first angle:',firstangle)
                    target = fuzzy.input_target(firstangle)
                    if visualize: fuzzy.setting_visualize_functions_realtime()
                    timemeasure = time.perf_counter()
                    # whether_firstframe = False


                
                if True:

                    if frame_id == 0 or frame_id % control_interval == 0:
                        noneedframe, angle_0, angle_1, angle_2 = tracker.extract_angle()
                        angles = np.array([angle_0, angle_1, angle_2])
                        currrent_error =  angles - target
                        cv2.imshow('Video Preview', tracker.frame)
                
                    
                if control:
                    if frame_id == 0 or frame_id % control_interval == 0: 
                        fuzzy.Fuzzy_process(angles, whether_firstframe)
                        whether_firstframe = False
                        t = time.perf_counter() - timemeasure
                        timemeasure = time.perf_counter()
                        print('outout duty ratio', fuzzy.output_levels)
                        print(t)

                        if visualize:
                            fuzzy.visualize_functions_realtime(0.01, fuzzy.x, fuzzy.y1, fuzzy.y2)
                        
                frame_id += 1
                frame_times.append(cur_time)


                if True:
                    if frame_id > 45: cur_fps = 30 / (cur_time - frame_times[0])
                    else : cur_fps = -1
                    cv2.putText(frame_raw, f'Time: {time.strftime("%Y%m%d-%H%M%S")},{cur_time}',
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    
                    cv2.putText(frame_raw, f'Current Frame {frame_id}; FPS: {int(cur_fps)}',
                                (10,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            if cv2.waitKey(1) & 0xFF == ord('q'): 
                fuzzy.stop_DR()
                break
        except :
            fuzzy.stop_DR()

            break
 

    cap.release()
    cv2.destroyAllWindows()
    if is_recod_video: tracker.finalize()