# MAINLY FOR FINGER ACTUATION 
# * Multi Process version
# Class of SMA single finger robot 
# Created by Askar. Liu @ 20240729
# Modified @20240628

# Based on https://cloud.tencent.com/developer/article/2192324

import sys,os
print(sys.version)
os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin")

import time,cv2
from PIL import Image, ImageTk 
from collections import deque
import copy

import tkinter as tk
from tkinter.messagebox import askyesno
from tkinter.scrolledtext import ScrolledText
import threading,multiprocessing

# from tkinter import tk
import ttkbootstrap as ttk

from lib.GENERALFUNCTIONS import *
from SMA_finger.SMA_finger_MP import *

# from lib.GUI_Image_Editor.ViewGUI import ViewGUI

class exprimentGUI():

    def __init__(self, root,process_share_dict={}, width=[], height=[]):
        super().__init__()
        self.root_window = root
        self.actuator_device = []

        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        window_width = root.winfo_width()
        self.process_share_dict = process_share_dict

        if width==[]:    
            self.width = int(screen_width * 0.992)
            self.height = int(screen_height * 0.928)

        self.root_window.geometry( str(self.width)+'x'+str(self.height) )  # 设置窗口大小 

        """ 点击右上角关闭窗体弹窗事件 """
        self.root_window.protocol('WM_DELETE_WINDOW', self.exit)
        
        """ 组件容器创建 """
        self.nav_bar_width = 0.04
        self.frame_nav_bar = ttk.Frame(self.root_window,bootstyle="primary") 

        self.frame_nav_bar.place(relx=0,rely=0,relheight=1,relwidth=self.nav_bar_width)

        margin_page_H = 0.01
        self.frame_page = ttk.Frame(self.root_window,) 
        self.frame_page.place( relx = self.nav_bar_width ,rely = margin_page_H,
                              relheight = 1-2*margin_page_H, relwidth=1-self.nav_bar_width)
        # self.page_frame.place(x=self.nav_bar_weidth,rely=0,relheight=1,width=window_width-self.nav_bar_weidth)
        
        cd_video = 'C:\\Users\\51165\\OneDrive\\Pictures\\Camera Roll\\'
        # ViewGUI(self.root_window,cd_video) #
        self.page_video_scale_contoller(self.frame_page,cd_video)

        self.channels = PWMGENERATOR.CH_EVEN

        self.fullScreenState = False
        self.root_window.bind("<F11>", self.toggleFullScreen)
        self.root_window.bind("<Escape>", self.exit)


    def toggleFullScreen(self, event):
        self.fullScreenState = not self.fullScreenState
        self.root_window.attributes("-fullscreen", self.fullScreenState)
    
    def page_video_scale_contoller(self,container_window,cd_video):
        
        log_video_frame_width = 0.5
        Frame_video_log = ttk.Frame(container_window)   
        Frame_video_log.place(relx=0,rely=0,relheight=1,relwidth=log_video_frame_width)

        frame_scale_butten = ttk.Frame(container_window)   
        margin_scale_butten_W = 0.006
        frame_scale_butten.place(relx = log_video_frame_width+margin_scale_butten_W,
                                 rely=0,relheight=1,relwidth = 1-log_video_frame_width-margin_scale_butten_W )

        # Frame scale
        height_scale_box = 0.8
        frame_scale = ttk.Frame(frame_scale_butten)   
        frame_scale.place(relx=0,rely=0,relheight=height_scale_box,relwidth=1)
        

        # Video Box
        margin_video_log_box = margin_scale_butten_W

        width_video_box = 1 -margin_video_log_box
        height_video_box =   0.5625 * width_video_box

        log_video_frame = ttk.Labelframe(Frame_video_log,text='Video',bootstyle="info",)
        log_video_frame.place(relx=margin_video_log_box,rely=0,
                           relheight=height_video_box,relwidth= width_video_box )

        # image = self.process_share_dict['photo']
        image = cv2.imread(IMG_FOLDER+'1.jpg')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        photo = ImageTk.PhotoImage(image)  

        self.video_label_0 = tk.Label(log_video_frame)
        self.video_label_0.place(relx=0,rely=0,relheight= 0.95,relwidth=1)#.pack(expand = "yes")# 
        
        self.video_label_0.configure(image=photo)
        self.video_label_0.image = photo 
        
        # FPS box
        self.variable_fps = ttk.StringVar();self.variable_fps.set('Fps: ')
        self.entry_fps = ttk.Label(log_video_frame,textvariable= self.variable_fps,bootstyle='info')  
        self.entry_fps.place(relx=0.2,rely=0.95,relheight=0.05,relwidth=0.1)

        # Log Box
        log_label_frmae = ttk.Labelframe(Frame_video_log,text='Log',bootstyle="info",)
        log_label_frmae.place(relx=margin_video_log_box,rely= height_video_box,
                           relheight=1-height_video_box,relwidth=1-margin_video_log_box)
      
        from ttkbootstrap.scrolled import ScrolledText,ScrolledFrame
        self.scroller_log = ScrolledText(log_label_frmae, 
                    font=('Calibri Light',8),bootstyle='dark',vbar=True,autohide=True) # width=49, height=17,     
        self.scroller_log.place(relx=0,rely=0,relheight=1,relwidth=1)

 
        sys.stdout = self.ScollerLogger(self.scroller_log,Frame_video_log)
        sys.stderr = sys.stdout

        # Scales
        num_scale = 8
        scales = []
        margin_scale_W = margin_scale_butten_W
        margin_scale_H = 0.02
        self.output_levels = []

        # Variables for  callback
        for _ in range(num_scale): self.output_levels.append(ttk.DoubleVar()) 
            # self.output_levels[-1].set(0)
            # self.output_levels[_].trace_add("write",self.callback_scorller)

        
        # scales_colors = ["#%02x%02x%02x"%(25,255-int((_+1)*255/(num_scale+10)),255) for _ in range(num_scale)]  
        # self.run_log_print(str(scales_colors))
        height_ch_butten = 0.1
        height_ch_label_frame = 0.11
        for _i in range(num_scale):

            _ch_main_frame = ttk.Frame(frame_scale,)
            _ch_main_frame.place(relx = _i/(num_scale), rely = 0,
                         relheight=1, relwidth = 1/(num_scale)-margin_scale_W)

            _ch_label_text = ('Ch ' if _i==0 else '' )+str(_i*2)+''
            _ch_label_frmae = ttk.Labelframe(_ch_main_frame,text=_ch_label_text,bootstyle="info",)
            _ch_label_frmae.place(relx = 0, rely = 0, relheight= height_ch_label_frame, relwidth = 1-margin_scale_W)
            
            _ch_label = ttk.Entry(_ch_label_frmae, textvariable = self.output_levels[_i] )
            _ch_label.place(relx=0,rely=0,relheight=1,relwidth=1)   
            

            _ch_butten_max = ttk.Button(_ch_main_frame,text='MAX',bootstyle="success-outline",
                                        command = lambda arg=_i : self.output_levels[arg].set(1))
            
            _ch_butten_max.place(relx=0,rely=0.1+margin_scale_H,relheight=height_ch_butten,relwidth=1)


            _ch_butten_min = ttk.Button(_ch_main_frame,text='MIN',bootstyle="success-outline",
                                        command = lambda arg=_i : self.output_levels[arg].set(0))
            _ch_butten_min.place(relx=0,rely=1-height_ch_butten-margin_scale_H,
                                 relheight=height_ch_butten,relwidth=1)

            scales.append( ttk.Scale(_ch_main_frame, value=0, orient=ttk.VERTICAL,
                            takefocus=1,bootstyle="SUCCESS",name = 'ch'+str(_i), from_= 1,
                            to=0,variable = self.output_levels[_i]) )#self.callback_scorller
            
            scales[-1].place(relx = 0, rely = height_ch_label_frame +height_ch_butten +2*margin_scale_H,
                              relheight = 1 - height_ch_label_frame -2*height_ch_butten -4*margin_scale_H, 
                              relwidth = 1)
        
        # Frame System Buttens
        butten_list = ['','Connect\n\nPCA-9685','STOP \n\nAll channel','APPLY\n\nDuty Ratio','EXIT']
        butten_style_list = ['scondary','info','danger','warning','scondary']
        butten_style_list = [_ + 'outline' for _ in butten_style_list]
        butten_func_list = [[],self.butten_connect,self.butten_stop,self.butten_apply,self.exit]

        num_butten = len(butten_list)
        frame_button = ttk.Frame(frame_scale_butten) 
        frame_button.place( relx = 0, rely = height_scale_box+margin_scale_butten_W,
                    relheight = 1-height_scale_box-margin_scale_butten_W,relwidth=1)

        for _i in range(num_butten):
            butten_connect = ttk.Button(frame_button, width=20,
                    text= butten_list[_i],style=butten_style_list[_i], command = butten_func_list[_i])
            butten_connect.place(relx = _i/num_butten,rely=0,
                                 relheight=1,relwidth = 1/num_butten-margin_scale_butten_W)
        self.photo_acquired_t = 0
        self.butten_connect() 
        self.make_thread(self.refresh_img)

    def refresh_img(self):
        # t_old = self.timestamp_recived_img
        # t_new = process_share_dict['t_ready']
        # if not (t_new - t_old) == 0:
        #     self.root_window.after(500,self.refresh_img)
        #     return []
        # else: 
        #     self.timestamp_recived_img = t_new
        self.timer_fps = time.time()
        try:
            _photo_acquired_t = self.process_share_dict['photo_acquired_t']

            if self.photo_acquired_t == _photo_acquired_t: # Img not ready
                self.root_window.after(5,self.refresh_img)
                return []
            else :
                _dt = -self.photo_acquired_t + _photo_acquired_t
                self.photo_acquired_t = _photo_acquired_t
                image = self.process_share_dict['photo']
            
            photo = ImageTk.PhotoImage(image)  
            self.video_label_0.configure(image=photo,)
            self.video_label_0.image = photo    
            self.root_window.after(1,self.refresh_img)
            
            # dt =  time.time() - self.timer_fps
            # self.timer_fps = time.time()
            self.variable_fps.set('FPS: '+str(int(1/_dt)))
            return []

        except Exception as err:
            # print('\rVideo frame load from thread manager failed: ', end='')
            # print('\tDue to:',err, end='')
            # print('Tring again ... ...', end='')
            self.root_window.after(1000,self.refresh_img)

    def make_thread(self, func, *args):
        self.newThread = threading.Thread(target=func, args=args)
        self.newThread.daemon = True
        self.newThread.start()

    def butten_connect(self): 
        print('\nConnecting the PCA9685')  
        
        RUNTIME = time.perf_counter()

        if self.actuator_device:
            self.actuator_device.i2c_controller.close()

        url_test_len = 4
        # url_0 = os.environ.get('FTDI_DEVICE', 'ftdi://ftdi:232h:0:FF/0')
        actuator_device = []
        for _i in range(url_test_len):
            # _url = os.environ.get('FTDI_DEVICE', 'ftdi://ftdi:232h:0:F'+ hex(0xF-_i)[-1]+'/1')
            _url = os.environ.get('FTDI_DEVICE', 'ftdi:///1')

            try: 
                print("\nConnecting: ",_url)
                actuator_device = ctrlProcess(_url,'ADC001')
            except  Exception as err: 
                print(err)
            if actuator_device:
                self.actuator_device = actuator_device
                break

    def butten_stop(self,retry=True): 
        print('Stopping all channel output')
        for ch in self.output_levels: ch.set(0)
        self.butten_apply(disp=False,retry=False)

    def butten_apply(self,disp=True,retry = True):
        for ch_DR in self.output_levels:
            val = ch_DR.get()
            # print(val)
            if val > 1: ch_DR.set(val / (10** (len(str(val))-2) ))
            elif val>0: ch_DR.set(val)
            else: ch_DR.set(0)
        try:
            for channels,ch_DR, in zip( self.channels,self.output_levels ):
                self.actuator_device.setDutyRatioCH(channels,ch_DR.get(),relax=False)
            if disp: 
                print('Applying channel output (%): ')
                print(' '.join(str(int(100*_.get()))+'  ' for _ in self.output_levels))
        except AttributeError:
            print('No connection!')
            if retry: print('Auto connecting!'); self.butten_connect()
    
    class ScollerLogger(object):
        def __init__(self, scroller=[],master=[]):
            self.terminal = sys.stdout
            self.log = scroller
            self.master = master
 
        def write(self, message):
            self.terminal.write(message)
            self.log.insert(ttk.END, message)
            self.log.update()
            self.log.text.yview_moveto(1)

        def flush(self):
            pass
            
    # def scorller_print(self, message):
    #     self.scroller_log.insert(ttk.END, message+'\n')
    #     return message

    def callback_scorller(self, var,index,mode): #message=[],
         
        for ch in self.output_levels:
            val = ch.get()
            if val > 100: ch.set(val*0.001)
            elif val > 1: ch.set(val*0.01)
            elif val>0: ch.set(val)
            else: ch.set(0)
        
        print( ' '.join(str(_.get())+'  ' for _ in self.output_levels))
        pass   

    def exit(self,*args): 
        from ttkbootstrap.dialogs import MessageDialog

        dialog = MessageDialog(title='EXIT',
            message="Close all outputs and Exit ?", parent=self.root_window,font=('Calibri',16),
            # width= int(self.root_window.winfo_screenwidth()/3),
            padding= [200,200],
            buttons=["No", "Yes:primary"],
            alert=True, localize=False)
        position = [int(self.root_window.winfo_screenwidth()/3),int(self.root_window.winfo_screenheight()/3) ] 

        dialog.show(position)
        # dialog

        if dialog.result == 'Yes':
            try : self.butten_stop(retry=False)
            except Exception as err: print('Err during Exsisting: ',err)
            self.root_window.destroy(); sys.exit()
        else: return None

def process_GUI(pid,process_share_dict={}):
    root = ttk.Window(hdpi=True,scaling=3,themename='darkly')  # darkly sandstone sandstone
    # process_share_dict['root'] = root

    root.title("Contorl SMA")  # 设置窗口标题
    root.geometry('+0+0')
    exprimentGUI(root,process_share_dict)
    root.mainloop()

    pass

# Static
def process_camera(pid,process_share_dict={}):
    from camera.ASYNCSAVER_with_ANGLESREADER import AsyncVideoSaver as VideoSaver
    from cv_angle_traking.angles_reader_multicolor import AngleTracker
    ## Create CAM obj
    cam_num = 0
    
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

    if is_recod_video: saver = VideoSaver(video_file_name, fourcc, target_fps, resolution)
    frame_id = 0
    # time_cv_st = time.perf_counter()
    
    # 初始化时间戳队列^^^^---^
    frame_times = deque(maxlen=30)  # 保持最近30帧的时间戳

    tracker = AngleTracker(video_name=video_file_name, denoising_mode='monocolor')
    cv_choose_wd_name = 'Video Preview'
    while True: # Video Loop // 90 Hz
        cur_time = time.perf_counter()
        ret, frame_raw = cap.read()
        # print(ret)

        if ret:
            if frame_id == 0: tracker.acquire_marker_color(frame_raw,cv_choose_wd_name,tracker)
            
            if is_recod_video: saver.add_frame(frame_raw)
            frame_id += 1
            frame_times.append(cur_time)

            if True: #frame_id % int(actual_fps // 20) == 0:  # 每S两次
                if frame_id > 45: cur_fps = 30 / (cur_time - frame_times[0])
                else : cur_fps = -1
                cv2.putText(frame_raw, f'Time: {time.strftime("%Y%m%d-%H%M%S")},{cur_time}',
                             (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                
                cv2.putText(frame_raw, f'Current Frame {frame_id}; FPS: {int(cur_fps)}',
                             (10,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                
            if frame_id % int(actual_fps // 20) == 0:  # 每S两次
                # deepcopy. frame_raw(800,600) 
                # frame_raw_copied = frame_raw.deepcopy()
                
                copied_frame = copy.deepcopy(frame_raw)
                # cv2.resize(frame_raw.copy(), (800,600))

                copied_frame = cv2.resize(copied_frame,(720,480))
                
                # image_to_send = Image.fromarray(cv2.cvtColor(frame_raw.copy(), cv2.COLOR_BGR2RGB))
                image_to_send = Image.fromarray(cv2.cvtColor(copied_frame, cv2.COLOR_BGR2RGB))
                
                process_share_dict['photo'] = image_to_send
                process_share_dict['photo_acquired_t'] = time.time()

                if True: #read angles
                    tracker.acquire_marker_color(frame_raw)
                    frame, angle_0, angle_1, angle_2  = tracker.extract_angle(frame_raw, False)
                    
                    print(angle_0)
                    print(angle_1)
                    print(angle_2)


                    process_share_dict['angles'] = [angle_0, angle_1, angle_2]

                # process_share_dict['angles']
            pass        
        else: continue

        if cv2.waitKey(1) & 0xFF == ord('q'):  # 按'q'键退出
            break
 
    cap.release()
    cv2.destroyAllWindows()
    if is_recod_video: saver.finalize()

    

if __name__ == '__main__':
    # sys.stdout = Logger()
    # sys.stderr = sys.stdout		# redirect std err, if necessary

    # ctypes.windll.shcore.SetProcessDpiAwareness(1) #调用api设置成由应用程序缩放
    # ScaleFactor=ctypes.windll.shcore.GetScaleFactorForDevice(0) #调用api获得当前的缩放因子
    # root.tk.call('tk', 'scaling', ScaleFactor/100)    #设置缩放因子
 
    # """ tk界面置顶 """
    # root.attributes("-topmost", 1)
    print('Running on env: ',sys.version_info)

    with multiprocessing.Manager() as process_manager:

        process_share_dict = process_manager.dict() # inital

        process_share_dict['photo'] = []
        process_share_dict['angles']=[]

        process_root = multiprocessing.Process(
            target= process_GUI,name='GUI', args=(1,process_share_dict) )
        
        process_cam = multiprocessing.Process( 
            target= process_camera, name='CAM', args=(2,process_share_dict))
        
        
        process_cam.start()
        time.sleep(3)
        process_root.start()
        process_root.join()


     