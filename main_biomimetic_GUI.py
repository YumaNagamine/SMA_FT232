 
# * Multi Process version
# Class of SMA single finger robot 
# Created by Askar. Liu @ 20231230
# Modified @20231231

# Based on https://cloud.tencent.com/developer/article/2192324

import sys,ctypes
import time

# import tkinter as tk
from tkinter.messagebox import askyesno
from tkinter.scrolledtext import ScrolledText
import threading,multiprocessing

import ttkbootstrap as ttk

from lib.GENERALFUNCTIONS import *
from SMA_finger.SMA_finger_MP import *



class exprimentGUI(object):

    def __init__(self, root_window,process_manager, width=[], height=[]):
        self.root_window = root_window
        
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        window_width = root.winfo_width()

        if width==[]:    
            self.width = int(screen_width * 0.998)
            self.height = int(screen_height * 0.92)

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
        
        # Non Multi-Processing
        self.page_video_scale_contoller(container_window=self.frame_page)

        self.channels = PWMGENERATOR.CH_EVEN

        self.fullScreenState = False
        self.root_window.bind("<F11>", self.toggleFullScreen)
        self.root_window.bind("<Escape>", self.exit)


    def toggleFullScreen(self, event):
        self.fullScreenState = not self.fullScreenState
        self.root_window.attributes("-fullscreen", self.fullScreenState)
        
    def page_video_scale_contoller(self,container_window):
        
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
        height_video_box = 0.6

        margin_video_log_box = margin_scale_butten_W
        log_video_frmae = ttk.Labelframe(Frame_video_log,text='Video',bootstyle="info",)
        log_video_frmae.place(relx=margin_video_log_box,rely=0,
                           relheight=height_video_box,relwidth=1-margin_video_log_box)

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

            _ch_label_text = ('Ch ' if _i==0 else '' )+str(_i)+''
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
         
        self.butten_connect()
        return []

    def thread_it(self, func, *args):
        """ 将函数打包进线程 """
        self.myThread = threading.Thread(target=func, args=args)
        self.myThread .setDaemon(True)  # 主线程退出就直接让子线程跟随退出,不论是否运行完成。
        self.myThread .start()

    def butten_connect(self): 
        print('\nConnecting the PCA9685')  
        
        RUNTIME = time.perf_counter()

        url_test_len = 4
        # url_0 = os.environ.get('FTDI_DEVICE', 'ftdi://ftdi:232h:0:FF/0')
        actuator_device = []
        for _i in range(url_test_len):
            _url = os.environ.get('FTDI_DEVICE', 'ftdi://ftdi:232h:0:F'+ hex(0xF-_i)[-1]+'/0')
            try: 
                print("\nConnecting: ",_url)
                actuator_device = ctrlProcess(_url,'ADC001')
            except  Exception as err: 
                print(err)
            if actuator_device:
                self.actuator_device = actuator_device
                break

    def butten_stop(self): 
        print('Stopping all channel output')
        for ch in self.output_levels: ch.set(0)
        self.butten_apply(disp=False)

    def butten_apply(self,disp=True):
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
            print('No connection, Auto connecting!')
            self.butten_connect()
    
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
            message="Close all outputs and Exit ?", parent=self.root_window,
            buttons=["No", "Yes:primary"],
            alert=True, localize=False)
        position = [int(root.winfo_screenwidth()/3),int(root.winfo_screenheight()/3) ] 

        dialog.show(position)

        if dialog.result == 'Yes':
            try : self.butten_stop()
            except Exception as err: print('Err during Exsisting: ',err)
            self.root_window.destroy(); sys.exit()
        else: return None

if __name__ == '__main__':
    sys.stdout = Logger()
    sys.stderr = sys.stdout		# redirect std err, if necessary

    # ctypes.windll.shcore.SetProcessDpiAwareness(1) #调用api设置成由应用程序缩放
    # ScaleFactor=ctypes.windll.shcore.GetScaleFactorForDevice(0) #调用api获得当前的缩放因子
    # root.tk.call('tk', 'scaling', ScaleFactor/100)    #设置缩放因子


    root = ttk.Window(hdpi=True,scaling=3,themename='darkly')  # darkly sandstone sandstone

    root.title("Contorl SMA")  # 设置窗口标题
    root.geometry('+0+0')
    # """ tk界面置顶 """
    # root.attributes("-topmost", 1)

    """ 创建Gui类对象 """
    test_gui = exprimentGUI(root,process_manager=multiprocessing.Manager())
    
    """ 初始化GUi组件 """
    root.mainloop()


     