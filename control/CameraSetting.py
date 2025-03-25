# This file is for shortening camera setting process created by Nagamine on March 5th
import cv2
import time

class Camera():
    def __init__(self, cam_num):
        self.cam_num = cam_num

    def main(self): #ここにカメラセッティングをすべて入れる
        is_lighting = True
        is_record_video = True
        cam_name = 'AR0234'

        self.cap = cv2.VideoCapture(self.cam_num, cv2.CAP_DSHOW)  #cv2.CAP_DSHOW  CAP_WINRT

        if cam_name == 'AR0234': # Aptina AR0234
            target_fps = 90
            resolution =  (1600,1200)#(1920,1200)#q(800,600)# (800,600)#(1920,1200) (1280,720)#
            width, height = resolution

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0]) 
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
            # Set FPS
            self.cap.set(cv2.CAP_PROP_FPS,target_fps)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG')) # 'I420'
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)  # 设置缓冲区大小为2
            
            if is_lighting:            # 曝光控制
                # 设置曝光模式为手动
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # 0.25表示手动模式，0.75表示自动模式
                self.cap.set(cv2.CAP_PROP_GAIN, 0)  # 调整增益值，具体范围取决于摄像头
                self.cap.set(cv2.CAP_PROP_EXPOSURE, -11)  # 设置曝光值，负值通常表示较短的曝光时间
            else:            
                self.cap.set(cv2.CAP_PROP_GAIN, 0)  # 调整增益值，具体范围取决于摄像头
                self.cap.set(cv2.CAP_PROP_EXPOSURE, -3)  # 设置曝光值，负值通常表示较短的曝光时间
            # Save video
            fourcc = 'X264'#'MJPG' # 'I420' X264

        if fourcc == 'X264':  
            self.video_file_name = 'IMG/video/' +cam_name +'_' + time.strftime("%m%d-%H%M%S")  + '.mp4'

        self.frame_id = 0 
    
    def read(self):
        return self.cap.read()
    
    def release(self):
        self.cap.release()
    


