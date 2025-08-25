# This file is for shortening camera setting process created by Nagamine on March 5th
import cv2
from cv2 import VideoCapture
import time,json
import numpy as np
from collections import deque
import concurrent.futures

_prop_map = {getattr(cv2, name): name
                        for name in dir(cv2)
                        if name.startswith('CAP_PROP_')}
def prop_name(prop_id):
    return _prop_map.get(prop_id, f'UNKNOWN({prop_id})')


CAM_INDICES    = [0, 1]
CAM_POSITIONS  = ['side', 'top']
# --- Slider configuration for camera parameters --------------------------------
PARAMS = [
    (cv2.CAP_PROP_AUTO_EXPOSURE, 100, lambda x: 0.25 + (x/100)*0.5, lambda v: int((v-0.25)/0.5*100), 0.25),
    (cv2.CAP_PROP_EXPOSURE, 12,lambda x: -13 + x, lambda v: int(v + 13), -10),
    (cv2.CAP_PROP_GAIN, 255,lambda x: x, lambda v: int(v), 0),
    (cv2.CAP_PROP_AUTO_WB, 1, lambda x: x, lambda v: int(v), 0),
    (cv2.CAP_PROP_BRIGHTNESS, 512, lambda x: x-256, lambda v: int(v+256), 0),
    (cv2.CAP_PROP_CONTRAST, 512,lambda x: x-256, lambda v: int(v+256), 0),
    (cv2.CAP_PROP_SATURATION, 255,lambda x: x, lambda v: int(v), 70),
    (cv2.CAP_PROP_HUE, 179,lambda x: x, lambda v: int(v),70),
    (cv2.CAP_PROP_SHARPNESS, 255, lambda x: x, lambda v: int(v), 0),
]



class Camera(VideoCapture):
 
    def __init__(self,SOURCE, CAP_API=None,cam_name="side",video_path = None):
        super().__init__(SOURCE, CAP_API)
        self.cam_name = cam_name
        self.video_path = video_path
        self.frame_id = 0
        self.maxlen = 30
        self.frame_queue = deque()

        self.is_opened = self.isOpened()
        if not self.is_opened:
            print('Cannot open the camera')
            raise IOError("Cannot open the camera")
        else:
            print('Camera opened successfully')
        
    def load_calibration(self, config_path: str=None):

        "Flat while feild calibration for side or top camera."
        print("Camera loading for json file, cam No.",self.cam_name)
        if 'side' in self.cam_name:
            if config_path is None:
                config_path = './CAL/cam/side/intrinsics_flat_side_20250527_204730.json'
            
            """
            Load and normalize flat-field reference at IMG/<mode>cali/flat_white.jpg.
            """
            flat_path = f'./IMG/{self.cam_name}cali/flat_white.jpg'
            # path = os.path.join(BASE_IMG_DIR, f'{mode}cali', 'flat_white.jpg')
            flat_bgr = cv2.imread(flat_path, cv2.IMREAD_COLOR)
            if flat_bgr is None:            raise FileNotFoundError(f'Flat-field file not found: {flat_path}')
            
            # to float32 and add small epsilon
            fmap = flat_bgr.astype(np.float32) + 1e-3
            # heavy blur to capture smooth illumination field
            fmap = cv2.GaussianBlur(fmap, (101, 101), 0)
            # normalize each channel so its mean is 1.0
            for c in range(3):
                ch = fmap[:, :, c]
                fmap[:, :, c] = ch / np.mean(ch)
            self.flat_ref = fmap  # shape H×W×3, float32
            print(f"Loaded RGB flat-field for '{self.cam_name}' (shape {fmap.shape})")

        elif 'top' in self.cam_name:
            if config_path is None:
                config_path = './CAL/cam/top/intrinsics_flat_top_20250527_204845.json'
        


        """ Distortion calibration for side or top camera.
        Load one camera's calibration from JSON and
        set self.cam_name, self.camera_matrix, self.dist_coeffs,
        self.resolution, self.map1/map2, and self.morphology.
        """
        # Build default path if not provided
        if config_path is None:
            config_path = f'./CAL/cam/{self.cam_name}/intrinsics_flat_{self.cam_name}_{self.timestamp}.json'
        # Load JSON file
        with open(config_path, 'r') as f:
            cfg = json.load(f)
        if not cfg:
            raise ValueError(f"JSON file {config_path} is empty or invalid.")        
       

        # Assign camera name and matrices
        self.cam_name       = cfg.get('mode',           self.cam_name)
        self.camera_matrix  = np.array(cfg['camera_matrix'], dtype=np.float64)
        self.dist_coeffs    = np.array(cfg['dist_coeffs'],   dtype=np.float64)
        # Determine output resolution from flat_map_shape [rows, cols]
        rows, cols = cfg.get('flat_map_shape', [0, 0])
        if rows and cols:
            self.resolution = (cols, rows)
        else:
            # fallback to current capture size
            w = int(self.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.resolution = (w, h)
 
        # Precompute q
        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            self.camera_matrix, self.dist_coeffs, None,
            self.camera_matrix, self.resolution, cv2.CV_16SC2
        )
        print(f"Loaded calibration for '{self.cam_name}':")
        print(f"  Timestamp: {cfg.get('timestamp')}")
        print(f"  Camera Matrix: {self.camera_matrix}")
        print(f"  Dist Coeffs: {self.dist_coeffs}")
        print(f"  Resolution: {self.resolution}")
 
    def process_frame(self, frame,
                      doflat=True,
                      doundistort=True,
                      dowhitebalance=False,
                      do_white_balance=True,blur_ksize: int = 25):
        """
        Returns:
          - an HSV image (uint8, H×W×3) with all channels:
              • flat-field corrected
              • undistorted
              • white-balance calibrated
        """

        img = frame.astype(np.float32)

        # 1) per-channel flat-field (and white-balance)
        if doflat and hasattr(self, 'flat_ref'):
            img /= self.flat_ref
        img = np.clip(img, 0, 255).astype(np.uint8)

        # 3) Undistort color image
        if doundistort:
            img = cv2.remap(img, self.map1, self.map2, cv2.INTER_LINEAR)

        # Gray-world WB
        if do_white_balance:
            means = cv2.mean(img)[:3]; overall = sum(means)/3.0
            for c in range(3): 
                img[:,:,c] = np.clip(img[:,:,c].astype(np.float32)*(overall/(means[c] or 1)),0,255)
            img = img.astype(np.uint8)
        
        # Gaussian blur
        if blur_ksize and blur_ksize > 1:
            k = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
            img = cv2.GaussianBlur(img, (k, k), 0)
        
        # 4) White balance (gray-world): scale each channel so its mean equals overall mean
        if dowhitebalance: # May have BUG
            # compute per-channel means
            means = cv2.mean(img)[:3]
            overall = sum(means) / 3.0
            scale = [overall / m if m > 0 else 1.0 for m in means]
            # apply scaling and clip
            for c in range(3):
                img[:, :, c] = np.clip(img[:, :, c].astype(np.float32) * scale[c], 0, 255)
        
        img = img.astype(np.uint8)
        # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        return img

    
    
    def _configure_and_report(self,settings):
        """
        cap      : cv2.VideoCapture instance
        settings : dict of {prop_id: desired_value}
        """
        # First apply all settings
        for prop, val in settings.items():
            ori = self.get(prop)
            ok = self.set(prop, val)
            # print(f"SET  {prop:30} → {val!r}    {'[OK]' if ok else '[FAILED]'}")

            got = self.get(prop)
            print(f"GET  {prop_name(prop):30}: {ori} → {val};RES: {got!r}  \t{'[OK]' if ok else '[FAILED]'} ")
        print("-" * 50, "\n")


    def realtime(self, resolution: tuple, target_fps:int): #Refined in 250519
        is_lighting = True
        is_record_video = True
        cam_name = 'AR0234'
        if cam_name == 'AR0234': # Aptina AR0234
            # target_fps = 60
            # resolution = (1920,1200)#(1920,1200)#q(800,600)# (800,600)#(1920,1200) (1280,720)#
            # width, height = resolution
            # # Usage example:
            # cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

            # Define your manual parameters here:
            if self.cam_name == 'side':
                manual_settings = {
                    cv2.CAP_PROP_AUTO_EXPOSURE:       .25,#0.25,   # DirectShow: 0.25=manual, 0.75=auto
                    # cv2.CAP_PROP_EXPOSURE:            -10 if is_lighting else -3,    # your desired exposure
                    cv2.CAP_PROP_EXPOSURE:            -8 if is_lighting else -3,    # your desired exposure
                    cv2.CAP_PROP_GAIN:                0,      # your desired gain

                    cv2.CAP_PROP_FRAME_WIDTH:          resolution[0],
                    cv2.CAP_PROP_FRAME_HEIGHT:         resolution[1],
                    cv2.CAP_PROP_AUTO_WB:             0,      # disable auto white balance
                    # cv2.CAP_PROP_WHITE_BALANCE_BLUE_U: 4600,  # tweak as needed
                    # cv2.CAP_PROP_WHITE_BALANCE_RED_V:  -1,

                    # cv2.CAP_PROP_AUTOFOCUS:           0,      # turn off autofocusCannot open the cameraCannot open the camera
                    # cv2.CAP_PROP_FOCUS:               10,     # manual focus value

                    # cv2.CAP_PROP_ISO_SPEED:           0,    # Linux/V4L2 only
                    cv2.CAP_PROP_BRIGHTNESS:          0,    # mid‐point
                    cv2.CAP_PROP_CONTRAST:            0,
                    cv2.CAP_PROP_SATURATION:          70,
                    cv2.CAP_PROP_HUE:                 70,
                    cv2.CAP_PROP_SHARPNESS:           0,


                    cv2.CAP_PROP_FPS:                  target_fps,
                    cv2.CAP_PROP_FOURCC:               cv2.VideoWriter_fourcc(*'YUY2'), # 'YUY2' MJPG
                    cv2.CAP_PROP_BUFFERSIZE:           0,  # 设置缓冲区大小为2
                }
            elif self.cam_name == 'top':
                manual_settings = {
                    cv2.CAP_PROP_FPS:                  target_fps,
                    cv2.CAP_PROP_FOURCC:               cv2.VideoWriter_fourcc(*'YUY2'), # 'YUY2' MJPG
                    cv2.CAP_PROP_BUFFERSIZE:           0,  # 设置缓冲区大小为2
                    cv2.CAP_PROP_FRAME_WIDTH:          resolution[0],
                    cv2.CAP_PROP_FRAME_HEIGHT:         resolution[1],

                    cv2.CAP_PROP_AUTO_EXPOSURE:       .25,#0.25,   # DirectShow: 0.25=manual, 0.75=auto
                    # cv2.CAP_PROP_EXPOSURE:            -9 if is_lighting else -3,    # your desired exposure
                    cv2.CAP_PROP_EXPOSURE:            -7 if is_lighting else -3,    # your desired exposure
                    cv2.CAP_PROP_GAIN:                0,      # your desired gain

                    cv2.CAP_PROP_AUTO_WB:             0,      # disable auto white balance
                    # cv2.CAP_PROP_WHITE_BALANCE_BLUE_U: 4600,  # tweak as needed
                    # cv2.CAP_PROP_WHITE_BALANCE_RED_V:  -1,

                    # cv2.CAP_PROP_AUTOFOCUS:           0,      # turn off autofocusCannot open the cameraCannot open the camera
                    # cv2.CAP_PROP_FOCUS:               10,     # manual focus value

                    # cv2.CAP_PROP_ISO_SPEED:           0,    # Linux/V4L2 only
                    cv2.CAP_PROP_BRIGHTNESS:          250,    # mid‐point
                    cv2.CAP_PROP_CONTRAST:            0,
                    cv2.CAP_PROP_SATURATION:          120,
                    cv2.CAP_PROP_HUE:                 0,
                    cv2.CAP_PROP_SHARPNESS:           0, 
                }
            # Apply and report
            self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)
            self._configure_and_report(manual_settings)
 
            # Save video
            fourcc = 'X264'#'MJPG' # 'I420' X264

        if fourcc == 'X264':  
            self.video_file_name = 'IMG/video/' +cam_name +'_' + time.strftime("%m%d-%H%M%S")  + '.mp4'

        self.frame_id = 0
        self.out = cv2.VideoWriter(self.video_file_name, cv2.VideoWriter_fourcc(*fourcc), target_fps, resolution)

    def existingvideo(self, video_path,  following_name = 'processed', frame_shift = 0, output_fps=90):
        is_lighting = True
        
        self.set(cv2.CAP_PROP_POS_FRAMES, frame_shift)
        self.set(cv2.CAP_PROP_FPS, output_fps)
        self.output_path = video_path + '_' + following_name
        resolution = (1920,1200)
        if self.cam_name == 'side':   
            manual_settings = {
                cv2.CAP_PROP_AUTO_EXPOSURE:       .25,#0.25,   # DirectShow: 0.25=manual, 0.75=auto
                cv2.CAP_PROP_EXPOSURE:            -10 if is_lighting else -3,    # your desired exposure
                cv2.CAP_PROP_GAIN:                0,      # your desired gain

                cv2.CAP_PROP_AUTO_WB:             0,      # disable auto white balance
                cv2.CAP_PROP_BRIGHTNESS:          239,    # mid‐point
                cv2.CAP_PROP_CONTRAST:            137,
                cv2.CAP_PROP_SATURATION:          70,
                cv2.CAP_PROP_HUE:                 0,
                cv2.CAP_PROP_SHARPNESS:           0,

                cv2.CAP_PROP_FRAME_HEIGHT:         resolution[1],
                cv2.CAP_PROP_FRAME_WIDTH:          resolution[0],

                cv2.CAP_PROP_FPS:                  output_fps,
                cv2.CAP_PROP_FOURCC:               cv2.VideoWriter_fourcc(*'YUY2'), # 'YUY2' MJPG
                cv2.CAP_PROP_BUFFERSIZE:           0,  # 设置缓冲区大小为2
            }
        elif self.cam_name == 'top':
            manual_settings = {
                cv2.CAP_PROP_AUTO_EXPOSURE:       .25,#0.25,   # DirectShow: 0.25=manual, 0.75=auto
                cv2.CAP_PROP_EXPOSURE:            -10 if is_lighting else -3,    # your desired exposure
                cv2.CAP_PROP_GAIN:                0,      # your desired gain

                cv2.CAP_PROP_AUTO_WB:             0,      # disable auto white balance
                cv2.CAP_PROP_BRIGHTNESS:          239,    # mid‐point
                cv2.CAP_PROP_CONTRAST:            137,
                cv2.CAP_PROP_SATURATION:          70,
                cv2.CAP_PROP_HUE:                 0,
                cv2.CAP_PROP_SHARPNESS:           0,

                cv2.CAP_PROP_FRAME_HEIGHT:         resolution[1],
                cv2.CAP_PROP_FRAME_WIDTH:          resolution[0],

                cv2.CAP_PROP_FPS:                  output_fps,
                cv2.CAP_PROP_FOURCC:               cv2.VideoWriter_fourcc(*'YUY2'), # 'YUY2' MJPG
                cv2.CAP_PROP_BUFFERSIZE:           0,  # 设置缓冲区大小为2
            }
        for prop, val in manual_settings.items():
            self.set(prop, val)
            
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

    def finalize(self):
        if self.frame_queue:
            self.save_frame_batch(list(self.frame_queue))
        self.executor.shutdown(wait=True)
        self.out.release()

    def realtime_OLD(self): #ここにカメラセッティングをすべて入れる
        is_lighting = True
        is_record_video = True
        cam_name = 'AR0234'

        if cam_name == 'AR0234': # Aptina AR0234
            target_fps = 90
            resolution = (1920,1200)#(1920,1200)#q(800,600)# (800,600)#(1920,1200) (1280,720)#
            width, height = resolution
            self.set(cv2.CAP_PROP_AUTO_WB,    0)  # turn off auto–white–balance NEW!
            self.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0]) 
            self.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
            # Set FPS
            self.set(cv2.CAP_PROP_FPS,target_fps)
            self.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG')) # 'I420'
            self.set(cv2.CAP_PROP_BUFFERSIZE, 0)  # 设置缓冲区大小为2
            
            if is_lighting:            # 曝光控制
                # 设置曝光模式为手动
                self.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # 0.25表示手动模式，0.75表示自动模式
                self.set(cv2.CAP_PROP_GAIN, 0)  # 调整增益值，具体范围取决于摄像头
                self.set(cv2.CAP_PROP_EXPOSURE, -11)  # 设置曝光值，负值通常表示较短的曝光时间
            else:            
                self.set(cv2.CAP_PROP_GAIN, 0)  # 调整增益值，具体范围取决于摄像头
                self.set(cv2.CAP_PROP_EXPOSURE, -3)  # 设置曝光值，负值通常表示较短的曝光时间
            
            # Save video
            fourcc = 'X264'#'MJPG' # 'I420' X264

        if fourcc == 'X264':  
            self.video_file_name = 'IMG/video/' +cam_name +'_' + time.strftime("%m%d-%H%M%S")  + '.mp4'

        self.frame_id = 0


def on_trackbar(pos, cap, prop_id, to_cam):
    cap.set(prop_id, to_cam(pos))

def create_sliders(window_name, cap):
    for prop_id, maxval, to_cam, from_cam, default in PARAMS:
        name = prop_name(prop_id)
        init = from_cam(default)
        cv2.createTrackbar(name, window_name, init, maxval,
            lambda pos, pid=prop_id, tcam=to_cam: on_trackbar(pos, cap, pid, tcam)
        )
        cap.set(prop_id, default)

# --------------------------------------------------------------------------------


if __name__ == '__main__': #   Quick Test the Camera class
    # Add project root for importing control.CameraSetting
    import os, sys
    # import cv2
    import numpy as np
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, PROJECT_ROOT)

    cam_num = int(input('Use top Camera? (0/1): '))
    cam_name = CAM_POSITIONS[cam_num]


    retry = 5
    while retry > 0:
        cam = Camera(cam_num, cv2.CAP_MSMF, cam_name=cam_name)
        if cam.isOpened():
            break
        cam.release()
        time.sleep(1)
        retry -= 1
    # Initialize camera
    # cam = Camera(0, cv2.CAP_DSHOW, cam_name='side')


    # cam.load_calibration()
    # cam.setup_capture(resolution=(1920, 1200), fps=90, lighting=True)
    cam.realtime()
 

    # Setup display
    win='Original | Processed'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.moveWindow(win,0,0); cv2.resizeWindow(win,1920,1080)
    ctrl='Controls'; cv2.namedWindow(ctrl,cv2.WINDOW_AUTOSIZE)

    # Create camera sliders and process toggle
    create_sliders(ctrl, cam)
    cv2.createTrackbar('Process ON/OFF',ctrl,0,1,lambda x:None) # (name, window_name, value, maxval, onChange)

    while True:
        ret,frame=cam.read()
        if not ret: continue
        sw=cv2.getTrackbarPos('Process ON/OFF',ctrl)
        if sw:
            processed=cam.process_frame(frame)
            
            right=cv2.cvtColor(processed,cv2.COLOR_HSV2BGR)
        else:
            right=frame.copy()
        cv2.imshow(win,np.hstack((frame,right)))
        if cv2.waitKey(1)&0xFF==ord('q'): break

    cam.release(); time.sleep(1); cv2.destroyAllWindows()