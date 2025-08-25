# applying predetermined dutyratios and save results to csv

from SMA_finger.SMA_finger_MP import *
from control.CameraSetting import Camera
from cv_angle_traking.Multicamera_realtime_tracking import AngleTracking
import numpy as np
import cv2

OUTPUT_NUMBER = 6

class Interface:
    def __init__(self):
        self.actuator_device = []
        self.channels = PWMGENERATOR.CH_EVEN
        self.flag_for_forcequit = False
        self.output_levels = np.zeros(OUTPUT_NUMBER) # current output
        self.DR_history = np.zeros(OUTPUT_NUMBER, dtype=np.float32) 
        
        self.connect()
        print('\ndevice connected...\n')
        
    def connect(self):
        url_test_len = 4
        actuator_device = []

        for _ in range(url_test_len):
            _url = os.environ.get('FIDI_DEVICE', 'ftdi:///1')
            try:
                print('connecting:')
                actuator_device = ctrlProcess(_url, 'ADC001')
                
            except Exception as err:
                print(err)
            if actuator_device:
                self.actuator_device = actuator_device
                break

    def apply_DR(self, retry):
        try:
            for channels, ch_DR in zip(self.channels, self.output_levels):
                self.actuator_device.setDutyRatioCH(channels, ch_DR, relax=False)
        except AttributeError:
            print('NO CONNECTION!')
            if retry: print('Auto connecting!'); self.connect()
        
    def stop_DR(self):
        print('Set all Duty Ratio Zero!!')
        for i in range(len(self.output_levels)): self.output_levels[i] = 0
        self.apply_DR(retry=False)

def main_sequential_processing():
    tracker = AngleTracking()
    resolution= (800,600)#(1920,1200)

    # if resolution = (1920,1200)
    # tracker.set_params(theta_side=0.55, theta_top = 0, distance=-30)
    # if resolution = (800,600)
    tracker.set_params(theta_side=0.55, theta_top = 0, distance=-15)
    videosave_dir = "./sc01/multi_angle_tracking"
    sidecamera = Camera(0, cv2.CAP_MSMF, 'side')
    target_fps = 90
    sidecamera.realtime(resolution = resolution, target_fps= target_fps)
    topcamera = Camera(1, cv2.CAP_MSMF, 'top')
    topcamera.realtime(resolution = resolution, target_fps = target_fps)

    frame_id = 0
 
    is_first_frame = True
    time.sleep(1) # wait for camera to be ready
    controller = Interface()
    start = time.time()

    try:
        while True:
            framestart = time.time()
            ret1, frame_side = sidecamera.read()
            ret2, frame_top = topcamera.read()
            if not (ret1 and ret2):
                print('missed frame!')
                break
            output_level = [0.1, 0.0, 0.0, 0.0, 0.0, 0.0]
            for i in range(len(controller.output_levels)):
                controller.output_levels[i] = output_level[i]
            # output_level = np.array(output_level)
            tracker.processing_frame(frame_id, frame_side, frame_top, is_first_frame, True, output_level)
            controller.apply_DR(retry=True)
    
            is_first_frame = False
            frame_id += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            end = time.time()
            t = end - framestart
            if t < 0.10:
                delay = 0.10 - t
                time.sleep(delay)
    finally:
        controller.stop_DR()
        effective_fps = frame_id / int(end - start)
        print(f'processing time; {int(end-start)}, fps: {effective_fps}')
        tracker.data_saver_finalize(videosave_dir, fps=effective_fps, is_dutyratio=True)
        tracker.video_saver_finalize(videosave_dir, fps=effective_fps, resolution=resolution)
        sidecamera.release()
        topcamera.release()
        cv2.destroyAllWindows()


def main_parallel_processing():
    pass

if __name__ == "__main__":

    '''Sequential processing version'''
    # main_sequential_processing()

    ''' parallel processing version ''' 
    main_parallel_processing()
