# applying predetermined dutyratios and save results to csv

from SMA_finger.SMA_finger_MP import *
from control.CameraSetting import Camera
from cv_angle_traking.modify_markers_angles_reader import ModifiedMarkers

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

if __name__ == "__main__":
    controller = Interface()
    for i in range(len(controller.output_levels)):
        controller.output_levels[i] = 0.1
    controller.apply_DR(retry=False)
    time.sleep(5.0)
    controller.stop_DR()





if __name__ == "__main__":
    import cv2
    from cv_angle_traking.Multicamera_realtime_tracking import AngleTracking 

    # is_angle_based = True

    # setting controller
    controller = Interface()
    output = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
    controller.output_levels = np.array(output)

    tracker = AngleTracking()
    videosave_dir = "./"
    sidecamera = Camera(0, cv2.CAP_MSM, 'side')
    sidecamera.realtime()
    topcamera = Camera(1, cv2.CAP_MSMF, 'top')
    topcamera.realtime()
    output_level = [0.1,0.1,0.1,0.1,0.1,0.1]
    try:
        while True:
            ret1, frame_side = sidecamera.read()
            ret2, frame_top = topcamera.read()
            if not (ret1 and ret2):
                print('missed frame!')
                continue
            controller.apply_DR(retry=False)

    finally:
        controller.stop_DR()
        
        sidecamera.release()
        topcamera.release()

