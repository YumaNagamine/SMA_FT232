from SMA_finger.SMA_finger_MP import *

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
