import sys
import os,time
import cv2
import numpy as np
import json
from datetime import datetime
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QComboBox,
    QHBoxLayout, QVBoxLayout, QFileDialog, QSizePolicy, QMessageBox, QSlider,
    QGroupBox, QFormLayout, QToolBar, QAction
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap, QIcon

# Paths
dir_path = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(dir_path, './IMG')
os.makedirs(IMG_DIR, exist_ok=True)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from control.CameraSetting import Camera


CONFIG_PATH = os.path.join(dir_path, 'camera_config.json')
CAM_INDICES    = [0, 1]
CAM_POSITIONS  = ['side', 'top']

# Load or init config
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, 'r') as f:
        CONFIG = json.load(f)
    CONFIG.setdefault('exposure', -11)
    CONFIG.setdefault('gain', 0)
    CONFIG.setdefault('white_balance', 4000)
    with open(CONFIG_PATH, 'w') as f:
        json.dump(CONFIG, f, indent=4)
else:
    CONFIG = {
        'method': 'Flat-field',
        'flat_image': os.path.join(dir_path, 'flat_white.jpg'),
        'morph_kernel': 15,
        'exposure': -11,
        'gain': 0,
        'white_balance': 4000
    }

class CameraApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Camera Application  ')
        self.setWindowIcon(QIcon.fromTheme('camera'))

        # cam_name = 'side'
        cam_num = int(input('Use top Camera? (0/1): '))
        cam_name = CAM_POSITIONS[cam_num]
        # Camera connect with retries
        self.camera = None
        for _ in range(7):
            # try:
            cam = Camera(SOURCE=cam_num, CAP_API=cv2.CAP_MSMF, cam_name=cam_name)# CAP_MSMF
            if cam.isOpened():
                cam.realtime()
                self.camera = cam
                break
            cam.release()
            time.sleep(1)
            # usb = cam.get_usb()
            
            # except Exception as e:
            #     print(f'Error opening camera: {e}')
            #     print('Retrying to open camera...')

        if not self.camera or not self.camera.isOpened():
            print('Cannot open camera after 5 tries, exiting')
            # QMessageBox.critical(self, 'Error', 'Cannot open camera after 5 tries')
            sys.exit(1)
 
        # cam.load_calibration()
        self.cam = cam

        # # Set default properties
        # self.set_camera_props()

        # Create UI
        self.create_toolbar()
        self.create_preview()
        self.create_controls()

        # Main layout
        central = QWidget()
        vbox = QVBoxLayout(central)
        vbox.addWidget(self.preview_group)
        vbox.addWidget(self.control_group)
        self.setCentralWidget(central)

        # Timer for frame
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)
        self.recording = False


    def set_camera_props(self):
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1200)
        self.camera.set(cv2.CAP_PROP_FPS, 90)
        self.camera.set(cv2.CAP_PROP_EXPOSURE, float(CONFIG['exposure']))
        self.camera.set(cv2.CAP_PROP_GAIN, float(CONFIG['gain']))
        # white balance via OpenCV not standardized; omitted

    def create_toolbar(self):
        toolbar = QToolBar('File')
        self.addToolBar(toolbar)
        open_action = QAction(QIcon.fromTheme('folder-open'), 'Open Flat', self)
        open_action.triggered.connect(self.load_flat_image)
        toolbar.addAction(open_action)
        exit_action = QAction(QIcon.fromTheme('application-exit'), 'Exit', self)
        exit_action.triggered.connect(self.close)
        toolbar.addAction(exit_action)

    def create_preview(self):
        self.orig_label = QLabel('Original')
        self.proc_label = QLabel('Processed')
        for lbl in (self.orig_label, self.proc_label):
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setMinimumSize(320, 240)
            lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.preview_group = QGroupBox('Live Preview')
        h = QHBoxLayout(self.preview_group)
        h.addWidget(self.orig_label)
        h.addWidget(self.proc_label)

    def create_controls(self):
        self.control_group = QGroupBox('Controls')
        form = QFormLayout()
        # Correction method
        self.method_combo = QComboBox()
        self.method_combo.addItems(['Flat-field', 'Morphological Opening'])
        self.method_combo.setCurrentText(CONFIG.get('method','Flat-field'))
        self.method_combo.currentTextChanged.connect(self.save_method)
        form.addRow('Correction:', self.method_combo)
        # Exposure
        self.exp_slider = QSlider(Qt.Horizontal)
        self.exp_slider.setRange(-13,-1)
        self.exp_slider.setValue(CONFIG['exposure'])
        self.exp_slider.valueChanged.connect(self.save_exposure)
        form.addRow('Exposure:', self.exp_slider)
        # Gain
        self.gain_slider = QSlider(Qt.Horizontal)
        self.gain_slider.setRange(0,128)
        self.gain_slider.setValue(CONFIG['gain'])
        self.gain_slider.valueChanged.connect(self.save_gain)
        form.addRow('Gain:', self.gain_slider)
        # Snapshot and Video
        self.btn_photo = QPushButton(QIcon.fromTheme('camera-photo'), 'Photo')
        self.btn_photo.clicked.connect(self.take_photo)
        self.btn_video = QPushButton(QIcon.fromTheme('camera-video'), 'Record')
        self.btn_video.setCheckable(True)
        self.btn_video.clicked.connect(self.toggle_recording)
        h = QHBoxLayout()
        h.addWidget(self.btn_photo)
        h.addWidget(self.btn_video)
        form.addRow('Capture:', h)
        self.control_group.setLayout(form)

    def save_method(self, text):
        CONFIG['method'] = text
        self.save_config()

    def save_exposure(self, val):
        CONFIG['exposure'] = val
        self.camera.set(cv2.CAP_PROP_EXPOSURE, float(val))
        self.save_config()

    def save_gain(self, val):
        CONFIG['gain'] = val
        self.camera.set(cv2.CAP_PROP_GAIN, float(val))
        self.save_config()

    def load_flat_image(self):
        fname,_ = QFileDialog.getOpenFileName(self,'Flat Image',dir_path,'Images (*.jpg *.png)')
        if fname:
            CONFIG['flat_image']=fname; self.flat_img=cv2.imread(fname,0); self.save_config()

    def save_config(self):
        with open(CONFIG_PATH,'w') as f: json.dump(CONFIG,f,indent=4)

    def process_frame(self,frame):
        m=CONFIG['method']
        if m=='Flat-field' and os.path.exists(CONFIG['flat_image']):
            gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY).astype(np.float32)
            flat=cv2.resize(cv2.imread(CONFIG['flat_image'],0).astype(np.float32),(gray.shape[1],gray.shape[0]))
            norm=cv2.divide(gray,flat,scale=255.0)
            return cv2.cvtColor(norm.astype(np.uint8),cv2.COLOR_GRAY2BGR)
        elif m=='Morphological Opening':
            k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(CONFIG['morph_kernel'],)*2)
            return cv2.morphologyEx(frame,cv2.MORPH_OPEN,k)
        return frame

    def update_frame(self):
        ret,frame=self.camera.read()
        if not ret: return
        proc=self.process_frame(frame)
        self.orig_label.setPixmap(self.to_pixmap(frame,self.orig_label))
        self.proc_label.setPixmap(self.to_pixmap(proc,self.proc_label))
        if self.recording: self.video_writer.write(proc)

    def to_pixmap(self,frame,label):
        img=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        h,w,ch=img.shape
        q=QImage(img.data,w,h,ch*w,QImage.Format_RGB888)
        return QPixmap.fromImage(q).scaled(label.width(),label.height(),Qt.KeepAspectRatio)

    def take_photo(self):
        ret,frm=self.camera.read()
        if not ret: return
        p=self.process_frame(frm)
        ts=datetime.now().strftime('%Y%m%d_%H%M%S')
        o,f=os.path.join(IMG_DIR,f'orig_{ts}.jpg'),os.path.join(IMG_DIR,f'proc_{ts}.jpg')
        cv2.imwrite(o,frm);cv2.imwrite(f,p)
        QMessageBox.information(self,'Saved',f'O:{o}\nP:{f}')

    def toggle_recording(self):
        if not self.recording:
            ts=datetime.now().strftime('%Y%m%d_%H%M%S')
            path=os.path.join(IMG_DIR,f'video_{ts}.avi')
            fourcc=cv2.VideoWriter_fourcc(*'MJPG')
            fps=int(self.camera.get(cv2.CAP_PROP_FPS))
            w=int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH));h=int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.video_writer=cv2.VideoWriter(path,fourcc,fps,(w,h))
            self.recording=True;self.btn_video.setText('Stop')
        else:
            self.recording=False;self.video_writer.release();self.btn_video.setText('Record')
            QMessageBox.information(self,'Saved',f'Video saved.')

    def closeEvent(self,e):
        if self.recording: self.video_writer.release()
        self.camera.release(); super().closeEvent(e)


def main():
    app=QApplication(sys.argv)
    win=CameraApp();win.show();sys.exit(app.exec_())

if __name__=='__main__': main()