import yaml
import os, sys
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor
from models.network import SimpleJointNet
import cv2
from video_processing.mask_processor import VideoMaskProcessor
from cv_angle_traking.modify_markers_angles_reader import ModifiedMarkers
from predict import predict_frame


def calc_angles_from_jointpos(jointpos: np.ndarray) -> np.float32:
    pass 

if __name__ == "__main__":

    with open("./cv_angle_traking/DeepLearningJointEstimation/configs/config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    device = torch.device(cfg['train']['device'])

    checkpoint = 

    colors = [(43,74,134), (0,0,255), (255,0,0), (0,255,255)]
    font_scale = 1
    text_position_cnt = (100, 100)
    text_position_time = (100, 120)

    video_name = ".mp4"
    frame_jump = 0

    ## For algorithm tuning
    # Are for optime
    kernel = np.ones((5,5),np.uint8)
    threshold_area_size = [200, 50, 50, 10]# [80, 20, 10, 40]
    frame_shift = 0
    output_video_fps = 90 

    tracker = ModifiedMarkers(video_name,denoising_mode = 'monocolor')
    cap = cv2.VideoCapture(tracker.video_path) 
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_shift)
    cap.set(cv2.CAP_PROP_FPS, output_video_fps)


    measure = [] # for storing angles
    frames_to_store = []
    cnt = frame_shift # for storing frame count


    # モデルロード
    model = SimpleJointNet(
        hidden_dim=cfg['model']['hidden_dim'],
        output_dim=cfg['model']['output_dim']
    ).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
