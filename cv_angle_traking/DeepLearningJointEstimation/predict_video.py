import yaml
import os, sys, time
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
angle_tracker_path = os.path.abspath(
    os.path.join(project_root, 'cv_angle_traking')
)
if angle_tracker_path not in sys.path:
    sys.path.insert(0, angle_tracker_path)



import torch
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor
from models.network import SimpleJointNet
import cv2
from video_processing.mask_processor import VideoMaskProcessor
from cv_angle_traking.modify_markers_angles_reader import ModifiedMarkers
from predict import predict_frame

def plot_joint(frame, coords, colors):
    for (x,y), color in zip(coords, colors):
        cv2.circle(frame, (int(x), int(y)), radius = 10, color=color, thickness=2)

def calc_angles_from_jointpos(jointpos: np.ndarray) -> np.float32:
    pass 

if __name__ == "__main__":

    with open("./cv_angle_traking/DeepLearningJointEstimation/configs/config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    device = torch.device(cfg['train']['device'])

    checkpoint = 'checkpoints/epoch_49.pt'

    colors = [(43,74,134), (0,0,255), (255,0,0), (0,255,255)]
    font_scale = 1
    text_position_cnt = (100, 100)
    text_position_time = (100, 120)

    video_name = "FDP.mp4"
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

    mask_processor = VideoMaskProcessor(threshold=115, area_threshold_black=5000, area_threshold_white=1000)

    # モデルロード
    model = SimpleJointNet(
        hidden_dim=cfg['model']['hidden_dim'],
        output_dim=cfg['model']['output_dim']
    ).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    
    try:
        while True:
            strt = time.time()
            ret, frame = cap.read()
            if not ret: break
            # frame = tracker.frame_trimer(frame, 1300, 1200)
            
            frame = np.array(frame)
            mask = mask_processor.make_mask_per_frame(frame)
            # mask = mask*255
            mask_RGB = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
            mask_RGB = np.array(mask_RGB)
            preds = predict_frame(mask_RGB, checkpoint, model, device)
            preds = np.array(preds).reshape((4,2))
            plot_joint(frame, preds, colors)


            # Add text to the frame
            frame = tracker.add_text_to_frame(frame, str(cnt), position=text_position_cnt, font_scale=font_scale)
            end = time.time()
            # Calculate and add time information
            frame = tracker.add_text_to_frame(frame, str(end - strt), position=text_position_time, font_scale=font_scale)

            
            frames_to_store.append(frame)
            cnt += 1
            if cnt % 20 == 0: print(cnt)


            if frame_jump == 0:
                pass
            elif not cnt % frame_jump ==0 :
                cnt += 1;continue
            else: 
                pass
                # print(cnt)
            # if cnt > 1000:break
            # print(cnt)
            cv2.imshow('Video Preview', frame)
            if cv2.waitKey(1) & 0xFF == 27: # cv2.waitKey(1000) & 0xFF == ord('q')
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()

        print("\nFinished video extraction")

        ## Store processed video
        # Store the video with updated frames
        # Set the desired output video path
        
        tracker.store_video_rename(frames_to_store,output_video_fps,'DL')
        print(tracker.marker_position_frame0)
        tracker.store_raw_data(measure, output_video_fps)
        print(tracker.video_pos_file_url)

