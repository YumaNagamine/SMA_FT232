from control.CameraSetting import Camera
import numpy as np
import cv2, time, os
from datetime import datetime
from interface import Interface
import pandas as pd

def set_DR_outputlevel(currenttime:float, T:float, dT:float, h0 = 1.0, h1 = 0.02):
    if currenttime < T:
        return 0
    elif currenttime < T + dT:
        return h0
    else:
        return h1
    
def saver(fps, resolution, frames1, frames2, input_history, save_dir="./sc01/multi_angle_tracking"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")    
    video_filename1 = os.path.join(save_dir, f"{timestamp}_raw_side.mp4")
    video_filename2 = os.path.join(save_dir, f"{timestamp}_raw_top.mp4")
    csv_filename = os.path.join(save_dir, f"{timestamp}.csv")
    columns = ['duty_ratio0', 'duty_ratio1', 'duty_ratio2','duty_ratio3', 'duty_ratio4', 'duty_ratio5', 'frame_id' , 'time']
    df = pd.DataFrame(data = input_history, columns=columns) 
    df['fps_time'] = df['frame_id'] / fps
    df.to_csv(csv_filename, index=False)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Win
    out = cv2.VideoWriter(video_filename1, fourcc, fps, resolution)
    for frame in frames1:
        out.write(frame)
    out.release()
    out = cv2.VideoWriter(video_filename2, fourcc, fps, resolution)
    for frame in frames2:
        out.write(frame)
    out.release()
    print(f"Data saved to {save_dir}")
    
if __name__ == "__main__":
    controller = Interface()
    resolution = (800, 600)  # (1920,1200)
    target_fps = 50
    frame_id = 0
    cv2.namedWindow('side', cv2.WINDOW_NORMAL)
    sidecamera = Camera(0, cv2.CAP_MSMF, 'side')
    topcamera = Camera(1, cv2.CAP_MSMF, 'top')
    sidecamera.realtime(resolution=resolution, target_fps=target_fps)
    topcamera.realtime(resolution=resolution, target_fps=target_fps)
    time.sleep(1)  # wait for camera to be ready
    side_frame_stocker = []
    top_frame_stocker = []
    input_history = []
    T_list = [0, 0, 0, 0, 0, 0]
    dT_list = [1.0, 0, 0, 0, 0, 0]
    h0_list = [1.0, 0, 0, 0, 0, 0]
    h1_list = [0.04, 0, 0, 0, 0, 0]
    is_first_frame = True
    try:
        while True:
            ret1, frame_side = sidecamera.read()
            ret2, frame_top = topcamera.read()
            if not (ret1 and ret2):
                print('missed frame!')
                break
            side_frame_stocker.append(frame_side)
            top_frame_stocker.append(frame_top)
            if is_first_frame:
                start = time.time()
                is_first_frame = False
            current_time = time.time() - start
            for i in range(len(controller.output_levels)):
                controller.output_levels[i] = set_DR_outputlevel(
                    current_time, T_list[i], dT_list[i], h0_list[i], h1_list[i]
                )
            # print(current_time)
            output = controller.output_levels.copy()
            output = output.tolist()
            output.append(frame_id)
            output.append(current_time)
            input_history.append(output)   
            # print(input_history[-1])
            controller.apply_DR(retry=True)
            frame_id += 1
            cv2.imshow('side', frame_side)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
    except:
        import traceback
        traceback.print_exc()
    finally:
        controller.stop_DR()
        actual_fps = frame_id / current_time
        save_dir = "./sc01/multi_angle_tracking"
        # saver(target_fps, resolution, side_frame_stocker, top_frame_stocker, input_history, save_dir=save_dir)
        saver(actual_fps, resolution, side_frame_stocker, top_frame_stocker, input_history, save_dir=save_dir)
        sidecamera.release()
        topcamera.release()
        cv2.destroyAllWindows()