# u : duty ratio
# this program assumes u0 and u1 are for flexion, u2, u3 and u4 are fot extension

def random_coefficient_matrix(num_input, num_output, minimal_value, maximum_value): #random coefficients
    K  = np.random.randint(minimal_value, maximum_value, size = (num_output, num_input))
    return K

def calcurate_DR(P, I, D, error, total_error, incrementel_error):
    max_DR = 0.2
    DR = (P @ error) + (I @ total_error) + (D @ incrementel_error)
    DR_bool = DR < 0
    DR[DR_bool] = 0
    Max = np.max(DR)
    if Max != 0:
        normalize = max_DR/Max
        DR = np.array(DR*normalize)
    else: pass
    return DR

if __name__ == "__main__" : 
    import numpy as np

    #testing calculate code
    P = random_coefficient_matrix(3,5,1,10)
    I = random_coefficient_matrix(3,5,1,10)
    D = random_coefficient_matrix(3,5,1,10)
    error = np.array([-30,-45,-20])
    total_error = np.array([-100,-245,-200])
    incremental_error = np.array([-1,-2,-3])
    u = calcurate_DR(P, I, D, error, total_error, incremental_error)
    print(u)

    
    from camera.NOGUI_ASYNCSAVER_with_ANGLESREADER import AsyncVideoSaver, AngleTracker
    import cv2
    import time
    from collections import deque
    target = []
    for i in range(3):
        print("angle_", i , ':', sep = '', end = '')
        angle = int(input())
        target.append(angle)

    target = np.array(target)

    cam_num = 0

    is_lightning = True
    is_recod_video = True
    cam_name = 'AR0234'

    cap = cv2.VideoCapture(cam_num, cv2.CAP_DSHOW)
    if cam_name == 'AR0234':
        target_fps = 90
        resolution = (1600,1200)
        width, height = resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

        cap.set(cv2.CAP_PROP_FPS, target_fps)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)

        if is_lightning:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            cap.set(cv2.CAP_DROP_GAIN, 0)
            cap.set(cv2.CAP_PROP_EXPOSURE, -11)
        else: 
            cap.set(cv2.CAP_PROP_GAIN, 0)
            cap.set(cv2.CAP_PROP_EXPOSURE, -3)

        fourcc = 'X264'
    
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print("Target FPS: {target_fps}, Actual FPS: {actual_fps}")
    if fourcc == 'X264':
        video_file_name = 'IMG/video/' +cam_name +'_' + time.strftime("%m%d-%H%M%S")  + '.mp4'


    frame_id = 0
    whether_first_frame = True

    frame_times = deque(maxlen = 30)

    cv_preview_wd_name = 'Video Preview'

    cv2.nameWindow(cv_preview_wd_name, cv2.WINDOW_GUI_EXPANDED)
    cv2.nameWindow("Mask", cv2.WINDOW_GUI_EXPANDED)

    if is_recod_video : saver = AngleTracker(video_file_name, fourcc, target_fps, resolution, 'monocolor' )

    #PID parameters
    num_input = 3
    num_output = 4
    Kp = random_coefficient_matrix(num_input, num_output, 1,10)
    Ki = random_coefficient_matrix(num_input, num_output, 1,10)
    Kd = random_coefficient_matrix(num_input, num_output, 1,10)
    prev_error = np.zeros(3)
    ie = 0
    prev_time = time.perf_counter()
    while True:
        cur_time = time.perf_counter()
        ret, frame_raw = cap.read()

        if ret:
            if is_recod_video: saver.add_frame(frame_raw)

            if whether_first_frame:
                saver.acquire_marker_color()
                whether_first_frame = False

            frame_id += 1
            frame_times.append(cur_time)

            if True: #read angles
                noneedframe, angle0, angle1, angle2 = saver.extract_angle(False)
                if angle0 > 180: angle0 = 360 - angle0

                if angle1 > 180: angle1 = 360 - angle1
                angles = [angle0, angle1, angle2]
                print("angle: ", angles)
                cv2.imshow(cv_preview_wd_name, saver.frame)
            
            if True: #determine duty ratio
                t = time.perf_counter()
                dt = t - prev_time
                cur_angles = np.array(angles)
                error = target - cur_angles
                de = (error - prev_error)/dt
                ie += (error + prev_error) * dt/2

                u = calcurate_DR(P, I, D, error, ie, de)

                if error[0] < 0 and error[1]<0 : #if 
                    u[3], u[4] = 0, 0
                if error[0] > 0 and error[1] > 0:
                    u[0], u[1] = 0, 0
                
                u = np.concatenate((u, [u[3]]))  #u4 and u5 should have the same value
                print(u)
                prev_error = error
                prev_time = t
                pass
            if True:
                if frame_id > 45:
                    cur_fps  = 30 / (cur_time - frame_times[0])
                else:
                    cur_fps = -1
                cv2.putText(frame_raw, f'Time: {time.strftime("%Y%m%d-%H$M%S")},{cur_time}',
                             (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA) 
                cv2.putText(frame_raw, f'Current Frame {frame_id}; FPS: {int(cur_fps)}',
                             (10,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)


        else:
            print("cannot read video")
            break

        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()
    if is_recod_video: saver.finalize()
    print(Kp, Ki, Kd)


