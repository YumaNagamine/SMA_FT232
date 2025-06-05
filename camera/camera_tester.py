import cv2
import numpy as np

def show_message_CV(window_name="C0 | C1",
                              text="",
                              duration=0.2,
                              width=1920,
                              height=1200):
    # Create a black image
    msg_img = np.zeros((height, width, 3), dtype=np.uint8)

    # Text settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.6
    thickness = 1
    line_spacing = 2  # multiplier for line height

    # Split into lines and draw each
    lines = text.split("\n")
    (base_x, base_y) = (10, 100)  # starting position
    for i, line in enumerate(lines):
        # Compute y position for this line
        y = int(base_y + i * (cv2.getTextSize(line, font, font_scale, thickness)[0][1] * line_spacing))
        cv2.putText(msg_img, line, (base_x, y), font, font_scale, (255, 255, 255), thickness)

    # Show in window
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, width, height)
    cv2.imshow(window_name, msg_img)
    cv2.waitKey(int(duration * 1000))
 
def main():

    backend = cv2.CAP_MSMF # 1400
    
    # backend = cv2.CAP_DSHOW # 700

    # backend = cv2.CAP_ARAVIS # 700

    # print(backend);exit()

    # 2. Create a NORMAL (resizable) window
    cv2.namedWindow("C0 | C1", cv2.WINDOW_NORMAL)
    # 3. Resize window to exactly w×h (or w*zoom×h*zoom for integer zoom)
    cv2.resizeWindow("C0 | C1", 1920,1200)
    cv2.moveWindow("C0 | C1", 0, 0)

    show_message_CV("C0 | C1",text="Twin camera viwer ", duration=0.5)


    cap0 = cv2.VideoCapture(0, backend)
    cap1 = cv2.VideoCapture(1, backend)

    cap_list = [cap1,cap0]
    resolution = [1920,1200]#MSMF [640,480]  [1280,960] ,  - 800×600
    show_message_CV("C0 | C1",text="Setting up Camera parameters ... ... ",duration=0.5)

    for _ in cap_list:
        # show_initializing_message("C0 | C1",text="Seting up: "+str(_),duration=0.5)

        _.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        _.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        _.set(cv2.CAP_PROP_FPS, 90)
        _.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        _.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)   # Manual mode
        _.set(cv2.CAP_PROP_EXPOSURE,  -7)
        _.set(cv2.CAP_PROP_AUTO_WB,       0)   # Off
        _.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUY2'))

        lines = [
            f"Seting up: {str(_)}",
            f"CAP_PROP_FRAME_WIDTH: {_.get(cv2.CAP_PROP_FRAME_WIDTH):.0f}",
            f"CAP_PROP_FRAME_HEIGHT: {_.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f}",
            f"CAP_PROP_FPS: {_.get(cv2.CAP_PROP_FPS):.2f}"
        ]
        text = "\n".join(lines)

        # show the properties for half a second in the "C0 | C1" window
        show_message_CV("C0 | C1", text=text, duration=0.5)

    # for cap in cap_list:
    #     if not cap.isOpened():
    #         print("Cannot open one of the cameras")
    #         return

    while True:
        # Grab both frames first
        # cap0.grab()
        # cap1.grab()
        try:
            latest_frames = []
            for cap in cap_list:            
                _ret , _frame = cap.read()            
                if _ret: latest_frames.append(_frame)
                else : print("Failed to grab frame from one of the cameras");continue # latest_frames.append([])

            # ret0, frame0 = cap0.read()
            # ret1, frame1 = cap1.read()

            # if not ret0 or not ret1:
            #     print("Failed to grab frame from one of the cameras")
            #     continue

            # Resize to same height and stitch
            h = min(latest_frames[0].shape[0],latest_frames[0].shape[0])
            f0 = cv2.resize(latest_frames[0], (int(latest_frames[0].shape[1]*h/latest_frames[0].shape[0]), h))
            f1 = cv2.resize(latest_frames[1], (int(latest_frames[1].shape[1]*h/latest_frames[1].shape[0]), h))
            combined = np.hstack((f0, f1))
            cv2.imshow("C0 | C1", combined)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except Exception as _:
            print("Exception found:",_)
            continue

    cap0.release()
    cap1.release()
    cv2.destroyAllWindows()



def list_supported_resolutions(device_index=0, backend=cv2.CAP_MSMF, candidate_resolutions=None, warmup_frames=2):
    """
    Probe a list of resolutions by recreating the VideoCapture object for each test
    and return those the camera actually supports. 
    CAP_MSMF
    CAP_DSHOW
    """
    if candidate_resolutions is None:
        candidate_resolutions = [
            # (160, 120), (320, 240), 
            (640, 480),(800, 600), 
            (1024, 768), (1280, 720),
            (1280, 1024), (1600, 1200), 
            (1920, 1080), (1920, 1200),
            # (2560, 1440), (3840, 2160),
        ]

    supported = []
    for w_req, h_req in candidate_resolutions:
        # 1) (Re)create the capture for each resolution test
        cap = cv2.VideoCapture(device_index, backend)
        if not cap.isOpened():
            cap.release()
            continue

        # 2) Request the mode
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w_req)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h_req)

        # 3) Warm up negotiation
        for _ in range(warmup_frames):
            cap.read()

        # 4) Try a real read
        ret, frame = cap.read()
        if ret and frame is not None and frame.size != 0:
            # 5) Check what we actually got
            w_act = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h_act = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if (w_act, h_act) == (w_req, h_req):
                supported.append((w_req, h_req))
                print("Found supporting res: ",(w_req, h_req))

        # Release before next iteration
        cap.release()

    return supported

def main_show_res():
# if __name__ == "__main__":
    resolutions = list_supported_resolutions(device_index=0)
    print("Supported resolutions:")
    for w, h in resolutions:
        print(f"  • {w}×{h}")
 
if __name__ == "__main__":
    # main_show_res()
    main()
