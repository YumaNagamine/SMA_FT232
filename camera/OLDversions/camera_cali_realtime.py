# -*- coding: utf-8 -*-
# camera_calibration.py
# Simplified: intrinsic, index selection only, flat-field estimation UI
# CODE BY: Renke Askar LIU, assisted by ChatGPT
# DATE: 20250518
# VERSION: 2.2

import os, sys, time
from datetime import datetime

# Add project root for importing control.CameraSetting
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import glob
import cv2
import numpy as np
from control.CameraSetting import Camera

# --- CONFIGURATION ---
CHESS_GLOB      = './IMG/chessboard*.jpg'
BOARD_SIZE      = (10, 5)      # inner corners (cols, rows)
SQUARE_SIZE_MM  = 25.0
OUTPUT_PATH     = './CAL/cam/' # side/
# FILENAME_TPL    = 'intrinsics_{mode}_{timestamp}.npz'

# Morphology parameters
MAX_MORPH_K     = 401
MAX_BLUR_K      = 51

# Button layout
BTN_W, BTN_H    = 200, 80
BTN_PAD         = 20

# Fallback camera indices
CAM_INDICES     = [0, 1]
CAM_NAME     = ['side', 'top']

# --- FUNCTIONS ---

def calibrate_intrinsics():
    objp = np.zeros((BOARD_SIZE[1]*BOARD_SIZE[0], 3), np.float32)
    objp[:, :2] = (np.mgrid[0:BOARD_SIZE[0], 0:BOARD_SIZE[1]].T.reshape(-1, 2)
                   * SQUARE_SIZE_MM)
    obj_points, img_points = [], []
    for path in glob.glob(CHESS_GLOB):
        img = cv2.imread(path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ok, corners = cv2.findChessboardCorners(gray, BOARD_SIZE, None)
        if ok:
            obj_points.append(objp)
            img_points.append(corners)
    if not obj_points:
        raise RuntimeError('No chessboard corners found for calibration.')
    h, w = gray.shape
    ret, mtx, dist, *_ = cv2.calibrateCamera(obj_points, img_points, (w, h), None, None)
    if not ret:
        raise RuntimeError('Intrinsic calibration failed.')
    print('Intrinsics computed.')
    return mtx, dist


def estimate_illumination(gray, k, s):
    bg = cv2.morphologyEx(gray, cv2.MORPH_OPEN,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    bg = bg.astype(np.float32) + 1e-3
    bg /= np.mean(bg)
    if s > 1:
        bg = cv2.GaussianBlur(bg, (s, s), 0)
    return bg


def draw_buttons(labels, title, window_size=None):
    n = len(labels)
    width = BTN_PAD*(n+1) + BTN_W*n
    height = BTN_PAD*2 + BTN_H
    canvas = np.full((height, width, 3), 50, np.uint8)
    coords = [(lbl, BTN_PAD + i*(BTN_W+BTN_PAD), BTN_PAD) for i, lbl in enumerate(labels)]
    selection = [-1]
    def click_cb(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            for idx, (_, x0, y0) in enumerate(coords):
                if x0 < x < x0+BTN_W and y0 < y < y0+BTN_H:
                    selection[0] = idx
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    if window_size:
        cv2.resizeWindow(title, *window_size)
    cv2.setMouseCallback(title, click_cb)
    while selection[0] < 0:
        canvas[:] = 50
        for lbl, x0, y0 in coords:
            cv2.rectangle(canvas, (x0, y0), (x0+BTN_W, y0+BTN_H), (200,200,200), -1)
            cv2.putText(canvas, lbl, (x0+10, y0+BTN_H//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
        cv2.imshow(title, canvas)
        if cv2.waitKey(100) == ord('q'):
            break
    cv2.destroyWindow(title)
    return selection[0]


def choose_camera():
    labels = [f'Index {i}' for i in CAM_INDICES] + ['Cancel']
    while True:
        sel = draw_buttons(labels, 'Select Camera Index', window_size=(1280,720))
        if 0 <= sel < len(CAM_INDICES):
            idx = CAM_INDICES[sel]
            print(f"Testing camera {idx}...")
            # warm = cv2.VideoCapture(idx)
            time.sleep(0.5)
            # warm.release()
            try:
                cap = Camera(idx, cv2.CAP_DSHOW)
                print(f"Opened index: {idx}")
                return cap,CAM_NAME[sel]
            except Exception as e:
                print(f"Failed to open index {idx}: {e}")
                time.sleep(1)
        else:
            print('Invalid selection; retrying...')
            time.sleep(1)


def run_illumination_ui(cap, mtx, dist, bg_color,cam_name):
    title = 'Illumination Preview'
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, 2560, 1440)
    cv2.moveWindow(title, 0, 0)
    cv2.createTrackbar('MorphK', title, 101, MAX_MORPH_K, lambda x:None)
    cv2.createTrackbar('BlurK',  title,   1, MAX_BLUR_K,  lambda x:None)
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        und = cv2.undistort(frame, mtx, dist, None, mtx)
        gray = cv2.cvtColor(und, cv2.COLOR_BGR2GRAY)
        k = cv2.getTrackbarPos('MorphK', title) | 1
        s = cv2.getTrackbarPos('BlurK',  title) | 1
        illum = estimate_illumination(gray, k, s)
        map8 = cv2.normalize(illum, None, 0,255,cv2.NORM_MINMAX).astype(np.uint8)
        disp = cv2.applyColorMap(map8, cv2.COLORMAP_JET)
        h, w = illum.shape
        canvas = np.full((h, w, 3), bg_color, np.uint8)
        img_r = cv2.resize(und, (w, h))
        mask = (img_r > 0)
        canvas[mask] = img_r[mask]
        combo = np.hstack((canvas, disp))
        cv2.imshow(title, combo)
        key = cv2.waitKey(1)
        if key == ord('s'):
            # Save calibration results and metadata to JSON
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            data = {
                'cali_cam_name': cam_name,
                'cali_timestamp': ts,
                'cali_camera_matrix': mtx.tolist(),
                'cali_dist_coeffs': dist.reshape(-1).tolist(),
                'cali_morphology': {
                    'kernel_size': k,
                    'blur_size': s
                },
                # 'background_color': bg_color,
                # 'chessboard': {
                #     'size': BOARD_SIZE,
                #     'square_size_mm': SQUARE_SIZE_MM
                # },
                # 'image_size': (w, h),
                # 'image_path': CHESS_GLOB
            }
            # File paths
            json_path = os.path.join(OUTPUT_PATH + '' +cam_name+'/', f"intrinsics_side_{ts}.json")
            import json
            with open(json_path, 'w') as jf:
                json.dump(data, jf, indent=2)
            print(f"Saved JSON: {json_path}")
            break
        if key == ord('q'):
            break
    cv2.destroyWindow(title)


def main():
    mtx, dist = calibrate_intrinsics()
    cap = None
    try:
        cap,cam_name = choose_camera()
        bg_idx = draw_buttons(['Black','White'], 'Select Background', window_size=(1280,720))
        bg_color = (255,255,255) if bg_idx==1 else (0,0,0)
        run_illumination_ui(cap, mtx, dist, bg_color,cam_name)
    finally:
        if cap:
            cap.release()
            time.sleep(1)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
