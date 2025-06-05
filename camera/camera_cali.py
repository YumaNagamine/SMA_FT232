# -*- coding: utf-8 -*-
# camera_calibration.py
# Intrinsic calibration and flat-field reference generation
# CODE BY: Renke Askar LIU, assisted by ChatGPT
# DATE: 20250520
# VERSION: 5.3

import os
import sys
import time
import json
import glob
from datetime import datetime

import cv2
import numpy as np

# Add project root for importing control.CameraSetting
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from control.CameraSetting import Camera

# --- CONFIGURATION ---
BOARD_SIZE     = (10, 7)    # inner corners per row, column (10,5) if side
SQUARE_SIZE    = 25.0       # millimeters
BASE_IMG_DIR   = '.\\IMG'
BASE_OUTPUT    = '.\\CAL\\cam'
JSON_TPL       = 'intrinsics_flat_{mode}_{timestamp}.json'
CAM_INDICES    = [0, 1]
CAM_POSITIONS  = ['side', 'top']

DEFAULT_FRAMES = 30

# --- UTILITY FUNCTIONS ---

def draw_menu(options, prompt):
    """
    Display a console menu and return selected index.
    """
    print(prompt)
    for idx, opt in enumerate(options):
        print(f'  [{idx}] {opt}')
    while True:
        choice = input('Enter choice: ')
        if choice.isdigit():
            i = int(choice)
            if 0 <= i < len(options):
                return i
        print('Invalid selection, try again.')

# --- CALIBRATION FUNCTIONS ---

def calibrate_intrinsics(mode):
    """
    Calibrate intrinsics using chessboard images in IMG/<mode>cali/.
    Returns camera matrix and distortion coefficients.
    """
    glob_pattern = os.path.join(BASE_IMG_DIR, f'{mode}cali', 'chessboard*.jpg')
    objp = np.zeros((BOARD_SIZE[1]*BOARD_SIZE[0],3),np.float32)
    objp[:,:2] = np.mgrid[0:BOARD_SIZE[0],0:BOARD_SIZE[1]].T.reshape(-1,2) * SQUARE_SIZE
    obj_points, img_points = [], []
    last_gray = None
    
    for path in glob.glob(glob_pattern):
        img = cv2.imread(path)
        if img is None: continue
        else:print(f'\nImage FOUND as: {path}\nProcessing')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print(f'Graysclaed: {path}, shape={gray.shape}')
        
        # found, corners = cv2.findChessboardCorners(gray, BOARD_SIZE, None)
        found, corners = cv2.findChessboardCornersSB(gray, BOARD_SIZE, flags=cv2.CALIB_CB_NORMALIZE_IMAGE)

        if found:
            obj_points.append(objp)
            img_points.append(corners)
        else: print(f'Chessboard not found in {path}!!!!')
             
        last_gray = gray
    print(f'Found {len(obj_points)} chessboard images for mode="{mode}"')
    print(f'Last image shape: {last_gray.shape if last_gray is not None else "None"}')
    
    if not obj_points or last_gray is None:
        
        print(f'Check if images are in {BASE_IMG_DIR}/{mode}cali/',glob.glob(glob_pattern))
        raise RuntimeError(f'No valid chessboard images found for mode="{mode}"')
    h, w = last_gray.shape
    ret, mtx, dist, _, _ = cv2.calibrateCamera(obj_points, img_points, (w,h), None, None)
    if not ret:
        raise RuntimeError('Intrinsic calibration failed')
    print(f'Intrinsic calibration complete for mode="{mode}"')
    return mtx, dist


def load_flat_field(mode):
    """
    Load and normalize flat-field reference at IMG/<mode>cali/flat_white.jpg.
    """
    path = os.path.join(BASE_IMG_DIR, f'{mode}cali', 'flat_white.jpg')
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f'Flat-field file not found: {path}')
    fmap = img.astype(np.float32) + 1e-3
    fmap = cv2.GaussianBlur(fmap,(101,101),0)
    fmap /= np.mean(fmap)
    print(f'Loaded flat-field for mode="{mode}", shape={fmap.shape}')
    return fmap


def capture_flat_field(camera_index, num_frames=DEFAULT_FRAMES):
    """Capture and average flat-field from specified camera."""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open camera {camera_index} for flat-field')
    print(f'Capturing {num_frames} frames from camera {camera_index} for flat-field')
    acc = None
    count = 0
    while count < num_frames:
        ret, frame = cap.read()
        if not ret: continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        acc = gray if acc is None else acc + gray
        count += 1
    cap.release()
    fmap = acc/num_frames + 1e-3
    fmap = cv2.GaussianBlur(fmap,(101,101),0)
    fmap /= np.mean(fmap)
    print(f'Captured flat-field, shape={fmap.shape}')
    return fmap


def save_results(mtx, dist, flat_map, mode, cam_idx):
    """Save calibration results to JSON."""
    out_dir = os.path.join(BASE_OUTPUT, mode)
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    data = {
        'mode': mode,
        'camera_index': cam_idx,
        'timestamp': ts,
        'camera_matrix': mtx.tolist(),
        'dist_coeffs': dist.flatten().tolist(),
        'flat_map_shape': [1200,1920]
    }
    filename = JSON_TPL.format(mode=mode, timestamp=ts)
    path = os.path.join(out_dir, filename)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f'Saved results to {path}')

# --- MAIN ---

def main():
    # Select camera position
    mode_idx = draw_menu(CAM_POSITIONS, 'Select camera position:')
    mode = CAM_POSITIONS[mode_idx]

    # Select camera index
    cam_options = [f'Camera {i}' for i in CAM_INDICES] + ['Cancel']
    cam_sel = draw_menu(cam_options, 'Select camera index:')
    if cam_sel >= len(CAM_INDICES):
        print('Calibration cancelled')
        return
    cam_idx = CAM_INDICES[cam_sel]

    # Intrinsic calibration (uses sidecali/ or topcali/ directory)
    mtx, dist = calibrate_intrinsics(mode)

    doflatcali = False    
    doflatcali= input(f'Do flat calibration? (0,1):\n{doflatcali}')
    # Flat-field reference
    if doflatcali:
        try:
            is_capture_flat_field = input('Capture flat-field? (0,1):\n')
            if is_capture_flat_field:
                flat_map = capture_flat_field(cam_idx)
            else:
                flat_map = load_flat_field(mode)
        except FileNotFoundError:
            print('Flat-field Cali failed, skipping .... .... ')
    else:
        flat_map = None
        print('Flat-field calibration skipped.')
    # Save all results
    save_results(mtx, dist, flat_map, mode, cam_idx)

if __name__ == '__main__':
    main()
