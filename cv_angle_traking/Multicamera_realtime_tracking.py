# Realtime tracking for multicamera
import time,os
# os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin")
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy, time
from datetime import datetime

# Realtime tracking; Receiving image and extracting angle, 3D positions
# Save the raw video, video with text and line, and csv file with angles and 3D positions

# constants
NUMBER_OF_MASK = 4

class AngleTracking: 
    def __init__(self):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M') # e.g. 20250627_1557
        self.raw_videofile_name_side = timestamp + '_raw_side.mp4'
        self.raw_videofile_name_top = timestamp + '_raw_top.mp4'
        self.extracted_videofile_name_side = timestamp + '_extracted_side.mp4'
        self.extracted_videofile_name_top = timestamp + '_extracted_top.mp4'

        self.raw_frames_side = []
        self.raw_frames_top = []
        self.extracted_frames_side = []
        self.extracted_frames_top = []
        self.measure = []
        # preset params
        self.threshold_area_size = [200, 50, 50, 10]
        self.colors = [(255,0,0), (127,0,255), (0,127,0),(0,127,255)]

    def set_params(self, theta_side, theta_top, distance):
        self.theta_side = theta_side
        self.theta_top = theta_top
        self.distance = distance

    @staticmethod            
    def calculate_distance(point0, point1):
        point0 = np.array(point0)
        point1 = np.array(point1)
        return np.linalg.norm(point0-point1)

    def _add_text_to_frame(self, frame, text, position=(30, 30), font=cv2.FONT_HERSHEY_DUPLEX, font_scale=0.2, color=(0, 255, 0), thickness=2):
        # Add text into video frame
        frame_with_text = copy.deepcopy(frame)
        cv2.putText(frame_with_text, text, position, font, font_scale, color, thickness)
        return frame_with_text

    def _rotate_scaling_vector(self, vector:np.ndarray, theta: float, rate:float = 1) -> np.ndarray:
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
        rotated_vec = rotation_matrix @ vector
        norm = np.linalg.norm(rotated_vec)
        processed_vector = np.array(rate * rotated_vec, dype=int)
        return processed_vector # 要素がintのベクトル
    
    def _translation_markers(self, markers:list, d:float): # markers should be [distal, proximal], each element: [x(int), y(int)]
        # markersから構成されるベクトルをdだけ平行移動する
        markers_np = np.array(markers)
        vector = markers_np[0] - markers_np[1] # 近位から遠位へのベクトル, 成分はint
        shift_vector = self._rotate_scaling_vector(vector, theta=np.pi/2, rate=(d/np.linalg.norm(vector))) # 元のベクトルを90°回転したノルムdのベクトル、成分はint
        modified_markers = np.array([markers_np[0] + shift_vector, markers_np[1] + shift_vector])
        return modified_markers

    def _marker_discriminator(self, markers): #中節骨と基節骨のマーカーを区別する　[遠位のマーカー　近位のマーカー]にする
        distance0 = self.calculate_distance(self.palm_marker_position, markers[0])
        distance1 = self.calculate_distance(self.palm_marker_position, markers[1])
        # print('distances:', distance0, distance1)
        if distance0 > distance1:
            return markers
        elif distance0 < distance1:
            markers[0], markers[1] = markers[1], markers[0]
            return markers

    def _marker_discriminator_distalis(self, markers): #末節骨のマーカーを区別する　[遠位のマーカー　近位のマーカー]にする
        # distalmarker_mediaは中節骨の遠位のマーカー
        distance0 = self.calculate_distance(self.media_distalis, markers[0])
        distance1 = self.calculate_distance(self.media_distalis, markers[1])
        if distance0 > distance1:
            return markers
        elif distance0 < distance1:
            markers[0], markers[1] = markers[1], markers[0]
            return markers
        
    def _segment_marker_by_color(self, frame, side): # using OpenCV. Becareful; Input frame is RGB
        cielab = cv2.cvtColor(frame, cv2.COLOR_RGB2Lab)
        self.marker_rangers = np.array(self.marker_rangers)
        if side: # side camera
            masks = []
            for rng in self.marker_rangers:
                rng = rng.T
                lowerb = tuple(int(x) for x in rng[0])
                upperb = tuple(int(x) for x in rng[1])
                mask = cv2.inRange(cielab, lowerb, upperb)
                masks.append(mask > 0)
            # masks = np.stack(mask)
            return masks
        else: # top camera
            # rng = self.fingertip_range[0].T # brown marker
            lowerb = tuple(int(x) for x in self.fingertip_range[0][0])
            upperb = tuple(int(x) for x in self.fingertip_range[0][1])
            mask = cv2.inRange(cielab, lowerb, upperb)
            return [mask > 0]
        
    def _extract_angle_side(self, frame) -> tuple:
        masks = self._segment_marker_by_color(frame, side=True)
        markerset_per_frame = []
        processed_markerset_per_frame = []

        line_pad = 5
        direction_vector_0_1 = None
        
        # Process each mask
        for mask_id, mask, thr, direction_vector in zip(range(NUMBER_OF_MASK), masks, self.threshold_area_size, self.color, [direction_vector_0_1, None, None, None]):
            try:
                mask = np.uint8(mask) # True/False -> 0/1
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
                filtered_regions = [i for i, stat in enumerate(stats[1:]) if stat[4] >= thr]
                point_per_mask = []
                processed_point_per_mask = []
                
                # Go to Next mask if missing point
                if len(filtered_regions) < 2 and (not mask_id==3):
                    point_per_mask.extend([(-1,-1),(-1,-1)])
                    markerset_per_frame.append(point_per_mask)
                    processed_point_per_mask.extend([(-1,-1),(-1,-1)])
                    processed_markerset_per_frame.append(processed_point_per_mask)
                    continue

                # Process each filtered region in the mask
                for idx, index  in enumerate(filtered_regions):
                    left, top, width, height = stats[index + 1]

                    centroid_x, centroid_y = int(left + width / 2), int(top + height / 2)
                    point_per_mask.append((centroid_x, centroid_y))
                    if mask_id == 0 and idx == 1: # Distalis bone 指先の骨
                        point_per_mask = self._marker_discriminator_distalis(point_per_mask)
                    elif (mask_id == 1 and idx == 1) or (mask_id == 2 and idx == 1):
                        self.media_distalis = point_per_mask[0]
                    elif mask_id == 3 and idx == 0:
                        self.palm_marker_position = np.array([centroid_x, centroid_y])

                #---- markerの位置修正-------
                if mask_id == 0:
                    processed_distalis = 
                elif mask_id == 1:
                elif mask_id == 2:
                elif mask_id == 3:
                #---- markerの位置修正終了-------
            except:
                pass

    def _extract_angle_top(self, frame):
        # 指先は二点の中点をとる
        pass

    def _calculate_angle_side(self, distalis_line: list, # each line; [(distalis marker), (proximal marker)]
                        proximal_line: list
                        ) -> np.float32:
        try:
            distalis_line = np.array(distalis_line)
            proximal_line = np.array(proximal_line)
            distalis_vec = distalis_line[0] - distalis_line[1] # the vector from proximal marker to distalis marker
            proximal_vec = proximal_line[0] - proximal_line[1]

            dot_product = np.dot(proximal_vec, distalis_vec)
            cross_product = np.cross(proximal_vec, distalis_vec)
            angle_rad = np.arctan2(cross_product, dot_product)
            angle_degree = np.degrees(angle_rad)
            if angle_rad < 0:
                joint_angle = 180 + abs(angle_degree)
            else:
                joint_angle = 180 - angle_degree

            return joint_angle
        except:
            return []

    def _calculate_angle_top(self, frame_top) -> np.float32:

        pass

    def _extract_angle(self, frame_side, frame_top) -> tuple:
        pass
         
    def _video_saver(self, directory_to_save: str , raw_frame_side, raw_frame_top, extracted_frame_side, extracted_frame_top):
        self.raw_frames_side.append(raw_frame_side)
        self.raw_frames_top.append(raw_frame_top)
        self.extracted_frames_side.append(extracted_frame_side)
        self.extracted_frames_top.append(extracted_frame_top)

    def _data_saver(self, data_to_save: list):
        self.measure.append(data_to_save)
        pass
    def video_saver_finalize(self, directory_to_save: str, fps: int, resolution: tuple): # resolution; (width, height)
        fourcc = cv2.VideoWriter_fourcc('M','J','P','G') # Win
        # fourcc = cv2.VideoWriter_fourcc(*'x264')# # 'avc1' # Mac

        out = cv2.VideoWriter(directory_to_save + self.raw_videofile_name_side, fourcc, fps, resolution)
        for frame in self.raw_frames_side: out.write(frame)
        out.release()

        out = cv2.VideoWriter(directory_to_save + self.raw_videofile_name_top, fourcc, fps, resolution)
        for frame in self.raw_frames_top: out.write(frame)
        out.release()

        out = cv2.VideoWriter(directory_to_save + self.extracted_frames_side, fourcc, fps, resolution)
        for frame in self.extracted_frames_side: out.write(frame)
        out.release()

        out = cv2.VideoWriter(directory_to_save + self.extracted_frames_top, fourcc, fps, resolution)
        for frame in self.extracted_frames_top: out.write(frame)
        out.release()
    
    
    def _processing_first_frame():
        pass
    def processing_frame(frame_side: np.ndarray, frame_top: np.ndarray, is_first_frame=False) ->  tuple: # Execute this method for each frame
        pass
if __name__ == "__main__":
    pass