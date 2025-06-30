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
        self.marker_rangers = [[[103, 163],[132, 162],[64, 94]], 
                                [[86, 146],[160, 190], [46, 76]], 
                                [[139, 219], [138, 178], [146, 206]],
                                [[214, 234],[68, 88],[112, 132]]]
        self.fingertip_range = [[103, 132, 64], [163, 162, 94]]
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
    @staticmethod
    def calculate_vactor(point1, point2): # vector point1 -> point2
        return np.array(point2) - np.array(point1)

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
        
        # Process each mask
        for mask_id, mask, thr, color, direction_vector in zip(range(NUMBER_OF_MASK), masks, self.threshold_area_size, self.colors):
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

                    # Discriminate the markers whether distal or proximal
                    if mask_id == 0 and idx == 1: # Distalis bone 指先の骨
                        point_per_mask = self._marker_discriminator_distalis(point_per_mask)
                    elif (mask_id == 1 and idx == 1) or (mask_id == 2 and idx == 1):
                        point_per_mask = self._marker_discriminator(point_per_mask)
                        self.media_distalis = point_per_mask[0]
                    elif mask_id == 3 and idx == 0:
                        self.palm_marker_position = np.array([centroid_x, centroid_y])

                #---- markerの位置修正-------
                if mask_id == 0:
                    processed_distal = point_per_mask[0]
                    processed_point_per_mask.append(tuple(processed_distal))
                    processed_proximal = point_per_mask[1]
                    processed_point_per_mask.append(tuple(processed_proximal))

                elif mask_id == 1:
                    marker_vec = self.calculate_vactor(point_per_mask[0], point_per_mask[1])
                    rotated_vector = self._rotate_scaling_vector(marker_vec, self.theta_side)
                    processed_distal = np.array(point_per_mask[0])
                    processed_proximal = processed_distal + rotated_vector
                    processed_point_per_mask.append(tuple(processed_distal))
                    processed_point_per_mask.append(tuple(processed_proximal))
                elif mask_id == 2:
                    processed_distal, processed_proximal = self._translation_markers(point_per_mask, self.distance)
                    processed_point_per_mask.append(processed_distal)
                    processed_point_per_mask.append(processed_proximal)

                elif mask_id == 3:
                    processed_point_per_mask = point_per_mask.copy()
                    processed_point_per_mask.append((point_per_mask[0][0] + 100, point_per_mask[0][1]))
                #---- markerの位置修正終了-------

                # Visualize circles on markers
                for idx, point in enumerate(point_per_mask):
                    cv2.circle(frame, (point[0], point[1]), radius = idx * 10, color=[255,255,255], thickness=2)
                for idx, point in enumerate(point_per_mask):
                    cv2.circle(frame, (point[0], point[1]), radius = (idx + 1) * 10, color=[255,255,255], thickness=2)

                # Visualize circles and line on processed points
                for idx, point in enumerate(processed_point_per_mask):
                    cv2.circle(frame, (point[0], point[1]), radius = idx * 10, color=color, thickness=2)
                for idx, point in enumerate(processed_point_per_mask):
                    cv2.circle(frame, (point[0], point[1]), radius = (idx + 1) * 10, color=color, thickness=2)
                
                direction_vector = self.calculate_vactor(processed_point_per_mask[0], processed_point_per_mask[1])
                point0 = np.array(processed_point_per_mask[0]) - line_pad * direction_vector
                point1 = np.array(processed_point_per_mask[1]) + line_pad * direction_vector
                cv2.line(frame, (point0, point1), color, 3)


                markerset_per_frame.append(point_per_mask)
                processed_markerset_per_frame.append(processed_point_per_mask)

            except:
                print(f'unknown error ocurres in extract_angle_side at mask_id; {mask_id}')
                continue
            
            # Calculate angles based on extracted marker points
            self._estimate_joint(processed_markerset_per_frame)
            cv2.circle(frame, center=self.fingertip, radius=10, color = [0,255,0], thickness=-1)
            cv2.circle(frame, center=self.DIP, radius=10, color = [0,255,0], thickness=-1)
            cv2.circle(frame, center=self.PIP, radius=10, color = [0,255,0], thickness=-1)
            cv2.circle(frame, center=self.MCP, radius=10, color = [0,255,0], thickness=-1)

            try:
                angle_0 = self._calculate_angle_side(processed_markerset_per_frame[0], processed_markerset_per_frame[1])
                angle_0 = int(10*angle_0) / 10
            except IndexError:
                angle_0 = []
            try:
                angle_1 = self._calculate_angle_side(processed_markerset_per_frame[1], processed_markerset_per_frame[2])
                angle_1 = int(10*angle_1) / 10
            except IndexError:
                angle_1 = []
            try:
                angle_2 = self._calculate_angle_side(processed_markerset_per_frame[2], processed_markerset_per_frame[3])
                angle_2 = int(10*angle_2) / 10
            except IndexError:
                angle_2 = []

            _text_pos_x = 100
            frame = self.add_text_to_frame(frame, "ANGLE 0: {}".format(angle_0), position=(_text_pos_x, 210), font_scale=1, thickness=2, color=(255, 255, 0))
            frame = self.add_text_to_frame(frame, "ANGLE 1: {}".format(angle_1), position=(_text_pos_x, 240), font_scale=1, thickness=2, color=(255, 255, 0))
            frame = self.add_text_to_frame(frame, "ANGLE 2: {}".format(angle_2), position=(_text_pos_x, 270), font_scale=1, thickness=2, color=(255, 255, 0))
            
            return frame, angle_0, angle_1, angle_2, markerset_per_frame, processed_markerset_per_frame

    def _extract_angle_top(self, frame, MCP_point:np.ndarray):
        # 指先は二点の中点をとる
        mask = self._segment_marker_by_color(frame, side=False)
        markerpos_per_frame = []
        fingertip_per_frame = []
        line_pad = 5
        threshold = 100
        try:
            mask = np.uint8(mask)
            self.binary_mask = mask * 255
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
            filtered_regions = [index for index, stat in enumerate(stats[1:]) if stat[4] > threshold]
            if len(filtered_regions) <= 1: 
                markerpos_per_frame.append((-1,-1))
                fingertip_per_frame.append((-1,-1))
                return frame, [], markerpos_per_frame, fingertip_per_frame

            # extract only 2 x ball-joint
            # max_area = 0
            areas = []
            for idx, index in enumerate(filtered_regions):
                left, top, width, height, area = stats[index+1]
                # if area > max_area:
                #     area_max_index = index
                #     max_area = area
                #     ball_joint_stats = {'left': left, 'top': top, 'width': width, 'height': height, 'area': area}
                # centroid_x, centroid_y = int(ball_joint_stats['left']+ball_joint_stats['width']/2), int(ball_joint_stats['top']+ball_joint_stats['height']/2)
                areas.append([left, top, width, height, area])
            areas.sort(key=lambda x: x[4], reverse=True)
            temp = [0,0]
            for st in areas[:2]:
                centroid_x, centroid_y = int(st[0]+st[2]/2), int(st[1]+st[3]/2)
                markerpos_per_frame.append((centroid_x, centroid_y))
                cv2.circle(frame, (centroid_x, centroid_y), radius=10, color=[255,255,255], thickness=2)
                temp[0] += centroid_x
                temp[1] += centroid_y
            fingertip_pos = [int(temp[0]/2), int(temp[1]/2)]
            fingertip_per_frame.append(fingertip_pos)


    def _estimate_joint(self, processed_markerset_per_frame, shifters=[15,110]):
        vec = np.array(processed_markerset_per_frame)
        direction_vector_0 = vec[1][0] - vec[1][1] #近位から遠位へのベクトル
        direction_vector_1 = vec[2][0] - vec[2][1]
        self.fingertip = vec[0][0]
        self.DIP = vec[1][0] + (shifters[0]/np.linalg.norm(direction_vector_0))*direction_vector_0
        self.PIP = vec[2][0] + (shifters[1]/np.linalg.norm(direction_vector_1))*direction_vector_1
        self.DIP = self.DIP.astype(np.int32)
        self.PIP = self.PIP.astype(np.int32)
        self.MCP = (vec[3][0][0] - 130, vec[3][0][1])

        # return self.fingertip, self.DIP, self.PIP, self.MCP 

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
         
    def _video_saver(self, raw_frame_side, raw_frame_top, extracted_frame_side, extracted_frame_top):
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
    
    
    def _processing_first_sideframe(self, frame):
        # how to distinguish whether markers is proximal or distal
        return frame, 

    def processing_frame(self, frame_side: np.ndarray, frame_top: np.ndarray, is_first_frame=False, duty_ratios = None) ->  tuple: 
        # Execute this method for each frame
        raw_frame_side = copy.deepcopy(frame_side)
        raw_frame_top = copy.deepcopy(frame_top)
        
        if is_first_frame:
            self._processing_first_sideframe()
        else:
            frame_side, angle_0, angle_1, angle_2, markerset_per_frame, processed_markerset_per_frame = self._extract_angle_side(frame_side)
        
        measure = [angle_0, angle_1, angle_2, angle_top,  ]

        if not duty_ratios == None:
            measure.append(duty_ratios)

        self._data_saver(measure)
        self._video_saver(raw_frame_side, raw_frame_top, frame_side, frame_top)
        return 
if __name__ == "__main__":
    pass