import os
os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin")
import cv2
import numpy as np

class VideoMaskProcessor:
    def __init__(self, threshold: int, area_threshold_black: int, area_threshold_white: int):
        """
        :param threshold: グレースケール値の閾値 a (0–255)
        :param area_threshold: 黒領域の面積閾値 b (ピクセル数)
        """
        self.threshold = threshold
        self.area_threshold_black = area_threshold_black
        self.area_threshold_white = area_threshold_white

    def binarize_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        1.1 各フレームをグレースケール→二値化（0 or 1）する。
        :param frame: BGR カラー画像
        :return: 二値画像 (dtype=np.uint8, 値は0または1)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # THRESH_BINARY: ピクセル>=閾値 → maxval, else 0
        _, binary = cv2.threshold(gray, self.threshold, 1, cv2.THRESH_BINARY)
        return binary

    def filter_small_black_regions(self, binary: np.ndarray) -> np.ndarray:
        """
        1.2 黒(0)の連結領域を抽出し、面積<=area_threshold の領域を反転(1に)する。
        :param binary: 二値画像 (0 or 1)
        :return: 小領域が反転された二値画像
        """
        # 背景(0)領域をラベリングするため、一旦反転
        inv = 1 - binary
        # connectivity=8 で連結成分解析
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
        # stats[:, cv2.CC_STAT_AREA] に各ラベルの面積が入っている
        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area <= self.area_threshold_black:
                # 元の binary 上で、そのラベルの位置を1にセット
                binary[labels == label] = 1
        return binary
    
    def filter_small_white_regions(self, binary: np.ndarray) -> np.ndarray:
        """
        1.2 白(1)の連結領域を抽出し、面積<=area_threshold の領域を反転(0に)する。
        :param binary: 二値画像 (0 or 1)
        :return: 小領域が反転された二値画像
        """
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary.astype(np.uint8), connectivity=8
        )
        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area <= self.area_threshold_white:
                binary[labels==label] = 0
            
        return binary


    def save_mask_video(self, input_path: str, output_path: str = None):
        """
        1.3 入力動画を読み込み、各フレームを処理した結果を
        「元の名前 + '_mask.mp4'」として保存する。
        :param input_path: 入力動画ファイルパス
        :param output_path: 出力ファイルパス (None の場合、自動生成)
        """
        if output_path is None:
            base, _ = os.path.splitext(input_path)
            output_path = f"{base}_mask.mp4"

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {input_path}")

        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # isColor=False でグレースケール動画として保存
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # 二値化
            bin_f = self.binarize_frame(frame)
            # 小領域反転(黒穴→白)
            filt = self.filter_small_black_regions(bin_f)
            # 小領域反転(白穴→黒)
            filt = self.filter_small_white_regions(filt)
            # 0/1 → 0/255 に変換
            mask8 = (filt * 255).astype(np.uint8)
            out.write(mask8)

        cap.release()
        out.release()
        print(f"Saved mask video to: {output_path}")

# --- 使い方例 ---
if __name__ == '__main__':
    import time
    import multiprocessing
    start = time.time()
    # video_name = "./sc01/LM.mp4"
    # start = time.time()
    # processor = VideoMaskProcessor(threshold=115, area_threshold_black=5000)
    # processor.save_mask_video(video_name)

    video_name1 = "./sc01/LM.mp4"
    video_name2 = "./sc01/FDP.mp4"
    video_name3 = "./sc01/FDS.mp4"
    video_name4 = "./sc01/Extensor.mp4"

    processor1 = VideoMaskProcessor(threshold=115, area_threshold_black=5000, area_threshold_white=1000)
    processor2 = VideoMaskProcessor(threshold=115, area_threshold_black=5000,area_threshold_white=1000)
    processor3 = VideoMaskProcessor(threshold=115, area_threshold_black=5000,area_threshold_white=1000)
    processor4 = VideoMaskProcessor(threshold=115, area_threshold_black=5000,area_threshold_white=1000)

    with multiprocessing.Manager() as process_manager:
        process_1 = multiprocessing.Process(
            target=processor1.save_mask_video, args=(video_name1,)
        )
        process_2 = multiprocessing.Process(
            target=processor2.save_mask_video, args=(video_name2,)
        )
        process_3 = multiprocessing.Process(
            target=processor3.save_mask_video, args=(video_name3,)
        )
        process_4 = multiprocessing.Process(
            target=processor4.save_mask_video, args=(video_name4,)
        )
        process_1.start()
        process_2.start()
        process_3.start()
        process_4.start()

        process_1.join()
        process_2.join()
        process_3.join()
        process_4.join()
    
    seconds = time.time()-start
    minutes = (time.time()-start)/60
    print('processing time: ', minutes, ' min')