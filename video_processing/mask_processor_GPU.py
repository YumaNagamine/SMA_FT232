import os
os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin")
import cv2
import numpy as np

class GpuVideoMaskProcessor:
    def __init__(self, threshold: int, area_threshold: int, device: int = 0):
        """
        :param threshold: グレースケール値の閾値 a (0–255)
        :param area_threshold: 黒領域の面積閾値 b (ピクセル数)
        :param device: CUDA デバイス番号
        """
        # CUDA デバイスチェック
        if cv2.cuda.getCudaEnabledDeviceCount() == 0:
            raise RuntimeError("CUDA 対応の OpenCV ビルドではありません")
        cv2.cuda.setDevice(device)

        self.threshold = threshold
        self.area_threshold = area_threshold

    def binarize_frame_gpu(self, frame: np.ndarray) -> np.ndarray:
        """
        GPU 上でグレースケール→二値化を行い、CPU に戻す。
        :param frame: BGR カラー画像
        :return: 二値画像 (dtype=np.uint8, 値は0または1)
        """
        # GPU メモリにアップロード
        gpu_frame = cv2.cuda_GpuMat()
        gpu_frame.upload(frame)

        # グレースケール変換 (BGR→GRAY)
        gpu_gray = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2GRAY)

        # 二値化 (0 or 1)
        # cv2.cuda.threshold は (retval, GpuMat) を返す
        _, gpu_binary = cv2.cuda.threshold(
            gpu_gray, 
            self.threshold, 
            1, 
            cv2.THRESH_BINARY
        )

        # CPU メモリにダウンロード
        binary = gpu_binary.download()
        return binary

    def filter_small_black_regions(self, binary: np.ndarray) -> np.ndarray:
        """
        （CPU）黒(0)の連結領域を抽出し、面積<=area_threshold の領域を反転(1に)する。
        :param binary: 二値画像 (0 or 1)
        :return: 小領域が反転された二値画像
        """
        inv = 1 - binary
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
        for label in range(1, num_labels):
            if stats[label, cv2.CC_STAT_AREA] <= self.area_threshold:
                binary[labels == label] = 1
        return binary

    def save_mask_video(self, input_path: str, output_path: str = None):
        """
        動画をフレーム単位で処理し、「元の名前+'_mask.mp4'」として保存。
        :param input_path: 入力動画パス
        :param output_path: 出力動画パス（None で自動生成）
        """
        if output_path is None:
            base, _ = os.path.splitext(input_path)
            output_path = f"{base}_mask.mp4"

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {input_path}")

        w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h), isColor=False)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # GPU で二値化→CPU に戻す
            bin_f = self.binarize_frame_gpu(frame)
            # 小領域除去 (CPU)
            filt  = self.filter_small_black_regions(bin_f)
            # 0/1→0/255 にスケール
            mask8 = (filt * 255).astype(np.uint8)

            out.write(mask8)

        cap.release()
        out.release()
        print(f"Saved GPU-accelerated mask video to: {output_path}")


# --- 使い方例 ---
if __name__ == '__main__':
    import time
    # threshold=100, area_threshold=500 でデバイス0 を使う例
    video_name = "./sc01/test.mp4"
    start = time.time()
    gpu_proc = GpuVideoMaskProcessor(threshold=100, area_threshold=500, device=0)
    gpu_proc.save_mask_video(video_name)
    print("processing time", time.time()-start)