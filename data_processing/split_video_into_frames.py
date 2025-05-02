import os
os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin")
import cv2
import pandas as pd

def extract_and_save_frames(video_path: str, csv_path: str):
    """
    mp4 -> jpg
    :param video_path: 抽出元の動画ファイル (.mp4)
    :param csv_path: フレーム番号が入ったCSVファイルパス
    """
    # 出力ディレクトリを作成
    base_dir = os.path.dirname(video_path)
    output_dir = os.path.join(base_dir, 'frames')
    os.makedirs(output_dir, exist_ok=True)

    # CSV読み込み（'frame'列を想定）
    df = pd.read_csv(csv_path)
    frame_numbers = sorted(df['frame'].astype(int).unique())

    # 動画を開く
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    for frame_no in frame_numbers:
        # フレーム位置を設定
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: frame {frame_no} could not be read.")
            continue

        # 出力ファイル名
        out_path = os.path.join(output_dir, f"{frame_no}.jpg")
        # JPEGで保存
        cv2.imwrite(out_path, frame)
        # print(f"Saved frame {frame_no} -> {out_path}")

    cap.release()
    print("Done.")

if __name__ == "__main__":

    corename = 'FDS'

    video_path = './sc01/' + corename + '/' + corename + '_mask.mp4'
    csv_path   = './sc01/' + corename + '/annotations.csv'

    extract_and_save_frames(video_path, csv_path)