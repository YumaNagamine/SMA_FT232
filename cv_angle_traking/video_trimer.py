import os
os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin")
import cv2

def extract_video_segment(video_path, start_frame, end_frame):
    # 動画の読み込み
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("動画を開けませんでした:", video_path)
        return

    # 動画の基本情報取得
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print('total_frames:', total_frames)
    
    # end_frameが0の場合は動画の最後までを対象とする
    if end_frame == 0:
        end_frame = total_frames - 1

    codec = cv2.VideoWriter_fourcc(*'mp4v')  # 出力用のコーデック（mp4用）

    # 出力ファイルパスの生成：元のファイル名 + "captured" + ".mp4"
    dir_name = os.path.dirname(video_path)
    base_name = os.path.basename(video_path)
    name, ext = os.path.splitext(base_name)
    output_filename = f"{name}_trimed.mp4"
    output_path = os.path.join(dir_name, output_filename)

    # 動画書き込みオブジェクトの生成
    out = cv2.VideoWriter(output_path, codec, fps, (width, height))

    # フレーム番号で位置を調整
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current_frame = start_frame

    while cap.isOpened() and current_frame <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        # フレームを書き込み
        out.write(frame)
        current_frame += 1

    # リソースの解放
    cap.release()
    out.release()
    print("切り出しが完了しました。保存先:", output_path)

# 例: "sample.mp4" の 100 フレームから動画最後までを切り出す場合
if __name__ == "__main__":
    video_file = "./sc01/FDP_LM.mp4"  # 動画のパスを指定
    start = 500  # 切り出し開始フレーム番号
    end = 0      # 0の場合、動画の最後まで切り出す
    extract_video_segment(video_file, start, end)