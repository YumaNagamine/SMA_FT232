# mode1_data = 0b00011000
# print(mode1_data & 239,mode1_data&0xEF,mode1_data|128)

# print("\n\n22")
# a = str(bin(mode1_data))

# print(mode1_data,a)
# print("0"*4)


 
# # from asyncio.tasks import _T1
# import numpy as np,time
# import cv2 as cv
# # 创建VideoCapture，传入0即打开系统默认摄像头
# vc = cv.VideoCapture(0)
# cv.set
# while(True):
#     # 读取一帧，read()方法是其他两个类方法的结合，具体文档
#     # ret为bool类型，指示是否成功读取这一帧
#     t0 = time.time()
#     ret, frame = vc.read()
#     # 就是个处理一帧的例子，这里转为灰度图
#     t1 = time.time()

#     print(t1-t0)
#     print(type(frame))

#     gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#     # 不断显示一帧，就成视频了
#     # 这里没有提前创建窗口，所以默认创建的窗口不可调整大小
#     # 可提前使用cv.WINDOW_NORMAL标签创建个窗口
#     cv.imshow('frame',gray)
#     # 若没有按下q键，则每1毫秒显示一帧
#     if cv.waitKey(1) & 0xFF == ord('q'):
#         break
 
# # 所有操作结束后不要忘记释放
# vc.release()
# cv.destroyAllWindows()

import cv2

def list_camera_indices(max_index=10):
    """0～max_index-1 のインデックスを順に試し、Openに成功したものを返す"""
    available = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # WindowsならCAP_DSHOW推奨。Linuxなら引数なしでもOK
        if not cap.isOpened():
            cap.release()
            continue
        # フレームを１枚読み取れるか確認
        ret, _ = cap.read()
        if ret:
            available.append(i)
        cap.release()
    return available

if __name__ == "__main__":
    cams = list_camera_indices(5)  # 最初の５台分をチェック
    if cams:
        print(f"利用可能なカメラインデックス: {cams}")
    else:
        print("カメラが見つかりませんでした。")