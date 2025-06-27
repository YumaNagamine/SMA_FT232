import cv2

def preprocess(image, width, height):
    """
    画像をリサイズし、[0,1]正規化、(C,H,W)に変換。
    """
    img = cv2.resize(image, (width, height))
    img = img.astype('float32') / 255.0
    # OpenCVはBGR順なので必要に応じRGB変換を追加
    return img.transpose(2, 0, 1)
