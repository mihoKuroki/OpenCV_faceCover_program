import cv2

# 顔検出器のパス
cascade_path = './haarcascades/haarcascade_frontalface_default.xml'

# 元の画像を読み込む
image = cv2.imread('./sample09.png')

# グレースケールに変換
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 顔検出器の読み込み
face_cascade = cv2.CascadeClassifier(cascade_path)

# 顔の検出を行う
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# 検出された顔の領域に上書きする画像を配置
for (x, y, w, h) in faces:
    # 上書きする画像を読み込む
    overlay_image = cv2.imread('./sample.png')
    h += 30
    w += 15
    y += -30
    # 上書きする画像のサイズを調整
    overlay_image = cv2.resize(overlay_image, (w, h))

    # 顔の領域に上書きする画像を配置
    image[y:y+h, x:x+w] = overlay_image

# 結果を表示
cv2.imshow('Result', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
