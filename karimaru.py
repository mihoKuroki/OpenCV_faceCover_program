import cv2
import numpy as np

# 元の画像を読み込む
image = cv2.imread('./sample03.png')

# グレースケールに変換
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 円の検出を行う
circles = cv2.HoughCircles(
    gray, cv2.HOUGH_GRADIENT, dp=1, minDist=100, param1=50, param2=30, minRadius=10, maxRadius=100
)

if circles is not None:
    circles = np.uint16(np.around(circles))

    for circle in circles[0, :]:
        # 円の中心座標と半径を取得
        center_x, center_y, radius = circle[0], circle[1], circle[2]

        # 上書きする画像を読み込む
        overlay_image = cv2.imread('./greenImage.jpg')

        # 上書きする画像のサイズを調整
        overlay_image = cv2.resize(overlay_image, (radius * 2, radius * 2))

        # 円の中心に上書きする画像を配置
        x = center_x - radius
        y = center_y - radius
        image[y:y+overlay_image.shape[0], x:x+overlay_image.shape[1]] = overlay_image

# 結果を表示
cv2.imshow('Result', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
