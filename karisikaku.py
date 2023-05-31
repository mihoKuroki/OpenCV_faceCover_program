import cv2
import numpy as np

def detect_rectangle(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rectangles = []
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4:
            rectangles.append(approx)

    return rectangles

# 元の画像を読み込む
image = cv2.imread('./sample03.png')

# 四角形の検出を行う
rectangles = detect_rectangle(image)

if rectangles:
    for rectangle in rectangles:
        # 四角形の各頂点の座標を取得
        x, y, w, h = cv2.boundingRect(rectangle)

        # 上書きする画像を読み込む
        overlay_image = cv2.imread('./sample.png')

        # 上書きする画像のサイズを調整
        overlay_image = cv2.resize(overlay_image, (w, h))

        # 四角形の上に上書きする画像を配置
        image[y:y+h, x:x+w] = overlay_image

# 結果を表示
cv2.imshow('Result', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
