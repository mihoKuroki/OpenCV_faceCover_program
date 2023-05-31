import cv2
import numpy as np
from feat import Detector
from PIL import Image

# Py-Featの初期化
detector = Detector(
    face_model="retinaface",
    landmark_model="mobilefacenet",
    au_model='svm',  # 有効な値に修正する
    emotion_model="resmasknet",
)

# カメラキャプチャの初期化
video_capture = cv2.VideoCapture(0)

while True:
    # フレームの読み込み
    ret, frame = video_capture.read()

    # フレームをリサイズして処理を高速化する（オプション）
    frame = cv2.resize(frame, (640, 480))

    # 表情分析を実行
    # 画像をNumPy配列からPIL Imageに変換する
    frame_pil = Image.fromarray(frame)
    predictions = detector.detect_image(frame_pil)

    # 検出結果の可視化
    frame = predictions.plot_detections(frame, faces='aus', muscles=True)

    # 結果の表示
    cv2.imshow('Facial Expression Analysis', frame)

    # 'q'キーを押してループを中断
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# キャプチャの解放
video_capture.release()

# ウィンドウの破棄
cv2.destroyAllWindows()
