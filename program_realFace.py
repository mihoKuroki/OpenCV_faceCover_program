import cv2

# 顔検出器のパス
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

# 顔検出器の読み込み
face_cascade = cv2.CascadeClassifier(cascade_path)

# 上書きする画像の読み込み
overlay_image = cv2.imread('./sample10.png')

# カメラキャプチャの開始
cap = cv2.VideoCapture(0)

while True:
    # フレームの読み込み
    ret, frame = cap.read()

    if not ret:
        break

    # グレースケールに変換
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 顔の検出を行う
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))

    # 検出された顔の領域に上書きする画像を配置
    for (x, y, w, h) in faces:
        # 上書きする画像のサイズを調整
        resized_overlay = cv2.resize(overlay_image, (w, h))

        # 顔の領域に上書きする画像を配置
        frame[y:y+h, x:x+w] = resized_overlay

    # 結果を表示
    cv2.imshow('Result', frame)

    # 'q'キーを押して終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# キャプチャの解放とウィンドウの破棄
cap.release()
cv2.destroyAllWindows()

#顔検出した画像を保存する 
cv2.imwrite('./sample_cascade.avi', frame)
