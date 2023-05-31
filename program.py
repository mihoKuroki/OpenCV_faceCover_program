import cv2

# 学習済みモデルの読み込み
cascade_path = "./haarcascades/haarcascade_frontalface_default.xml"
cascade = cv2.CascadeClassifier(cascade_path)


def VideoCapt():
    cap = cv2.VideoCapture(0)  # カメラが複数あるときは 0を1や2にする
    aveSize = 0
    count = 0
    while cap.isOpened():
        try:
            faceImg = cv2.imread('./sample.png', cv2.IMREAD_UNCHANGED)
            _, frame = cap.read()
            # グレースケール変換
            image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            facerect = cascade.detectMultiScale(
                image_gray, scaleFactor=1.1, minNeighbors=2, minSize=(30, 30))

            if len(facerect) > 0:
                for rect in facerect:
                    count += 1
                    if count < 4:
                        aveSize += (rect[2]+rect[3])/2
                        break
                    elif count == 4:
                        aveSize += (rect[2]+rect[3])/2
                        aveSize /= 5
                    else:
                        aveSize = aveSize*0.8+rect[2]*0.1+rect[3]*0.1
                    thresh = aveSize*0.95  # 移動平均の95%以上を閾値
                    if rect[2] < thresh or rect[3] < thresh:
                        break
                    
                    # # 検出した顔を囲む矩形の作成
                    # cv2.rectangle(frame, tuple(rect[0:2]), tuple(rect[0:2]+rect[2:4]), (255, 100, 155), thickness=-1)
                    
                    # 顔のサイズを調整して画像を重ねる
                    resized_face = cv2.resize(faceImg, ((int)(rect[2]*1.3), (int)(rect[3]*1.3)), cv2.IMREAD_UNCHANGED)
                    if resized_face.shape[2] == 0:
                        continue
                    x_offset = int(rect[0] - rect[2]*0.15)
                    y_offset = int(rect[1] - rect[3]*0.15)
                    alpha = resized_face[:, :, 3:] / 255.0
                    blended_face = frame[y_offset:y_offset+resized_face.shape[0], x_offset:x_offset+resized_face.shape[1]] * (1 - alpha) + resized_face[:, :, :3] * alpha
                    frame[y_offset:y_offset+resized_face.shape[0], x_offset:x_offset+resized_face.shape[1]] = blended_face

            cv2.imshow('cv2', frame)

        except Exception as e:
            print("An error occurred:", str(e))
            pass
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Camera is closed.")

if __name__ == '__main__':
    VideoCapt()
