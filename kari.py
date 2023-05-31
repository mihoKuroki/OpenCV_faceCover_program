import cv2
import numpy as np
 
# モジュール読み込み 
import sys
sys.path.append('/opt/intel/openvino/python/python3.5/armv7l')
from openvino.inference_engine import IENetwork, IEPlugin
 
# ターゲットデバイスの指定 
plugin = IEPlugin(device="MYRIAD")
 
# モデルの読み込み（顔検出） 
net = IENetwork(model='FP16/face-detection-retail-0004.xml', weights='FP16/face-detection-retail-0004.bin')
exec_net = plugin.load(network=net)
 
# モデルの読み込み（感情分類） 
net_emotion = IENetwork(model='FP16/emotions-recognition-retail-0003.xml', weights='FP16/emotions-recognition-retail-0003.bin')
exec_net_emotion = plugin.load(network=net_emotion)
 
# カメラ準備 
cap = cv2.VideoCapture(0)
 
# メインループ 
while True:
    ret, frame = cap.read()
 
    # Reload on error 
    if ret == False:
        continue
 
    # 入力データフォーマットへ変換 
    img = cv2.resize(frame, (300, 300))   # サイズ変更 
    img = img.transpose((2, 0, 1))    # HWC > CHW 
    img = np.expand_dims(img, axis=0) # 次元合せ 
 
    # 推論実行 
    out = exec_net.infer(inputs={'data': img})
 
    # 出力から必要なデータのみ取り出し 
    out = out['detection_out']
    out = np.squeeze(out) #サイズ1の次元を全て削除 
 
    # 検出されたすべての顔領域に対して１つずつ処理 
    for detection in out:
        # conf値の取得 
        confidence = float(detection[2])
 
        # バウンディングボックス座標を入力画像のスケールに変換 
        xmin = int(detection[3] * frame.shape[1])
        ymin = int(detection[4] * frame.shape[0])
        xmax = int(detection[5] * frame.shape[1])
        ymax = int(detection[6] * frame.shape[0])
 
        # conf値が0.5より大きい場合のみ感情推論とバウンディングボックス表示 
        if confidence > 0.5:
           # 顔検出領域はカメラ範囲内に補正する。特にminは補正しないとエラーになる 
            if xmin < 0:
                xmin = 0
            if ymin < 0:
                ymin = 0
            if xmax > frame.shape[1]:
                xmax = frame.shape[1]
            if ymax > frame.shape[0]:
                ymax = frame.shape[0]
 
            # 顔領域のみ切り出し 
            frame_face = frame[ ymin:ymax, xmin:xmax ]
 
            # 入力データフォーマットへ変換 
            img = cv2.resize(frame_face, (64, 64))   # サイズ変更 
            img = img.transpose((2, 0, 1))    # HWC > CHW 
            img = np.expand_dims(img, axis=0) # 次元合せ 
 
            # 推論実行 
            out = exec_net_emotion.infer(inputs={'data': img})
 
            # 出力から必要なデータのみ取り出し 
            out = out['prob_emotion']
            out = np.squeeze(out) #不要な次元の削減 
 
            # 出力値が最大のインデックスを得る 
            index_max = np.argmax(out)
 
            # 各感情の文字列をリスト化 
            list_emotion = ['neutral', 'happy', 'sad', 'surprise', 'anger']
 
            # 文字列描画 
            cv2.putText(frame, list_emotion[index_max], (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
 
            # バウンディングボックス表示 
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color=(240, 180, 0), thickness=3)
 
            # 棒グラフ表示 
            str_emotion = ['neu', 'hap', 'sad', 'sur', 'ang']
            text_x = 10
            text_y = frame.shape[0] - 180
            rect_x = 80
            rect_y = frame.shape[0] - 200
            for i in range(5):
                cv2.putText(frame, str_emotion[i], (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (240, 180, 0), 2)
                cv2.rectangle(frame, (rect_x, rect_y), (rect_x + int(300 * out[i]), rect_y + 20), color=(240, 180, 0), thickness=-1)
                text_y = text_y + 40
                rect_y = rect_y + 40
 
            # １つの顔で終了 
            break
 
    # 画像表示 
    cv2.imshow('frame', frame)
 
    # 何らかのキーが押されたら終了 
    key = cv2.waitKey(1)
    if key != -1:
        break
 
# 終了処理 
cap.release()
cv2.destroyAllWindows()