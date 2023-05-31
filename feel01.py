from feat import Detector
import numpy as np
import cv2

detector = Detector(
    face_model="retinaface",
    landmark_model="mobilefacenet",
    au_model='jaanet', # ['svm', 'logistic', 'jaanet']
    emotion_model="resmasknet",
)

# single_face_prediction = detector.detect_image("single_face.jpg")
single_face_prediction = detector.detect_image("single_face.jpg", outputFname = "output.csv")

print(single_face_prediction.facebox)
print(single_face_prediction.aus)
print(single_face_prediction.emotions)
print(single_face_prediction.facepose) # (in degrees)
# figs = single_face_prediction.plot_detections(poses=True)
figs = single_face_prediction.plot_detections(faces='aus', muscles=True)
print(len(figs)) # 1

figs[0].canvas.draw()
image = np.array(figs[0].canvas.renderer.buffer_rgba())
# image = np.array(figs[0].canvas.renderer._renderer)
image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)

cv2.imshow("image", image)
cv2.waitKey(0)
