import cv2
from imutils.video import VideoStream
from imutils import face_utils, translate, resize
import numpy as np
import time
import dlib


def black_face():
    vs = VideoStream().start()
    time.sleep(1.5)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    eye_layer = np.zeros((450, 800, 3), dtype='uint8')
    eye_mask = eye_layer.copy()
    eye_mask = cv2.cvtColor(eye_mask, cv2.COLOR_BGR2GRAY)
    while True:
        frame = vs.read()
        frame = resize(frame, width=800)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)
        eye_layer.fill(0)
        eye_mask.fill(0)
        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            mouth = shape[60:69]
            mouth_c = shape[48:60]
            left_eye = shape[36:42]
            right_eye = shape[42:48]

            cv2.fillPoly(eye_mask, [mouth], 255)
            cv2.fillPoly(eye_mask, [mouth_c], 255)
            cv2.fillPoly(eye_mask, [left_eye], 255)
            cv2.fillPoly(eye_mask, [right_eye], 255)

            eye_layer = cv2.bitwise_and(frame, frame, mask=eye_mask)
        cv2.imshow("Black Face", eye_layer)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
    cv2.destroyAllWindows()
    vs.stop()


if __name__ == "__main__":
    black_face()
