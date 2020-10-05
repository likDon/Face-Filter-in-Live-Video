import cv2
from imutils.video import VideoStream
from imutils import face_utils, translate, resize
import numpy as np
import time
import dlib


def tags(tags_index):
    vs = VideoStream().start()
    time.sleep(1.5)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    # predictor = dlib.shape_predictor(args.predictor)

    filters = ['04.png', 'paw1.png',
               'hat.png', 'eye.png',
               'beard_1.png','beard_2.png',
               'paw2.png','paw3.png',
               'paw4.png','shiny01.png',
               'shiny02.png','shiny03.png']

    paw_index=0
    paw = [1,6,7,8]
    while True:
        frame = vs.read()
        frame = resize(frame, width=800)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)
        frame_rows, frame_cols, frame_ch = frame.shape
        for rect in rects:
            x, y, w, h = face_utils.rect_to_bb(rect)
            # Double the selection scale to prevent the sticker from being blocked
            x = int(x - w / 2)
            if x < 0:
                x = 0
            y = int(y - h)
            if y < 0:
                y = 0
            w = w * 2
            if x + w > frame_cols:
                w = frame_cols - x
            h = int(h * 2.5)
            if y + h > frame_rows:
                h = frame_rows - y

            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            if tags_index == 1:
                paw_index = (paw_index+1) % 4
                tag_img = cv2.imread('/images'+filters[paw[paw_index]])
            else:
                tag_img = cv2.imread('images/'+filters[tags_index])

            tag_img = cv2.resize(tag_img, (w, h), interpolation=cv2.INTER_CUBIC)
            rows, cols, ch = tag_img.shape

            # cat ears
            if tags_index == 0:
                p1 = np.float32([[cols / 4, rows / 3], [cols / 4 * 3, rows / 3], [cols * 0.4, rows * 0.7],
                                 [cols * 0.6, rows * 0.7]])
                p2 = np.float32([[shape[0][0] - x, shape[0][1] - y], [shape[16][0] - x, shape[16][1] - y],
                                 [shape[6][0] - x, shape[6][1] - y], [shape[10][0] - x, shape[10][1] - y]])
            # cat paws(live)
            elif tags_index == 1:
                p1 = np.float32([[0, rows * 0.41], [cols - 1, rows * 0.41], [0, rows * 0.7], [cols - 1, rows * 0.7]])
                p2 = np.float32([[shape[1][0] - x, shape[1][1] - y], [shape[15][0] - x, shape[15][1] - y],
                                 [shape[1][0] - x, shape[3][1] - y], [shape[15][0] - x, shape[13][1] - y]])
            # hat
            elif tags_index == 2:
                p1 = np.float32([[cols * 0.25, rows * 0.8 ], [cols * 0.75, rows * 0.75 ], [cols / 2, rows - 1]])
                p2 = np.float32([[shape[17][0] - x, shape[17][1] - y], [shape[26][0] - x, shape[26][1] - y],
                                 [shape[29][0] - x, shape[29][1] - y]])
            # cartoon eyes
            elif tags_index == 3:
                p1 = np.float32([[0, rows / 2], [cols - 1, rows / 2], [cols / 2, 0]])
                p2 = np.float32([[shape[17][0] - x, shape[36][1] - y], [shape[26][0] - x, shape[45][1] - y],
                                 [shape[27][0] - x, shape[19][1] - y]])

            # beard
            elif tags_index == 4:
                p1 = np.float32([[cols * 0.41, rows * 0.15],[cols * 0.58, rows * 0.15], [cols * 0.28, rows * 0.59],[cols * 0.72, rows * 0.62]])
                p2 = np.float32([[shape[31][0] - x, shape[31][1] - y],[shape[35][0] - x, shape[35][1] - y],
                                 [shape[6][0] - x, shape[6][1] - y],[shape[10][0] - x, shape[10][1] - y]])

            # Moustache
            elif tags_index == 5:
                p1 = np.float32([[cols * 0.3, rows / 2], [cols * 0.75, rows / 2], [cols / 2, rows / 4]])
                p2 = np.float32([[shape[48][0] - x, shape[48][1] - y], [shape[54][0] - x, shape[54][1] - y],
                                 [shape[33][0] - x, shape[33][1] - y]])

            M = cv2.getAffineTransform(p1, p2)
            tag_img = cv2.warpAffine(tag_img, M, (cols, rows), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
                                     borderValue=(255, 255, 255))

            # cut to prevent out of frame
            if y + rows > frame_rows:
                rows = frame_rows - y
                tag_img = tag_img[0:rows, 0:cols]
            if x + cols > frame_cols:
                cols = frame_cols - x
                tag_img = tag_img[0:rows, 0:cols]

            roi = frame[y:y + rows, x:x + cols]
            # Now create a mask of logo and create its inverse mask also
            tag_img_gray = cv2.cvtColor(tag_img, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(tag_img_gray, 1, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)

            face_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
            tag_fg = cv2.bitwise_and(tag_img, tag_img, mask=mask)

            dst = cv2.add(face_bg, tag_fg)
            frame[y:y + rows, x:x + cols] = dst

        tag_name = ['cat ears', 'cat paws(live)', 'hat',
                    'cartoon eyes', 'beard', 'Moustache']
        cv2.imshow("Face Filter:" + tag_name[tags_index], frame)
        key = cv2.waitKey(1) & 0XFF

        if key == ord("q"):
            break
    cv2.destroyAllWindows()
    vs.stop()


if __name__ == "__main__":
    tags(0)
