from scipy.spatial import distance as dist
import cv2
from imutils.video import VideoStream
from imutils import face_utils, translate, resize
import numpy as np
import time
import dlib


def tongue():
    # compute the mouth aspect ratio
    def mouth_aspect_ratio(mouth):
        # take the average of height1 2 3
        height1 = dist.euclidean(mouth[1], mouth[7])
        height2 = dist.euclidean(mouth[2], mouth[6])
        height3 = dist.euclidean(mouth[3], mouth[5])
        width = dist.euclidean(mouth[0], mouth[4])
        mar = (height1 + height2 + height3) / (3.0 * width)
        return mar

    mouth_ar_thresh = 0.3  # threshold of mouth aspect ratio

    frame_index = 0
    vs = VideoStream().start()
    time.sleep(1.5)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    filters = ['tongue01.png', 'tongue02.png',
               'tongue03.png', 'tongue04.png',
               'tongue03.png', 'tongue02.png',
               'tongue01.png']  # image of each frame in one loop
    mouth_index = 0
    while True:
        frame = vs.read()
        frame = resize(frame, width=1000)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)
        frame_rows, frame_cols, frame_ch = frame.shape
        frame_index += 1  # record frames number
        # tips sentence
        cv2.putText(frame, "try to open your mouth~", (270, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (160,100,100), 4)
        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            # compute the mouth aspect ratio
            mar = mouth_aspect_ratio(shape[60:68])
            # if aspect ratio exceeds threshold
            # mouth is open
            if mar > mouth_ar_thresh:
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

                # compute the image index
                mouth_index = (mouth_index+1)%7
                tag_img = cv2.imread('images/'+filters[mouth_index])
                tag_img = cv2.resize(tag_img, (w, h), interpolation=cv2.INTER_CUBIC)
                rows, cols, ch = tag_img.shape

                p1 = np.float32([[cols*0.3, rows*0.33 ], [cols*0.7, rows*0.33], [cols/2, rows/2]])
                p2 = np.float32([[shape[60][0] - x, shape[60][1] - y], [shape[64][0] - x, shape[64][1] - y],
                                 [shape[57][0] - x, shape[57][1] - y]])
                M = cv2.getAffineTransform(p1, p2)
                tag_img = cv2.warpAffine(tag_img, M, (cols, rows), flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_REPLICATE, borderValue=(255, 255, 255))
                # cut to prevent out of frame
                if y + rows > frame_rows:
                    rows = frame_rows - y
                    tag_img = tag_img[0:rows, 0:cols]
                if x + cols > frame_cols:
                    cols = frame_cols - x
                    tag_img = tag_img[0:rows, 0:cols]

                roi = frame[y:y + rows, x:x + cols]
                tag_img_gray = cv2.cvtColor(tag_img, cv2.COLOR_BGR2GRAY)
                ret, mask = cv2.threshold(tag_img_gray, 1, 255, cv2.THRESH_BINARY)  # mask bound 254 or 1
                mask_inv = cv2.bitwise_not(mask)

                face_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
                tag_fg = cv2.bitwise_and(tag_img, tag_img, mask=mask)

                dst = cv2.add(face_bg, tag_fg)
                frame[y:y + rows, x:x + cols] = dst

        cv2.imshow("tongue", frame)
        key = cv2.waitKey(1)

        if key == ord("q"):
            break
    cv2.destroyAllWindows()
    vs.stop()


if __name__ == "__main__":
    tongue()
