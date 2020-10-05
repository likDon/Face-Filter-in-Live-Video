from scipy.spatial import distance as dist
import cv2
from imutils.video import VideoStream
from imutils import face_utils, translate, resize
import numpy as np
import time
import dlib


def star():
    # compute the eyes aspect ratio
    def eye_aspect_ratio(eye):
        # take the average of height1 2
        height1 = dist.euclidean(eye[1], eye[5])
        height2 = dist.euclidean(eye[2], eye[4])
        width = dist.euclidean(eye[0], eye[3])
        ear = (height1 + height1) / (2.0 * width)
        return ear


    # eye
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    # eye threshold
    eye_ar_thresh = 0.2  # eye close threshold
    eye_frames_num_thresh = 1  # consecutive frames number of eye close, thought as 1 blink
    eye_blink_num_thresh = 3  # blink for 3 times, display twinkling stars
    blink_interval_thresh = 40  # interval between 2 blinks must less than 40, if not, recount blink num
    star_show_time = 20  # stars showing for 20 frames each time

    # initialize the frame counters and the total number of blinks
    frame_num = 0  # frame number of eye close
    blink_num = 0  # total number of blinks
    show_stars = False

    frame_index = 0
    vs = VideoStream().start()
    time.sleep(1.5)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    # image of each frame in one loop
    filters = ['shiny01.png', 'shiny02.png', 'shiny03.png', 'shiny02.png']
    blink_index = 0  #
    while True:
        frame = vs.read()
        frame = resize(frame, width=1000)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)
        frame_rows, frame_cols, frame_ch = frame.shape
        frame_index += 1  # record frames number
        # tips sentence
        cv2.putText(frame, "try blinking~", (360, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (160,100,100), 4)
        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            if not show_stars:
                # compute the eyes aspect ratio
                left_eye = shape[lStart:lEnd]
                right_eye = shape[rStart:rEnd]
                left_ear = eye_aspect_ratio(left_eye)
                right_ear = eye_aspect_ratio(right_eye)
                # average aspect ratio of two eyes
                ear = (left_ear + right_ear) / 2.0

                # when ratio is less than eye_ar_thresh, add frame number of eye close(frame_num)
                # when frame_num is more than eye_frames_num_thresh, considering blink once
                # if consecutively blink for 3 times, show stars
                if ear < eye_ar_thresh:
                    frame_num += 1
                else:
                    if frame_num >= eye_frames_num_thresh:
                        blink_num += 1
                        frame_index = 0
                        if blink_num >= eye_blink_num_thresh:
                            show_stars = True
                            blink_num = 0
                    frame_num = 0

                # If the blink interval is too long, recount
                if frame_index > blink_interval_thresh:
                    blink_num = 0
            else:
                # showing stars for 20 frames
                if frame_index > star_show_time:
                    show_stars = False
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

                blink_index=(blink_index+1)%4
                tag_img = cv2.imread('images/'+filters[blink_index])
                tag_img = cv2.resize(tag_img, (w, h), interpolation=cv2.INTER_CUBIC)
                rows, cols, ch = tag_img.shape

                p1 = np.float32([[cols/4, rows/2 ], [cols/4*3, rows/2], [cols/2, rows-1]])
                p2 = np.float32([[shape[0][0] - x, shape[0][1] - y], [shape[16][0] - x, shape[16][1] - y],
                                 [shape[33][0] - x, shape[33][1] - y]])
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
                ret, mask = cv2.threshold(tag_img_gray, 1, 255, cv2.THRESH_BINARY)
                mask_inv = cv2.bitwise_not(mask)

                face_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
                tag_fg = cv2.bitwise_and(tag_img, tag_img, mask=mask)

                dst = cv2.add(face_bg, tag_fg)
                frame[y:y + rows, x:x + cols] = dst

        cv2.imshow("My face", frame)
        key = cv2.waitKey(1) & 0XFF

        if key == ord("q"):
            break
    cv2.destroyAllWindows()
    vs.stop()


if __name__ == "__main__":
    star()
