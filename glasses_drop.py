import dlib
import cv2
import numpy as np
from PIL import Image
from imutils.video import VideoStream
from imutils import face_utils, resize


def glasses_drop():
    vs = VideoStream().start()

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    max_width = 500
    frame = vs.read()
    frame = resize(frame, width=max_width)

    fps = vs.stream.get(cv2.CAP_PROP_FPS)  # need this for animating proper duration

    animation_length = fps * 5
    current_animation = 0
    glasses_on = fps * 3

    deal = Image.open('images/glasses.png')
    dealing = True

    while True:
        frame = vs.read()

        frame = resize(frame, width=max_width)

        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = []

        rects = detector(img_gray, 0)
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        for rect in rects:
            face = {}
            shades_width = rect.right() - rect.left()

            # predictor used to detect orientation in place where current face is
            shape = predictor(img_gray, rect)
            shape = face_utils.shape_to_np(shape)

            # grab the outlines of each eye from the input image
            left_eye = shape[36:42]
            right_eye = shape[42:48]

            # compute the center of mass for each eye
            left_eye_center = left_eye.mean(axis=0).astype("int")
            right_eye_center = right_eye.mean(axis=0).astype("int")

            # compute the angle between the eye centroids
            dY = left_eye_center[1] - right_eye_center[1]
            dX = left_eye_center[0] - right_eye_center[0]
            angle = np.rad2deg(np.arctan2(dY, dX))

            current_deal = deal.resize((shades_width, int(shades_width * deal.size[1] / deal.size[0])),
                                       resample=Image.LANCZOS)
            current_deal = current_deal.rotate(angle, expand=True)
            current_deal = current_deal.transpose(Image.FLIP_TOP_BOTTOM)

            face['glasses_image'] = current_deal
            left_eye_x = leftEye[0, 0] - shades_width // 4
            left_eye_y = leftEye[0, 1] - shades_width // 6
            face['final_pos'] = (left_eye_x, left_eye_y)

            # I got lazy, didn't want to bother with transparent pngs in opencv
            # this is probably slower than it should be
            if dealing:
                if current_animation < glasses_on:
                    current_y = int(current_animation / glasses_on * left_eye_y)
                    img.paste(current_deal, (left_eye_x, current_y), current_deal)
                else:
                    img.paste(current_deal, (left_eye_x, left_eye_y), current_deal)

        if dealing:
            current_animation += 1
            # uncomment below to save pngs for creating gifs, videos
            # img.save("%05d.png" % current_animation)
            if current_animation > animation_length:
                dealing = False
                current_animation = 0
            else:
                frame = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

        cv2.imshow("deal generator", frame)
        # print(dealing)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        if key == ord("d"):
            dealing = not dealing

    cv2.destroyAllWindows()
    vs.stop()

if __name__ == "__main__":
    glasses_drop()
