import cv2
import numpy

def get_face_img():
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    cnt = 1
    FILE_PATH = 'C:\\Users\\user\\PycharmProjects\\FaceDetection\\venv\\src\\img\\image{}.jpg'.format(cnt)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')

    while True:
        ret, frame = capture.read()

        frame = cv2.flip(frame, 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in rects:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)
            img_crop = frame[y:y + h, x:x + w]
            cv2.imshow('cropped_image', img_crop)
            # cv2.imwrite(FILE_PATH, img_crop)
            # cnt += 1

        cv2.imshow("frame", frame)

        wait_key = cv2.waitKey(1) &0xff

        if wait_key == ord('k'):
            cv2.imwrite(FILE_PATH, img_crop)
            cnt += 1
        if wait_key == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()



get_face_img()
