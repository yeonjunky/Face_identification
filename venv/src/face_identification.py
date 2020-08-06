import face_recognition
import cv2
import numpy as np

video_capture = cv2.VideoCapture(0)

# Load a sample picture and learn how to recognize it.
known_image = face_recognition.load_image_file("img\\image1.jpg")
known_face_encoding = face_recognition.face_encodings(known_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    known_face_encoding
]
known_face_names = [
    "yeonjun"
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

font = cv2.FONT_HERSHEY_DUPLEX
is_open = False

while True:
    ret, frame = video_capture.read()
    frame = cv2.flip(frame, 1)

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []

        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                if is_open == False:
                    print("unlocked!")
                    is_open = True

            face_names.append(name)

    process_this_frame = not process_this_frame


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), 1)
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (0, 0, 0), 1)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.waitKey(1) & 0xff == ord('s'):
        is_open = False

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()