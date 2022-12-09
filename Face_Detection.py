import cv2
from random import randrange
#Acquiring some pre-trained data on face

trained_face_data = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')

#choosing a random image for this job.

# img = cv2imread('test.jpg')
webcam = cv2.VideoCapture(0)
# key = cv2.waitKey(1)

# #iterate forever over the frames.
while True:
    #read the current frame.
    successful_frame_read, frame = webcam.read()

    #convert the frame to grayscale.
    grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #draw rectangle around the frame
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_frame)
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h),
                      (randrange(255), randrange(255), randrange(255)), 5)

    cv2.imshow("Converting Frame to grayscale", frame)
    key = cv2.waitKey(1)

    if key == 83 or key == 115:
        break

webcam.release()
