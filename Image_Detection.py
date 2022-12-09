import cv2
from random import randrange

trained_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
img = cv2.imread('test.jpg')
cv2.waitKey(0)

# greyscaling the image
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# detecting faces
face_coordinates = trained_data.detectMultiScale(grayscaled_img)

for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x, y), (x + w, y + h),
                  (randrange(255), randrange(255), randrange(255)), 5)

# showing the image
cv2.imshow("Showing final output", img)
