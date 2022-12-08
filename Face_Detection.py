import cv2 as uwu

#Acquiring some pre-trained data on face
trained_face_data = uwu.CascadeClassifier(
    'haarcascade_frontalface_default.xml')

#choosing a random image for this job.
img = uwu.imread('test.jpg')

#greyscaling the image

grayscaled_img = uwu.cvtColor(img, uwu.COLOR_BGR2GRAY)
# uwu.imshow('Grayscaled img', grayscaled_img)
# uwu.waitKey()

#detecting faces
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
for (x, y, w, h) in face_coordinates:
    uwu.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
print(face_coordinates)

#showing the image
uwu.imshow('Output image', img)
uwu.waitKey()

print("Created")
