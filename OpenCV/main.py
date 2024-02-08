import cv2
import numpy as np

# cap = cv2.VideoCapture(0)

# while True:
#     face_cascade_db = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
#     eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_eye.xml")

#     success, img = cap.read()
#     img = cv2.flip(img, 1)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     faces = face_cascade_db.detectMultiScale(gray, 1.1, 19)
#     for (x, y, w, h) in faces:
#         cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
#         gray_face = gray[y: y + h, x: x + w]
#         eyes = eye_cascade.detectMultiScale(gray_face, 1.1, 19)
#         for (ex, ey, ew, eh) in eyes:
#             cv2.rectangle(img, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (255, 0, 0), 2)

#     cv2.imshow("Result", img)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
    
# cap.release()
# cv2.destroyAllWindows()








# cap = cv2.VideoCapture(0)

# while True:
#     success, img = cap.read()

#     img = cv2.flip(img, 1)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     faces = cv2.CascadeClassifier('faces.xml')

#     results = faces.detectMultiScale(gray, scaleFactor=2.8, minNeighbors=10)

#     for (x, y, w, h) in results:
#         cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), thickness=3)

#     cv2.imshow('Result', img)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break






# img = cv2.imread('images/Avengers.jpg')
# img = cv2.resize(img, (img.shape[1] // 4, img.shape[0] // 4))
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# faces = cv2.CascadeClassifier('faces.xml')

# results = faces.detectMultiScale(gray, scaleFactor=1.06, minNeighbors=4)

# for (x, y, w, h) in results:
#     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), thickness=3)

# cv2.imshow('Result', img)
# cv2.waitKey(0)









# photo = cv2.imread("images/Avengers.jpg")
# photo = cv2.resize(photo, (photo.shape[1] // 4, photo.shape[0] // 4))
# img = np.zeros(photo.shape[:2], dtype='uint8')

# circle = cv2.circle(img.copy(), (200, 300), 120, 255, -1)
# square = cv2.rectangle(img.copy(), (25, 25), (250, 350), 255, -1)

# # img = cv2.bitwise_and(circle, square)
# # img = cv2.bitwise_or(circle, square)
# # img = cv2.bitwise_xor(circle, square)
# # img = cv2.bitwise_not(square)
# img = cv2.bitwise_and(photo, photo, mask=square)

# cv2.imshow("Result", img)
# cv2.waitKey(0)








# img = cv2.imread("images/Avengers.jpg")
# img = cv2.resize(img, (img.shape[1] // 4, img.shape[0] // 4))

# img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

# # img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)

# img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)

# r, g, b = cv2.split(img)

# img = cv2.merge([b, g, r])

# cv2.imshow("Result", img)
# cv2.waitKey(0)





# img = cv2.imread("images/Avengers.jpg")

# img = cv2.resize(img, (img.shape[1] // 4, img.shape[0] // 4))
# new_img = np.zeros(img.shape, dtype='uint8')

# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img = cv2.GaussianBlur(img, (5, 5), 0)

# img = cv2.Canny(img, 100, 140)

# con, hir = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

# cv2.drawContours(new_img, con, -1, (16, 55, 135), 1)

# cv2.imshow("Result", new_img)
# cv2.waitKey(0)





# img = cv2.imread("images/Avengers.jpg")

# img = cv2.resize(img, (img.shape[1] // 4, img.shape[0] // 4))
# # img = cv2.flip(img, 0)


# def rotate(img_param, angle):
#     height, width = img_param.shape[:2]
#     point = (width // 2, height // 2)

#     mat = cv2.getRotationMatrix2D(point, angle, 1)
#     return cv2.warpAffine(img_param, mat, (width, height))


# # img = rotate(img, 90)

# def transform(img_param, x, y):
#     mat = np.float32([[1, 0, x], [0, 1, y]])
#     return cv2.warpAffine(img_param, mat, (img_param.shape[1], img_param.shape[0]))


# img = transform(img, 30, 200)

# cv2.imshow("Result", img)

# cv2.waitKey(0)













# photo = np.zeros((450, 450, 3), dtype='uint8')


# # photo[10:150, 200:280] = 7, 216, 154

# cv2.rectangle(photo, (350, 50), (400, 100), (7, 216, 154), thickness=cv2.FILLED)

# cv2.line(photo, (0, photo.shape[0] // 3), ( photo.shape[0], photo.shape[1] // 3), (7, 216, 154), thickness=3)

# cv2.circle(photo, (photo.shape[1] // 2, photo.shape[0] // 2), 100, (7, 216, 154), thickness=cv2.FILLED)

# cv2.putText(photo, 'INSTA', (25, 160), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 0), 2)

# cv2.imshow('Result', photo)
# cv2.waitKey(0)







# cap = cv2.VideoCapture(0)

# while True:
#     success, img = cap.read()

#     img = cv2.flip(img, 1)
#     cv2.imshow('Result', img)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break