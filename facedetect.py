import cv2

face_cas=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cas=cv2.CascadeClassifier("haarcascade_eye_tree_eyeglasses.xml")

img=cv2.imread("photo.jpg")
gray_im = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)



faces = face_cas.detectMultiScale(gray_im, scaleFactor=1.2, minNeighbors=6 )

eyes = eye_cas.detectMultiScale(gray_im,1.3,5)

for x,y,w,h in faces:
    img=cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),5)
    roi_gray = gray_im[y:y + h, x:x + w]


#eyes = eye_cas.detectMultiScale(roi_gray)

for x, y, w, h in eyes:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

print(type(faces))
print(faces)

resized=cv2.resize(img,(500,500))
cv2.imshow("gray",resized)
cv2.waitKey(0)
cv2.destroyAllWindows()