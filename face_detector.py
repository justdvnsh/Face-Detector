import cv2
import glob

images = glob.glob("images/*.jpg")

for image in images:
		
	face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

	img = cv2.imread(image)

	gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.03, minNeighbors=5)

	for x,y,w,h in faces:
		img = cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 3)
		
	resized = cv2.resize(img, (800, 600))

	cv2.imshow("GRAY", resized)
	cv2.waitKey(0)
	cv2.destroyAllWindows()