import cv2

cap = cv2.VideoCapture(0)

img = cap.read()[1]

cv2.imshow("current view", img)

cv2.waitKey(0)