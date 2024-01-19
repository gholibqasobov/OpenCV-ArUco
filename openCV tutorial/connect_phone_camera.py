import cv2

cap = cv2.VideoCapture('http://192.168.84.174:8080/video')
# cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, img = cap.read()
    resized = cv2.resize(img, (600, 400))
    cv2.imshow('Frame', resized)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
