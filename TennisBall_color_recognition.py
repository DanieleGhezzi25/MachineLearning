import numpy as np
import cv2

'''

    This obviously works if only there are not green or yellow objects in the background.

'''

cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    width = int(cap.get(3))
    height = int(cap.get(4))

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green_and_yellow = np.array([25, 52, 72])
    upper_green_and_yellow = np.array([102, 255, 255])

    mask = cv2.inRange(hsv, lower_green_and_yellow, upper_green_and_yellow)

    countours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in countours:
        area = cv2.contourArea(contour)
        if area > 1500:
            centroids = cv2.moments(contour)
            center_x = int(centroids['m10'] / centroids['m00'])
            center_y = int(centroids['m01'] / centroids['m00'])
            cv2.circle(frame, (center_x, center_y), 4, (0, 255, 0), -1)
            cv2.putText(frame, 'Tennis Ball', (center_x + 10, center_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    mask = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow('Mask', mask)
    cv2.imshow('App', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()