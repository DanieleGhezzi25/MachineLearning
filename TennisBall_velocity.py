'''

    This code doesn't use any ML model to detect the tennis ball, but only color and shape recognition.
    The velocity is measured in all the 3 dimension and it is calculated using the distance from the camera (y axis) and the distance
    on the perpendicular plane (xz plane) from the previous position of the ball.

    The code is not perfect but quite accurately and it is a good starting point to understand how to manipulate computer vision.

'''

import numpy as np
import cv2

cap = cv2.VideoCapture(0)
fps = cap.get(cv2.CAP_PROP_FPS)

radius = 3.8 # real tennis ball radius in cm
focal_length = 2.95 * 1280 / 6.35 # focal length of the camera in pixels

# this is a weird way to calculate the focal length, but it is acceptable.
# in particular, the focal length is 2.95 mm, the width of the image is 1280 pixels (HD resolution) and the dimension of the camera is 6.35 mm.
# this values are acceptable for a standard webcam but, because I don't know mine, I've calibrated the distances detected with a ruler.

previous_distance_from_screen = 0
previous_center_distance = 0
previous_center_x = 0
previous_center_y = 0

velocities = []
times = []
count_time = 0

while(True):
    ret, frame = cap.read()
    width = int(cap.get(3))
    height = int(cap.get(4))

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green_and_yellow = np.array([25, 52, 72])
    upper_green_and_yellow = np.array([102, 255, 255])

    mask = cv2.inRange(hsv, lower_green_and_yellow, upper_green_and_yellow)

    countours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # NOTE: in a x-y-z coordinate system with origin in the camera and y axis in observer direction,
    # I will use the distance from the screen as a y coordinate and the ball-center distance as a x and z coordinate.

    if len(countours) > 0:
        for contour in countours:
            area = cv2.contourArea(contour)
            if area > 1000:

                # center and radius detection
                (center_x, center_y), pixel_radius = cv2.minEnclosingCircle(contour)
                center_x = int(center_x)
                center_y = int(center_y)
                radius = int(radius)

                pixel_per_cm = pixel_radius / radius

                distance_y = (radius * focal_length) / pixel_radius # distance in cm, that's an accurate way to calculate the distance from the screen
                distance_xz_pix = np.sqrt((center_x - previous_center_x)**2 + (center_y - previous_center_y)**2) # distance in pixels
                distance_xz = distance_xz_pix / pixel_per_cm # distance in cm, little suspicious about how it works

                distance_from_prev_y = np.abs(distance_y - previous_distance_from_screen)
                distance_from_prev_xz = np.abs(distance_xz - previous_center_distance)

                distance = np.sqrt(distance_from_prev_y**2 + distance_from_prev_xz**2)
                velocity = distance / (1/fps) / 100 # velocity in m/s
                velocities.append(velocity)
                count_time += 1/fps
                times.append(count_time)

                cv2.circle(frame, (center_x, center_y), radius, (0, 255, 0), 1)

                previous_center_x = center_x
                previous_center_y = center_y
                previous_distance_from_screen = distance_y
                previous_center_distance = distance_xz
                text = f'Velocity: {velocity:.2f} m/s'
                cv2.putText(frame, text, (width - 500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    mask = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow('Mask', mask)
    cv2.imshow('App', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# plot the velocities
import matplotlib.pyplot as plt
plt.plot(times, velocities)
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.title('Velocity of the tennis ball')
plt.show()