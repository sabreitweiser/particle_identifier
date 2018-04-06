# import the necessary packages
import numpy as np
import argparse
import imutils
import cv2


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
ap.add_argument("-c", "--camera", help="Camera id number")
args = vars(ap.parse_args())

# define the lower and upper boundaries of the colors in HSV
# also define colors in BGR
min_sat = 100
max_sat = 255

min_val = 50
max_val = 255


green_tol = 30
green = (0, 255, 0)
greenLower = (60 - green_tol, min_sat, min_val)
greenUpper = (60 + green_tol, max_sat, max_val)

red_tol = 10
red = (0, 0, 255)
redLower = (-red_tol, min_sat, min_val)
redUpper = (red_tol, max_sat, max_val)

blue_tol = 20
blue = (255, 0, 0)
blueLower = (120 - blue_tol, min_sat, min_val)
blueUpper = (120 + blue_tol, max_sat, max_val)

# define the particles
particles = {(1, 1, 1): "Proton/Neutron",
             (2, 0, 0): "Kaon"}

def find_balls(frame, hsv, lower, upper, color):
        # construct a mask for the color "green", then perform
        # a series of dilations and erosions to remove any small
        # blobs left in the mask
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        # find contours in the mask and initialize the current
        # (x, y) center of the ball
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)[-2]
        center = None

        # only proceed if at least one contour was found
        if len(cnts) > 0:
                # find the largest contour in the mask, then use
                # it to compute the minimum enclosing circle and
                # centroid
                c = max(cnts, key=cv2.contourArea)
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                M = cv2.moments(c)
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

                # only proceed if the radius meets a minimum size
                if radius > 10:
                        # draw the circle and centroid on the frame,
                        # then update the list of tracked points
                        cv2.circle(frame, (int(x), int(y)), int(radius), color, 2)
                        return 1
        return 0

# if a video path was not supplied, grab the reference
# to the webcam
if not args.get("video", False):
	if not args.get("camera", False):
		camera = cv2.VideoCapture(0)
	else:
		camera = cv2.VideoCapture(args[camera])
 
# otherwise, grab a reference to the video file
else:
	camera = cv2.VideoCapture(args["video"])

# keep looping
while True:
	# grab the current frame
	(grabbed, frame) = camera.read()
 
	# if we are viewing a video and we did not grab a frame,
	# then we have reached the end of the video
	if args.get("video") and not grabbed:
		break
 
	# resize the frame, blur it, and convert it to the HSV
	# color space
	frame = imutils.resize(frame, width=600)
	blurred = cv2.GaussianBlur(frame, (15, 15), 0)
	hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
 
	
	green_balls = find_balls(frame, hsv, greenLower, greenUpper, green)
	blue_balls = find_balls(frame, hsv, blueLower, blueUpper, blue)
	red_balls = find_balls(frame, hsv, redLower, redUpper, red)
	particle = (green_balls, blue_balls, red_balls)
	if particle in particles:
		text = particles[particle]
		color = green
	else:
		text = "No known particle found!"
		color = red

	cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

	# show the frame to our screen
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# if the 'q' key is pressed, stop the loop
	if key == ord("q"):
		break
 
# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()

