import cv2 as cv
import time
import numpy as np

from VideoStream import Stream
from ConsoleLogger import CDLogger

VIDEO_CAPTURE = "./Arran_seabed.mp4"

# Initialise the video capture
stream = Stream(VIDEO_CAPTURE)

# Initialise console logger
console = CDLogger()

# Initialise windows for output
cv.namedWindow("Input Stream")

# Initialise array to store time to process each frame (FPT)
frame_processing_time = []

# Capture the first frame of capture stream
success, frame = stream.read()

# Track if feed is paused by user
paused = True       # pause on first frame

# Continuously capture video feed
while success:

    # Get an input key stroke
    key = cv.waitKey(1)

    # Exit if 'e' key pressed
    if key == ord('e'):
        break
    
    # Pause if 'p' key pressed
    if key == ord('p'):
        paused = True

    # Record the start timestamp of frame processing iteration
    frame_processing_start = time.time()

    # Display original feed to window
    cv.imshow("Input Stream", frame)

    # Record the end timestamp of frame processing iteration
    frame_processing_end = time.time()

    # Calculate and append FPT to array
    frame_processing_time.append(
        frame_processing_end - frame_processing_start
    )

    # Pause video stream logic
    if paused:
        while True:
            # get key presses while paused
            key = cv.waitKey(1)
            # resume with 'p' press
            if key == ord('p'):
                paused = False
                break

    # Console logging metrics
    nframes = len(frame_processing_time)
    afpt = np.mean(frame_processing_time)
    tfpt = stream.target_fpt
    afps = 1 / afpt
    tfps = stream.target_fps
    console.display(nframes, afpt, tfpt, afps, tfps)

    # Capture the next frame from the input stream
    success, frame = stream.read()

# Destroy windows and release video capture for clean exit
cv.destroyAllWindows()
stream.exit()
