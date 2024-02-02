import cv2 as cv
import numpy as np

class Window:

    def __init__(self, name, frame=None):
        self.name = name

        # created OpenCV named window
        cv.namedWindow(self.name)

        # initialise window with input frame or test img
        if frame:
            cv.imshow(self.name, frame)
        else:
            # create black image 800x600
            test = np.zeros((600, 800, 3), dtype=np.uint8)

            # Generate a rainbow gradient along the height (600)
            for i in range(600):
                hue = int(180 * i / 600)  # Vary the hue from 0 to 180
                color = list(map(int, cv.cvtColor(np.array([[[hue, 255, 255]]], dtype=np.uint8), cv.COLOR_HSV2BGR)[0, 0]))
                test[i, :, :] = color

            # display rainbow test image
            cv.imshow(self.name, test)

    def update(self, frame):
        cv.imshow(self.name, frame)

    def destroy(self):
        cv.destroyWindow(self.name)
