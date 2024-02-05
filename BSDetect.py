import cv2 as cv
import numpy as np
from WindowManager import Window

class Detector:

    def __init__(self, canny_threshold_sigma, histequ_step=True, debug_windows=True):
        self.canny_threshold_sigma = canny_threshold_sigma
        self.histequ_step = histequ_step
        self.debug_windows = debug_windows

        if self.debug_windows:
            self.grayscale_window = Window("BSDetector Debug: Grayscale")
            self.gausblur_window = Window("BSDetector Debug: Gaussian Blur")
            self.canny_window = Window("BSDetector Debug: Canny Algorithm")
            self.countour_window = Window("BSDetector Debug: Detected Countours")
            if self.histequ_step:
                self.histequ_window = Window("BSDetector Debug: Histogram Equalisation")

    def _grayscale(self, frame):
        # apply the single-channel conversion with grayscale filter
        grayscale = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # output to the debug window if enabled
        if self.debug_windows:
            self.grayscale_window.update(grayscale)

        # return the grayscaled frame
        return grayscale
    
    def _histequ(self, frame):
        # apply histogram equalisation
        histequ = cv.equalizeHist(frame)

        # output to the debug window if enabled
        if self.debug_windows:
            self.histequ_window.update(histequ)
        
        # return the histogram equalised frame
        return histequ
    
    def _gausblur(self, frame):
        # apply the Gaussian blur
        gausblur = cv.GaussianBlur(frame, (5,5), 0)

        # output to the debug window if enabled
        if self.debug_windows:
            self.gausblur_window.update(gausblur)
        
        # return the Gaussian blurred frame
        return gausblur
    
    def _findContours(self, edges):
        # RETR_EXTERNAL only retrieves the extreme outer countours
        # CHAIN_APPROX_SIMPLE compresses horizontal, vertical, and
        #   diagonal segments and leaves only their end points
        contours, _ = cv.findContours(
            edges,
            cv.RETR_EXTERNAL,       # RetrievalModes
            cv.CHAIN_APPROX_SIMPLE  # ContourApproximationModes
        )
    
        # output to the debug window if enabled
        if self.debug_windows:
            # create a black mask
            mask = np.zeros_like(edges)
            # draw countours white white fill
            cv.drawContours(mask, contours, -1, (255), cv.FILLED)
            # display window
            self.countour_window.update(mask)

        # return the countours
        return contours

    
    def detect(self, input):
        # single channel conversion using grayscaling
        frame = self._grayscale(input)
        
        # apply histogram equalisation to improve contrasts for better Canny
        if self.histequ_step:
            frame = self._histequ(frame)

        # apply Gaussian blur noise reduction and smoothening, prep for Canny
        frame = self._gausblur(frame)

        # compute the median single-channel pixel intensities
        gaus_median = np.median(frame)
        # compute threshold values for canny using single parameter Canny
        lower_threshold = int(max(0, (1.0 - self.canny_threshold_sigma) * gaus_median))
        upper_threshold = int(min(255, (1.0 + self.canny_threshold_sigma) * gaus_median))

        # perform Canny edge detection
        canny = cv.Canny(
            frame,
            lower_threshold,
            upper_threshold
        )

        contours = self._findContours(canny)

        # output to the debug window if enabled
        if self.debug_windows:
            self.canny_window.update(canny)

        # return canny and contours
        return canny, contours
