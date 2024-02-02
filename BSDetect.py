import cv2 as cv
import numpy as np

class Detector:

    def __init__(self, canny_threshold_sigma, histequ_step=True, debug_windows=True):
        self.canny_threshold_sigma = canny_threshold_sigma
        self.histequ_step = histequ_step
        self.debug_windows = debug_windows

        if self.debug_windows:
            self.grayscale_window = cv.namedWindow("BSDetector Debug: Grayscale")
            self.gausblur_window = cv.namedWindow("BSDetector Debug: Gaussian Blur")
            self.canny_window = cv.namedWindow("BSDetector Debug: Canny Algorithm")
            if self.histequ_step:
                self.histequ_window = cv.namedWindow("BSDetector Debug: Histogram Equalisation")

    def _grayscale(self, frame):
        # apply the single-channel conversion with grayscale filter
        grayscale = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # output to the debug window if enabled
        if self.debug_windows:
            cv.imshow("BSDetector Debug: Grayscale", grayscale)

        # return the grayscaled frame
        return grayscale
    
    def _histequ(self, frame):
        # apply histogram equalisation
        histequ = cv.equalizeHist(frame)

        # output to the debug window if enabled
        if self.debug_windows:
            cv.imshow("BSDetector Debug: Histogram Equalisation", histequ)
        
        # return the histogram equalised frame
        return histequ
    
    def _gausblur(self, frame):
        # apply the Gaussian blur
        gausblur = cv.GaussianBlur(frame, (5,5), 0)

        # output to the debug window if enabled
        if self.debug_windows:
            cv.imshow("BSDetector Debug: Gaussian Blur", gausblur)
        
        # return the Gaussian blurred frame
        return gausblur
    
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

        # output to the debug window if enabled
        if self.debug_windows:
            cv.imshow("BSDetector Debug: Canny Algorithm", canny)

        return canny
