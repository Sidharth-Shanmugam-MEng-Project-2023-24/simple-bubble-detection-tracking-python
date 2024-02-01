import cv2 as cv
from queue import Queue
from threading import Thread

class Stream:

    def __init__(self, source):
        # initialise the OpenCV stream
        self.capture = cv.VideoCapture(source)
        
        # check if capture is accessible
        if not self.capture.isOpened():
            raise Exception("Cannot open video stream!")
        
        # calculate FPS and FPT of the capture
        self.target_fps = self.capture.get(cv.CAP_PROP_FPS)
        self.target_fpt = (1 / self.target_fps) * 1000

    def read(self):
        success, frame = self.capture.read()
        return success, frame
    
    def exit(self):
        self.capture.release()





class TStream:

    def __init__(self, source, queueSize=4096):
        # initialise the OpenCV stream
        self.capture = cv.VideoCapture(source)
        # initialise parameter that stops video stream
        self.stopped = False
        # initialise the queue for pushing frames
        self.queue = Queue(maxsize=queueSize)

    def start(self):
        # start a thread to read frames from stream
        t = Thread(
            target=self._update,
            args=()
        )
        # allow thread to be killed when main app exits
        t.daemon = True
        # start the thread
        t.start()
        return self
    
    def _update(self):
        while True:
            # stop reading if stream is stopped
            if self.stopped:
                return
            
            # otherwise, keep reading and queuing until queue is full
            if not self.queue.full():
                # read the next frame from the file
                success, frame = self.capture.read()

                # check if we have reached the end of video stream
                if not success:
                    self.stop()
                    return
                
                # push the frame to the queue
                self.queue.put(frame)

    def read(self):
        # return a frame from the queue
        return self.queue.get()
    
    def stop(self):
        # indicate that thread should be stopped
        self.stopped = True
        self.capture.release()

    def empty(self):
        # returns True if queue is empty
        if self.queue.qsize():
            return True
        return False
