import cv2


class Video(object):
    """
    Helper to wrangle with OpenCV Video class.

    Note, first frame of video is gobbled up
    when directly using next_frame.
    """

    def __init__(self, file_name, verbose=False):
        self.file_name = file_name
        self.cap = cv2.VideoCapture(self.file_name)
        self.verbose = verbose

        if self.verbose:
            if self.is_opened():
                print('Opened {} with following attributes'.format(self.file_name))
                print('Width: {}, Height: {}, FPS: {}'.format(
                    self.cap.get(3), self.cap.get(4), self.cap.get(5)))
            else:
                print('Could not open {}'.format(self.file_name))

        # Set default values
        self.last_queried_frame_num = 0
        self.frame = 0
        self.cap_ret_val = False
        self.num_frames = 0

        # If video file open-able, test
        # out with one frame
        if self.is_opened():
            self.frame = self.next_frame()
            self.cap_ret_val = not self.end_reached()
            self.num_frames = int(self.cap.get(7))            

    def is_opened(self):
        """Helper to find if file is opened or open-able."""
        return self.cap.isOpened()

    def next_frame(self):
        """Grab the next frame from the video."""
        self.cap_ret_val, self.frame = self.cap.read()
        if self.verbose and not self.cap_ret_val:
            print('Reached end of {}'.format(self.file_name))
        return self.frame

    def end_reached(self):
        """To check if end of video reached."""
        return not self.cap_ret_val

    def get_num_frames(self):
        """Getter for number of frames in video."""
        return self.num_frames

    def get_cur_frame_num(self):
        """Get current frame number."""
        return int(self.cap.get(1))

    def get_frame(self, frame_num):
        """
        Get any random frame in the video. 
        If frame num out of bounds, return empty frame.
        """
        if frame_num >= 0 or frame_num < self.num_frames:
            # Take a shortcut if query is sequential
            # Else go the usual way
            if frame_num != self.last_queried_frame_num + 1:
                self.cap.set(1, frame_num)

        self.last_queried_frame_num = frame_num
        return self.next_frame()

    def __del__(self):
        self.cap.release()
