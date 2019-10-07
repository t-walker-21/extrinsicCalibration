import cv2


class Display(object):
    """
    Helper to display content with OpenCV.

    Config properties (and defaults):
    'name': 'frame',
    'width': 720,
    'height': 480,
    'quit-key': 'q'
    'additional-keys': ['s', 'b']
    """
    # Default configuration
    def_config = {
        'name': 'frame',
        'width': 720,
        'height': 480,
        'quit-key': 'q',
        'additional-keys': ['s', 'b', 'n']
    }

    def __init__(self, config={}, verbose=False):
        # Update default configuration with
        # only those values that have been
        # specified by the user
        self.config = self.def_config.copy()
        self.config.update(config)

        # Store flags for each additional key the
        # display should listen to
        self.add_keys_flags = [False] * len(self.config['additional-keys'])

        # Setup a named window with WINDOW_NORMAL attribute
        # This allows the windows to be resized to custom size
        cv2.namedWindow(self.config['name'], cv2.WINDOW_NORMAL)
        cv2.resizeWindow(
            self.config['name'], self.config['width'], self.config['height'])

        self.quit_key_pressed = False

    def __del__(self):
        cv2.destroyWindow(self.config['name'])

    def refresh(self, image):
        """Refresh the window with new image."""

        # Refresh image in the window
        cv2.imshow(self.config['name'], image)

        # Without call to waitKey, window won't be refreshed.
        # 1 ms doesn't matter. On next refresh the quit
        # key will be captured.
        ret = cv2.waitKey(1) & 0xFF
        if ret is ord(self.config['quit-key']):
            self.quit_key_pressed = True
        else:
            for i, key in enumerate(self.config['additional-keys']):
                if ret is ord(key):
                    self.add_keys_flags[i] = True

    def can_quit(self):
        """Helper to determine if quit key is pressed."""
        return self.quit_key_pressed

    def key_pressed(self, key):
        """Checks if some given key is serviced.

        If the key is serviced, the corresponding 
        flag is made False."""
        if key in self.config['additional-keys']:
            ret = self.add_keys_flags[
                self.config['additional-keys'].index(key)]
            if ret is True:
                self.add_keys_flags[
                    self.config['additional-keys'].index(key)] = False

        return ret
