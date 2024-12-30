import mss
import cv2
import numpy as np


def capture_game(game_window):
    """
    Capture frames from the specified game window.
    """
    with mss.mss() as sct:
        while True:
            frame = np.array(sct.grab(game_window))
            yield cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)