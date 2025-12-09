import numpy as np
import cv2
from car import Car

class Track:
    def __init__(self, track_img_path: str):
        self._track = self._build_track_from_image(track_img_path)
    
    def track(self):
        return self._track
    
    def resize(self, width: int, height: int):
        self._track = cv2.resize(
            self._track,
            (width, height),
            interpolation=cv2.INTER_CUBIC 
        )

    def car_in_track(self, car: Car):
        height, width = self._track.shape
        relative_x, relative_y = car.location()
        try:
            return (self._track[ int(relative_y * height), int(relative_x * width)] == 1)
        except:
            return False


    def _build_track_from_image(self, track_img_path: str) -> np.ndarray:
        track_img = cv2.imread(track_img_path, cv2.IMREAD_GRAYSCALE)
        _, track = cv2.threshold(track_img, 200, 1, cv2.THRESH_BINARY_INV)
        scale_factor = 0.5
        height, width = track.shape
        new_size = (int(width * scale_factor), int(height * scale_factor))

        track_small = cv2.resize(track, new_size, interpolation=cv2.INTER_NEAREST)
        return track


