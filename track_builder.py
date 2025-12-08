import numpy as np
import cv2

def build_track_from_image(track_img_path: str) -> np.ndarray:
    track_img = cv2.imread(track_img_path, cv2.IMREAD_GRAYSCALE)
    _, track = cv2.threshold(track_img, 200, 1, cv2.THRESH_BINARY_INV)
    scale_factor = 0.5
    height, width = track.shape
    new_size = (int(width * scale_factor), int(height * scale_factor))

    track_small = cv2.resize(track, new_size, interpolation=cv2.INTER_NEAREST)
    return track

if __name__ == "__main__":
    build_track_from_image('./tracks/smooth.jpeg')
