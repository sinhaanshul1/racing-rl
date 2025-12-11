import numpy as np
import cv2
from car import Car
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
import math

class Track:
    def __init__(self, track_img_path: str):
        self._track = self._build_track_from_image(track_img_path)
        self._center_line, self._width = self.get_centerline_and_width()

    
    def track(self):
        return self._track
    

    
    def center_width_representation(self):
        return self._center_line, self._width
    
    def resize(self, width: int, height: int):
        self._track = cv2.resize(
            self._track,
            (width, height),
            interpolation=cv2.INTER_CUBIC 
        )
        self._center_line, self._width = self.get_centerline_and_width()

    def car_in_track(self, car: Car):
        height, width = self._track.shape
        relative_x, relative_y = car.location()
        try:
            return (self._track[ int(relative_y * height), int(relative_x * width)] == 1)
        except:
            return False


    def get_centerline_and_width(self):
        """
        Given a binary track map (1 = track, 0 = off-track), 
        returns centerline coordinates and width at each point.
        
        Returns:
            x, y: arrays of centerline coordinates
            track_widths: array of width at each centerline point
        """
        # Convert track to uint8 (needed for distanceTransform)
        track_uint8 = (self._track * 255).astype(np.uint8)

        # Distance transform
        dist_map = cv2.distanceTransform(track_uint8, cv2.DIST_L2, 5)
        
        # Skeletonization to find 1-pixel-wide centerline
        skeleton = skeletonize(self._track.astype(bool), method="lee")
        # print(skeleton)
        # print(list(zip(*np.where(skeleton))))
        

        # Width along centerline: distance to nearest wall * 2
        centerline_distances = dist_map[skeleton]
        track_widths = centerline_distances * 2
        # Centerline coordinates
        y, x = np.where(skeleton)  # rows=y, cols=x

        return list(zip(y, x)), track_widths

    



    def _build_track_from_image(self, track_img_path: str) -> np.ndarray:
        track_img = cv2.imread(track_img_path, cv2.IMREAD_GRAYSCALE)
        _, track = cv2.threshold(track_img, 200, 1, cv2.THRESH_BINARY_INV)
        scale_factor = 0.5
        height, width = track.shape
        new_size = (int(width * scale_factor), int(height * scale_factor))

        track_small = cv2.resize(track, new_size, interpolation=cv2.INTER_NEAREST)
        
        return track

    def array_plotter(self, arr: np.ndarray):
        plt.imshow(arr, cmap='Greens', origin='upper')  # 'Greens' colormap for track
        plt.colorbar(label='0=off-track, 1=track')           # Optional color bar
        plt.title('Track Map')
        plt.show()

    def sort_center_line(self, start, end):
        sorted_center_line = []
        start_coordinate = np.array(self._center_line[0])
        distance_to_start = math.inf
        end_coordinate = np.array(self._center_line[0])
        distance_to_end = math.inf
        start = np.array(start)
        end = np.array(end)
        for coordinate in self._center_line:
            if np.linalg.norm(start - np.array(coordinate)) < distance_to_start:
                start_coordinate = np.array(coordinate)
                distance_to_start = np.linalg.norm(start - np.array(coordinate))
            if np.linalg.norm(end - np.array(coordinate)) < distance_to_end:
                end_coordinate = np.array(coordinate)
                distance_to_end = np.linalg.norm(end - np.array(coordinate))
        last_coordinate_looked_at = tuple(start_coordinate)
        found_match = True
        # sorted_center_line = self._find_direct_path(tuple(start_coordinate), tuple(end_coordinate), self._center_line)
        print("Sorted: ", len(sorted_center_line))
        print("Regular: ", len(self._center_line))

        # return sorted_center_line    
        return self._find_direct_path(tuple(start_coordinate), tuple(end_coordinate), self._center_line)
    
    def _find_direct_path(self, start, end, path):
        stack = [(start, [start])]
        all_paths = []

        while stack:
            current, current_path = stack.pop()

            # If we reached the end, store this path
            if current == end:
                print("Found a path")
                return current_path
                continue

            neighbors = self._get_neighbors(current)

            for neighbor in neighbors:
                t = tuple(neighbor)

                # Only follow neighbors that are on the centerline AND not yet in this path
                if t in path and t not in current_path:
                    stack.append((t, current_path.copy() + [t]))
        print("Number of paths found: ", len(all_paths))

        # return all_paths[0]


    
    def _get_neighbors(self, coordinate):
        neighbors = []
        neighbors.append((coordinate[0] + 1, coordinate[1] - 1))
        neighbors.append((coordinate[0] + 0, coordinate[1] - 1))
        neighbors.append((coordinate[0] - 1, coordinate[1] - 1))
        neighbors.append((coordinate[0] - 1, coordinate[1] + 0))
        neighbors.append((coordinate[0] - 1, coordinate[1] + 1))
        neighbors.append((coordinate[0] + 0, coordinate[1] + 1))
        neighbors.append((coordinate[0] + 1, coordinate[1] + 1))
        neighbors.append((coordinate[0] + 1, coordinate[1] + 0))

        return neighbors


if __name__ == "__main__":
    Track('./assets/tracks/track.jpeg').sort_center_line(1, 2)
