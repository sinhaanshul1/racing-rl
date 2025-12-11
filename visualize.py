from track_builder import Track
import pygame
import numpy as np
import cv2
import car

class TrackOverview:
    def __init__(self, track_image_path, car: car.Car):
        self._running = True

        self._car = car
        self._track_img = track_image_path
        self._track = Track(track_image_path)
        self._start_coordinate = None
        self._end_coordinate = None
        self._center_line = None
        self._center_lines = None
        self.car_history = self._car.history
        self.car_path = []
          


        

    def run(self) -> None:
        '''Handles main game loop.'''
        pygame.init()
        
        pygame.display.set_caption('Racing')
        pygame.display.set_mode((len(self._track.track()[0]), len(self._track.track())))
        self.car_image = pygame.image.load('./assets/car.png').convert_alpha()
        width, height = self.car_image.get_size()

        # Scale by 50%
        scale_factor = car.SIZE
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        self.car_image = pygame.transform.scale(self.car_image, (new_width, new_height))
        # self._resize_surface((len(self._track.track()[0]), len(self._track.track())))
        clock = pygame.time.Clock()
        # self._draw_track()
        # self._draw_center_line()
        pygame.display.flip()
        while self._running:
            clock.tick(30)
            self._handle_events()
            self._redraw()
            # print(self._track.car_in_track(self._car))

        pygame.quit()

    def _car_in_track_checker(self):
        '''Checks the valid positions in track and plots to see if everything is correct'''
        surface = pygame.display.get_surface()
        width, height = surface.get_size()
        
        for i in range(0, 100):
            for j in range(0, 100):
                # print(i * 0.01, j * 0.01)
                if self._track.car_in_track(car.Car(i * 0.01, j * 0.01)):
                    # print('here')
                    pygame.draw.circle(surface, pygame.Color(0, 0, 255), (i * 0.01 * width, j * 0.01 * height), car.SIZE * min(height, width), width=0)

    def _handle_events(self) -> None:
        '''Handles user input events.'''
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._end_game()
            elif event.type == pygame.VIDEORESIZE:
                print("No not allowed")
                # self._resize_surface(event.size)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                click_x, click_y = event.pos
                height, width = self._track.track().shape
                self._car.set_location(click_x / width, click_y / height)
                if self._start_coordinate is None:
                    print("set start")
                    self._start_coordinate = (click_y, click_x)
                elif self._end_coordinate is None:
                    print("set end")
                    self._end_coordinate = (click_y, click_x)
                    print('Start: ', self._start_coordinate)
                    print('End: ', self._end_coordinate)
                    # self._center_line = self._track.sort_center_line(self._start_coordinate, self._end_coordinate)
                    self._center_line = self._track.sort_center_line(self._start_coordinate, self._end_coordinate)
                    # new_coordinate_start, new_coordinate_end = self._track.sort_center_line(self._start_coordinate, self._end_coordinate)
                    # surface = pygame.display.get_surface()
                    # pygame.draw.circle(surface, pygame.Color(255, 0, 255), new_coordinate_start, 10, width=0)
                    # pygame.draw.circle(surface, pygame.Color(0, 255, 255), new_coordinate_end, 10, width=0)

                    # print('drew start at', new_coordinate_start)
                    # print('drew end at', new_coordinate_end)
                else:
                    self._start_coordinate = None
                    self._end_coordinate = None
                    self._center_line = None
    
    def _redraw(self):
        self._draw_track()
        self._draw_center_line()
        self._draw_car_path()
        self._draw_car()
        self._draw_start_stop()
        # self._car_in_track_checker()
        # self._draw_center_line()
        pygame.display.flip()



    def _draw_car(self):
        # surface = pygame.display.get_surface()
        # height, width = self._track.track().shape
        # car_x, car_y = self._car.location()
        # car_x *= width
        # car_y *= height
        # pygame.draw.circle(surface, pygame.Color(255, 0, 0), (car_x, car_y), car.SIZE * min(height, width), width=0)
        surface = pygame.display.get_surface()
        width, height = surface.get_size()
        # Get car position and angle
        # car_x, car_y = self._car.location()
        if (len(self.car_history ) > 0):
            data = self.car_history.pop(0)
            self.car_path.append(data)
            car_x, car_y, angle = data
            car_x *= width
            car_y *= height
            # angle = self._car.direction()  # in radians
            angle_degrees = np.degrees(angle)  # negative because pygame rotates counter-clockwise
            
            # Rotate the image
            rotated_image = pygame.transform.rotate(self.car_image, angle_degrees)
            
            # Get the rect and center it on the car position
            rect = rotated_image.get_rect(center=(car_x, car_y))
            
            # Draw it
            surface.blit(rotated_image, rect)
        else:
            self._end_game()

    def _draw_car_path(self):
        surface = pygame.display.get_surface()
        width, height = surface.get_size()
        # Get car position and angle
        # car_x, car_y = self._car.location()
        for data in self.car_path:
            car_x, car_y, _ = data
            car_x *= width
            car_y *= height
            
            pygame.draw.circle(surface, pygame.Color(255, 0, 0), (car_x, car_y), 2, width=0)

    def _draw_start_stop(self):
        colors = [pygame.Color("orange"), pygame.Color("purple"), pygame.Color("green"), pygame.Color("yellow")]

        if self._center_line is not None:
            for coordinate in self._center_line:
                y, x = coordinate

                surface = pygame.display.get_surface()
                pygame.draw.circle(surface, colors[0], (x, y), 5, width=0)

    def _draw_center_line(self):
        surface = pygame.display.get_surface()
        height, width = self._track.track().shape
        centerline, width = self._track.center_width_representation()
        for y, x in centerline:
            pygame.draw.circle(surface, pygame.Color(0, 255, 0), (x, y), 1, width=0)

    def _draw_track(self):
        surface = pygame.display.get_surface()
        surface.fill(pygame.Color(0, 0, 0))
        track = pygame.image.load(self._track_img)
        track = pygame.transform.scale(track, surface.get_size())
        surface.blit(track, (0, 0))

    def _resize_surface(self, size: tuple[int, int]) -> None:
        width, height = size
        pygame.display.set_mode(size, pygame.RESIZABLE)

    def _end_game(self):
        self._running = False


if __name__ == "__main__":
    test_car = car.Car(0.25,0.5)
    for _ in range(50):
        test_car.move(1)
    for _ in range(20):
        test_car.move(0)
    for _ in range(50):
        test_car.move(-2)

    TrackOverview('./assets/tracks/test_track.jpeg', test_car).run()
