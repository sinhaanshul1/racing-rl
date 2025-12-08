from track_builder import build_track_from_image
import pygame
import numpy as np
import cv2
import car

class TrackOverview():
    def __init__(self, track_image_path):
        self._running = True

        self._car = car.Car(0.25, 0.5)
        self._track = build_track_from_image(track_image_path)

    def run(self) -> None:
        '''Handles main game loop.'''
        pygame.init()
        pygame.display.set_caption('Racing')
        self._resize_surface((len(self._track[0]), len(self._track)))
        clock = pygame.time.Clock()
        while self._running:
            clock.tick(30)
            self._handle_events()
            self._redraw()

        pygame.quit()
    def _handle_events(self) -> None:
        '''Handles user input events.'''
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._end_game()
            elif event.type == pygame.VIDEORESIZE:
                self._resize_surface(event.size)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                click_x, click_y = event.pos
                height, width = self._track.shape
                self._car.set_location(click_x / width, click_y / height)
    
    def _redraw(self):
        self._draw_track()
        self._draw_car()
        pygame.display.flip()



    def _draw_car(self):
        surface = pygame.display.get_surface()
        height, width = self._track.shape
        car_x, car_y = self._car.location()
        car_x *= width
        car_y *= height
        pygame.draw.circle(surface, pygame.Color(255, 0, 0), (car_x, car_y), car.SIZE * min(height, width), width=0)
    def _draw_track(self):
        surface = pygame.display.get_surface()
        surface.fill(pygame.Color(0, 0, 0))
        
        self._track = cv2.resize(
            self._track,
            (pygame.display.get_window_size()[0], pygame.display.get_window_size()[1]),
            interpolation=cv2.INTER_CUBIC 
        )


        light_green = np.array([189, 224, 128], dtype=np.uint8)    # dark green
        track_color = np.array([117, 117, 117], dtype=np.uint8) # gray

        track_rgb = np.zeros((self._track.shape[0], self._track.shape[1], 3), dtype=np.uint8)

        track_rgb[self._track == 0] = light_green
        track_rgb[self._track == 1] = track_color
        track_surface = pygame.surfarray.make_surface(track_rgb.swapaxes(0, 1))
        surface.blit(track_surface, (0, 0))

    def _resize_surface(self, size: tuple[int, int]) -> None:
        pygame.display.set_mode(size, pygame.RESIZABLE)
    def _end_game(self):
        self._running = False


if __name__ == "__main__":
    TrackOverview('./assets/tracks/smooth.jpeg').run()
