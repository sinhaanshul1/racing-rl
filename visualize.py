from track_builder import Track
import pygame
import numpy as np
import cv2
import car

class TrackOverview:
    def __init__(self, track_image_path):
        self._running = True

        self._car = car.Car(0.25, 0.5)
        self._track_img = track_image_path
        self._track = Track(track_image_path)

    def run(self) -> None:
        '''Handles main game loop.'''
        pygame.init()
        pygame.display.set_caption('Racing')
        self._resize_surface((len(self._track.track()[0]), len(self._track.track())))
        clock = pygame.time.Clock()
        while self._running:
            clock.tick(30)
            self._handle_events()
            self._redraw()

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
                self._resize_surface(event.size)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                click_x, click_y = event.pos
                height, width = self._track.track().shape
                self._car.set_location(click_x / width, click_y / height)
    
    def _redraw(self):
        self._draw_track()
        self._draw_car()
        # self._car_in_track_checker()
        pygame.display.flip()



    def _draw_car(self):
        surface = pygame.display.get_surface()
        height, width = self._track.track().shape
        car_x, car_y = self._car.location()
        car_x *= width
        car_y *= height
        pygame.draw.circle(surface, pygame.Color(255, 0, 0), (car_x, car_y), car.SIZE * min(height, width), width=0)

    def _draw_track(self):
        surface = pygame.display.get_surface()
        surface.fill(pygame.Color(0, 0, 0))
        track = pygame.image.load(self._track_img)
        track = pygame.transform.scale(track, surface.get_size())
        surface.blit(track, (0, 0))

    def _resize_surface(self, size: tuple[int, int]) -> None:
        width, height = size
        self._track.resize(width, height)
        pygame.display.set_mode(size, pygame.RESIZABLE)
    def _end_game(self):
        self._running = False


if __name__ == "__main__":
    TrackOverview('./assets/tracks/track.jpeg').run()
