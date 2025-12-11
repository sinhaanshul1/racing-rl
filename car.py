import math

SIZE = 0.19
STEP_SIZE = 0.001

class Car():
    def __init__(self, start_x, start_y, start_direction=0):
        self.x = start_x
        self.y = start_y
        # Measured in radians with 0 as directly right
        self.theta = 1
        self.history = [(start_x, start_y, start_direction)]

    def location(self):
        return self.x, self.y
    
    def set_location(self, new_x, new_y):
        self.x = new_x
        self.y = new_y

    def direction(self):
        return self.theta

    def move(self, direction):
        self.theta = direction
        self.x += STEP_SIZE * math.cos(direction)
        self.y -= STEP_SIZE * math.sin(direction)
        self.history.append((self.x, self.y, direction))
    def history(self):
        return self.history