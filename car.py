SIZE = 0.006

class Car():
    def __init__(self, start_x, start_y, start_direction=0):
        self.x = start_x
        self.y = start_y
        # Measured in radians with 0 as directly right
        self.theta = 0

    def location(self):
        return self.x, self.y
    
    def set_location(self, new_x, new_y):
        self.x = new_x
        self.y = new_y

    def direction(self):
        return self.theta
    