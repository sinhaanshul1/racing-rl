class Car():
    def __init__(self, start_x, start_y):
        self.x = start_x
        self.y = start_y
        # Measured in radians with 0 as directly right
        self.theta = 0
        
    def location(self):
        return self.x, self.y
    
    def direction(self):
        return self.theta
    