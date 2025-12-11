# from visualize import TrackOverview
from track_builder import Track
from car import Car

class RacingEnv:
    def __init__(self, track, start, end):
        self.track_obj = Track(track)

        self.car = Car(*start)
        self._start_x, self._start_y = start
        self._end_x, self._end_y = end
    
    def reset(self):
        height, width = self.track_obj.track().shape
        car_x = self._start_x / width
        car_y = self._start_y / height
        self.car = Car(car_x, car_y)
        return self._get_obs()
    
    def _check_done(self):
        height, width = self.track_obj.track().shape
        car_x, car_y = self.car.location()
        car_row = int(car_y * height)
        car_col = int(car_x * width)
        if abs(self._end_y - car_row) <= 10 and abs(self._end_x - car_col) <= 10:
            return True
        return not self.track_obj.car_in_track(self.car)
    

    def _compute_reward(self):
        reward = -2
        if not self.track_obj.car_in_track(self.car):
            reward -= 10
        height, width = self.track_obj.track().shape
        car_x, car_y = self.car.location()
        car_row = int(car_y * height)
        car_col = int(car_x * width)
        if abs(self._end_y - car_row) <= 10 and abs(self._end_x - car_col) <= 10:
            reward += 10
        return reward
    
    def _get_obs(self):
        track_array = self.track_obj.track()
        height, width = track_array.shape
        car_x, car_y = self.car.location()
        car_row = int(car_y * height)
        car_col = int(car_x * width)
        obs = []
        for i in range(-10, 11):
            current = []
            for j in range(-10, 11):
                if 0 <= car_row + i <= height - 1 and 0 <= car_col + j <= width - 1:
                    current.append(track_array[car_row + i, car_col + j])
                else:
                    current.append(0)
            obs.append(current)
        return obs
    
    def step(self, action):
        self.car.move(action)
        reward = self._compute_reward()
        done = self._check_done()
        return self._get_obs(), reward, done