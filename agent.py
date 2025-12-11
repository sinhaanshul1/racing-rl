import random
import math

class SACAgent:
    def select_action(self, obs: 'state'):
        angle = random.uniform(0, 1)
        return angle