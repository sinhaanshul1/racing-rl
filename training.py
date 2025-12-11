from env import RacingEnv
from agent import SACAgent
from visualize import TrackOverview
import car
TRACK = "./assets/tracks/track.jpeg"

start = (71, 274)
# end = (261, 407)
end = (90, 263)





env = RacingEnv(TRACK, start, end)
agent = SACAgent()

for episode in range(10000):
    obs = env.reset()
    while True:
        action = agent.select_action(obs)
        next_obs, reward, done = env.step(action)

        # agent.replay_buffer.add(obs, action, reward, next_obs, done)
        # agent.train_step()

        obs = next_obs
        if done:
            break
    TrackOverview(TRACK, env.car).run()