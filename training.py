import math
from env import RacingEnv
from agent import Agent
from visualize import TrackOverview
import car
import numpy as np
TRACK = "./assets/tracks/track.jpeg"

start = (71, 274)
end = (261, 407)
# end = (90, 263)








if __name__ == '__main__':
    env = RacingEnv(TRACK, start, end)
    agent = Agent(input_dims=env._get_obs().shape, env=env,
            n_actions=1)
    n_games = 3000
    # uncomment this line and do a mkdir tmp && mkdir tmp/video if you want to
    # record video of the agent playing the game.
    #env = wrappers.Monitor(env, 'tmp/video', video_callable=lambda episode_id: True, force=True)


    best_score = - math.inf
    score_history = []
    # load_checkpoint = True

    # if load_checkpoint:
    #     agent.load_models()

    for i in range(n_games):
        print('Game: ', i + 1)
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation)
            action = (action + 1) * math.pi
            observation_, reward, done = env.step(action)
            score += reward
            agent.remember(observation, action, reward, observation_, done)
            # if not load_checkpoint:
            agent.update_agent()
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            # if not load_checkpoint:
            #     agent.save_models()

        print('episode ', i, 'score %.1f' % score, 'avg_score %.1f' % avg_score)
        # if i % 100 == 0:
        TrackOverview(TRACK, env.car).run()

    # if not load_checkpoint:
    #     x = [i+1 for i in range(n_games)]
    #     plot_learning_curve(x, score_history, figure_file)
